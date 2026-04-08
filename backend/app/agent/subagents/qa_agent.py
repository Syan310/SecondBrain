from __future__ import annotations

"""QA SubAgent: ReAct (max 2 rounds) + Reflection.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  ReAct Loop (max 2 iterations)                          │
  │    LLM decides tool calls → execute → observe           │
  │    → until LLM produces answer or max iterations        │
  ├─────────────────────────────────────────────────────────┤
  │  Reflection                                             │
  │    Review answer against original query                 │
  │    → pass: return answer                                │
  │    → fail: feed back to ReAct for 1 more round          │
  └─────────────────────────────────────────────────────────┘

Query decomposition is handled by the Orchestrator plan_node.
This agent focuses on single-concept retrieval and answering.

Tools: rag_search
"""

import asyncio
import json
import logging

from app.agent.tools.rag_search import rag_search as _rag_search_sync
from app.services.llm_client import llm_client
from app.services.milvus_client import milvus_service
from app.services.tracing import observe

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
你是一个知识问答助手。你需要基于知识库检索来回答用户问题。

## 可用工具

1. **rag_search(query, top_k)** — 从知识库中检索相关文档片段。

## 工作流程

1. 分析用户问题，调用 rag_search 检索相关内容
2. 基于检索结果生成回答

## 回答要求

- 每个关键论述必须能追溯到检索结果中的具体内容
- 允许对检索内容进行归纳、对比和因果推理
- 禁止引入检索内容中未提及的概念、数据、案例或结论
- 如果检索信息不足以完整回答，明确指出哪些部分无法从知识库中获取
- 篇幅控制在 200 字以内，只保留核心结论和关键依据
"""

REFLECTION_SYSTEM_PROMPT = """\
你是一个答案质量审查器。你需要判断回答是否充分覆盖了用户问题。

## 评估标准

1. **完整性**：是否覆盖了问题的所有方面？
2. **准确性**：回答是否有检索结果支撑，没有编造？
3. **相关性**：回答是否切题？

## 输出格式

你必须输出一个 JSON 对象（不要输出其他内容）：

```json
{
  "passed": true/false,
  "feedback": "如果不通过，指出缺失了什么或哪里不准确（一句话）"
}
```
"""

# ── Tool schema ──────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the knowledge base for relevant document chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ── Tool execution ───────────────────────────────────────────


async def _execute_tool_call(tool_call: dict) -> str:
    """Execute a single tool call and return the result string."""
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    try:
        if name == "rag_search":
            results = await asyncio.to_thread(
                _rag_search_sync, query=args["query"], top_k=args.get("top_k", 5)
            )
            return json.dumps(
                [
                    {
                        "text": r.text,
                        "source_doc": r.source_doc,
                        "heading_path": r.heading_path,
                        "rerank_score": round(r.rerank_score, 4),
                    }
                    for r in results
                ],
                ensure_ascii=False,
            )
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.error("Tool %s execution failed: %s", name, e)
        return json.dumps({"error": str(e)})


# ── ReAct Loop ───────────────────────────────────────────────

MAX_REACT_ITERATIONS = 2


@observe(name="qa_react")
async def _react(query: str, feedback: str | None = None) -> tuple[str, list[str]]:
    """Run ReAct loop: LLM decides tool calls, execute, observe.

    Args:
        query: The user's question.
        feedback: Optional reflection feedback for retry.

    Returns:
        (answer, sources) tuple.
    """
    user_content = query
    if feedback:
        user_content = (
            f"{query}\n\n"
            f"## 上一轮回答的反馈\n{feedback}\n"
            "请根据反馈补充检索并改进回答。"
        )

    messages: list[dict] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    sources: list[str] = []

    for iteration in range(MAX_REACT_ITERATIONS):
        response = await llm_client.chat(
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=2048,
        )

        tool_calls = response.get("tool_calls")

        if not tool_calls:
            answer = response["content"] or ""
            logger.info("ReAct done in %d iterations for: %s", iteration + 1, query[:50])
            return answer, sources

        # Execute tools and append to messages
        messages.append(response)
        for tc in tool_calls:
            result_str = await _execute_tool_call(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                }
            )
            # Track sources
            if tc["function"]["name"] == "rag_search":
                try:
                    chunks = json.loads(result_str)
                    for c in chunks:
                        if c.get("source_doc"):
                            sources.append(c["source_doc"])
                except (json.JSONDecodeError, TypeError):
                    pass

    # Max iterations reached — force a final answer from current context
    logger.warning("ReAct hit max iterations for: %s", query[:50])
    last_answer = response.get("content", "") if response else ""
    return last_answer or "该问题未能在有限步数内完成回答。", sources


# ── Reflection ───────────────────────────────────────────────


@observe(name="qa_reflect")
async def _reflect(query: str, answer: str) -> dict:
    """Reflect on answer quality.

    Returns:
        dict with keys: passed (bool), feedback (str).
    """
    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 用户问题\n{query}\n\n"
                f"## 回答\n{answer}"
            ),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.1,
        max_tokens=256,
    )

    content = response["content"].strip()

    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return {
            "passed": bool(result.get("passed", True)),
            "feedback": result.get("feedback", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Reflection parse failed: %s, treating as passed", e)
        return {"passed": True, "feedback": ""}


# ── Main entry ────────────────────────────────────────────────


@observe(name="qa_agent")
async def run_qa_agent(query: str, user_id: str = "default") -> dict:
    """Run the QA SubAgent with ReAct + Reflection.

    Flow:
      ReAct (max 2 rounds) → Reflection
        → pass: return
        → fail: ReAct 1 more round with feedback → return

    Args:
        query: The user's question.
        user_id: User identifier.

    Returns:
        dict with keys:
          - answer (str): The final answer.
          - confidence ("high"|"low"): Whether reflection passed.
          - sources (list[str]): Source documents referenced.
    """
    # Check if knowledge base has documents
    try:
        doc_count = milvus_service.count()
    except Exception:
        doc_count = 0

    if doc_count == 0:
        logger.info("QA agent: knowledge base is empty, falling back to plain LLM.")
        response = await llm_client.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一个知识问答助手。当前知识库为空，用户尚未上传任何文档。"
                    "请基于你的通用知识回答用户问题，并在回答开头提示用户：\n"
                    "「📚 当前知识库为空，以下回答基于通用知识。上传文档后可获得更精准的回答。」"
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return {
            "answer": response["content"] or "",
            "confidence": "low",
            "sources": [],
        }

    all_sources: list[str] = []

    # Phase 1: ReAct
    answer, sources = await _react(query)
    all_sources.extend(sources)

    # Phase 2: Reflection
    reflection = await _reflect(query, answer)

    if reflection["passed"]:
        logger.info("QA agent: reflection passed for: %s", query[:50])
        return {
            "answer": answer,
            "confidence": "high",
            "sources": list(set(all_sources)),
        }

    # Phase 3: ReAct retry with feedback
    logger.info("QA agent: reflection failed, retrying with feedback: %s", reflection["feedback"])
    answer, sources = await _react(query, feedback=reflection["feedback"])
    all_sources.extend(sources)

    return {
        "answer": answer,
        "confidence": "low",
        "sources": list(set(all_sources)),
    }
