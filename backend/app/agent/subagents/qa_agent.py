"""QA SubAgent: Plan & Execute + ReAct hybrid for knowledge Q&A.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 1: Planner (LLM, no tools)                      │
  │    Analyze query → output structured plan (JSON)        │
  │    → writes plan to state                               │
  ├─────────────────────────────────────────────────────────┤
  │  Phase 2: Executor (ReAct loop, with tools)             │
  │    For each step in plan:                               │
  │      → LLM decides tool calls → execute → observe      │
  │    → collects step results into state                   │
  ├─────────────────────────────────────────────────────────┤
  │  Phase 3: QC (LLM, no tools)                            │
  │    Review final answer against original query           │
  │    → pass: return answer                                │
  │    → fail (retries < 2): feedback → back to Planner     │
  │    → fail (retries >= 2): return with low confidence    │
  └─────────────────────────────────────────────────────────┘

Tools: rag_search, query_decompose
"""

import json
import logging

from app.agent.tools.rag_search import rag_search as _rag_search
from app.agent.tools.query_decompose import query_decompose as _query_decompose
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
你是一个知识问答规划器（QA Planner）。你的任务是分析用户问题，制定一个检索和回答的计划。

## 分析维度

1. **问题复杂度**：是否涉及多个概念、对比、因果链？
2. **是否需要拆解**：复杂问题需要拆成多个子查询分别检索
3. **每个子查询是否需要 RAG**：大多数情况需要检索知识库，但纯常识性问题可以直接回答

## 输出格式

你必须输出一个 JSON 对象，格式如下（不要输出其他内容）：

```json
{
  "needs_decompose": true/false,
  "plan": [
    {
      "step": 1,
      "query": "子查询或原始查询",
      "needs_rag": true/false,
      "purpose": "这一步要获取什么信息"
    }
  ],
  "synthesis_needed": true/false
}
```

- `needs_decompose`: 是否需要拆解
- `plan`: 执行步骤列表（1-4步）
- `synthesis_needed`: 是否需要最后综合多个子答案（多步时为 true）

## 示例

输入: "Transformer中self-attention和cross-attention的区别?"
输出:
```json
{
  "needs_decompose": true,
  "plan": [
    {"step": 1, "query": "Transformer self-attention 工作原理", "needs_rag": true, "purpose": "获取self-attention的定义和机制"},
    {"step": 2, "query": "Transformer cross-attention 工作原理", "needs_rag": true, "purpose": "获取cross-attention的定义和机制"}
  ],
  "synthesis_needed": true
}
```

输入: "什么是BPE分词?"
输出:
```json
{
  "needs_decompose": false,
  "plan": [
    {"step": 1, "query": "BPE分词算法原理", "needs_rag": true, "purpose": "检索BPE的定义和工作原理"}
  ],
  "synthesis_needed": false
}
```
"""

EXECUTOR_SYSTEM_PROMPT = """\
你是一个知识问答执行器（QA Executor）。你正在按照计划逐步执行检索和回答。

## 可用工具

1. **rag_search(query, top_k)** — 从知识库中检索相关文档片段。
2. **query_decompose(original_query)** — 将复杂问题拆解为子查询（仅在 planner 未拆解时备用）。

## 当前任务

你会收到一个具体的执行步骤。请：
1. 如果该步骤需要 RAG，调用 `rag_search` 检索相关内容
2. 基于检索结果（或你的知识），回答该步骤的查询
3. 输出该步骤的答案，要忠于检索到的内容

## 输出要求

- 回答必须基于检索到的文档内容，不要编造
- 如果检索不到相关信息，明确说明
- 保持回答简洁聚焦，只回答当前步骤的问题
"""

SYNTHESIZER_PROMPT = """\
你是一个知识问答综合器。你需要将多个子问题的答案综合成一个完整、连贯的最终回答。

## 输入

- 原始问题
- 各子问题及其答案

## 要求

- 综合所有子答案，形成一个结构清晰的完整回答
- 保持逻辑连贯，消除重复
- 适当引用来源文档
- 使用与原始问题相同的语言
"""

QC_SYSTEM_PROMPT = """\
你是一个答案质量检查器（QC）。你需要评估最终答案是否充分回答了用户的原始问题。

## 评估维度

1. **完整性**：是否覆盖了问题的所有方面？
2. **准确性**：答案是否有检索结果支撑，没有编造？
3. **相关性**：答案是否切题，没有偏离？

## 输出格式

你必须输出一个 JSON 对象（不要输出其他内容）：

```json
{
  "passed": true/false,
  "reason": "通过/不通过的原因",
  "suggestions": "如果不通过，给出改进建议（重新规划时会用到）"
}
```
"""

# ── Tool schemas for Executor ────────────────────────────────

EXECUTOR_TOOL_SCHEMAS = [
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
    {
        "type": "function",
        "function": {
            "name": "query_decompose",
            "description": "Decompose a complex query into 2-4 simpler sub-queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_query": {
                        "type": "string",
                        "description": "The complex user question to decompose.",
                    },
                },
                "required": ["original_query"],
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
            results = _rag_search(query=args["query"], top_k=args.get("top_k", 5))
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
        elif name == "query_decompose":
            sub_queries = await _query_decompose(args["original_query"])
            return json.dumps(sub_queries, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.error("Tool %s execution failed: %s", name, e)
        return json.dumps({"error": str(e)})


# ── Phase 1: Planner ─────────────────────────────────────────


async def _plan(query: str, feedback: str | None = None) -> dict:
    """Generate an execution plan for the query.

    Args:
        query: The user's question.
        feedback: Optional QC feedback for re-planning.

    Returns:
        Parsed plan dict with keys: needs_decompose, plan, synthesis_needed.
    """
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
    ]

    if feedback:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"原始问题: {query}\n\n"
                    f"上一轮的答案未通过质量检查，反馈如下:\n{feedback}\n\n"
                    "请根据反馈重新制定计划。"
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": query})

    response = await llm_client.chat(
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    content = response["content"].strip()

    # Parse JSON plan
    try:
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        plan = json.loads(content)

        # Validate structure
        if "plan" not in plan or not isinstance(plan["plan"], list):
            raise ValueError("Missing or invalid 'plan' field")

        logger.info(
            "Planner: %d steps, decompose=%s, synthesis=%s",
            len(plan["plan"]),
            plan.get("needs_decompose"),
            plan.get("synthesis_needed"),
        )
        return plan

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Planner output parse failed: %s, content: %s", e, content[:200])
        # Fallback: single-step plan
        return {
            "needs_decompose": False,
            "plan": [
                {"step": 1, "query": query, "needs_rag": True, "purpose": "检索并回答问题"}
            ],
            "synthesis_needed": False,
        }


# ── Phase 2: Executor (ReAct loop per step) ──────────────────

MAX_REACT_ITERATIONS = 8  # per step


async def _execute_step(step: dict, original_query: str) -> dict:
    """Execute a single plan step using ReAct loop.

    Args:
        step: A plan step dict with query, needs_rag, purpose.
        original_query: The original user question for context.

    Returns:
        dict with keys: step, query, answer, sources.
    """
    step_query = step["query"]
    needs_rag = step.get("needs_rag", True)
    purpose = step.get("purpose", "")

    user_prompt = (
        f"原始问题: {original_query}\n\n"
        f"当前步骤: {step_query}\n"
        f"目的: {purpose}\n"
        f"是否需要检索: {'是' if needs_rag else '否，可直接回答'}"
    )

    messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    sources = []

    for iteration in range(MAX_REACT_ITERATIONS):
        response = await llm_client.chat(
            messages=messages,
            tools=EXECUTOR_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=2048,
        )

        tool_calls = response.get("tool_calls")

        if not tool_calls:
            # Executor produced an answer for this step
            answer = response["content"] or ""
            logger.info(
                "Executor step %d done in %d iterations: %s",
                step.get("step", 0),
                iteration + 1,
                step_query[:50],
            )
            return {
                "step": step.get("step", 0),
                "query": step_query,
                "answer": answer,
                "sources": sources,
            }

        # Execute tools and collect sources
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
            # Track sources from rag_search
            if tc["function"]["name"] == "rag_search":
                try:
                    chunks = json.loads(result_str)
                    for c in chunks:
                        if c.get("source_doc"):
                            sources.append(c["source_doc"])
                except (json.JSONDecodeError, TypeError):
                    pass

    # Safety: max iterations
    logger.warning("Executor step hit max iterations for: %s", step_query[:50])
    return {
        "step": step.get("step", 0),
        "query": step_query,
        "answer": "该步骤未能在有限步数内完成。",
        "sources": sources,
    }


async def _execute_plan(plan: dict, original_query: str) -> list[dict]:
    """Execute all steps in the plan sequentially.

    Returns:
        List of step result dicts.
    """
    step_results = []
    for step in plan["plan"]:
        result = await _execute_step(step, original_query)
        step_results.append(result)
    return step_results


# ── Synthesis ─────────────────────────────────────────────────


async def _synthesize(original_query: str, step_results: list[dict]) -> str:
    """Synthesize multiple step answers into a final answer."""
    if len(step_results) == 1:
        return step_results[0]["answer"]

    sub_answers_text = "\n\n".join(
        f"### 子问题 {r['step']}: {r['query']}\n{r['answer']}"
        for r in step_results
    )

    messages = [
        {"role": "system", "content": SYNTHESIZER_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 原始问题\n{original_query}\n\n"
                f"## 各子问题答案\n{sub_answers_text}"
            ),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    return response["content"] or ""


# ── Phase 3: QC ──────────────────────────────────────────────


async def _quality_check(original_query: str, answer: str) -> dict:
    """Check answer quality.

    Returns:
        dict with keys: passed (bool), reason (str), suggestions (str).
    """
    messages = [
        {"role": "system", "content": QC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 原始问题\n{original_query}\n\n"
                f"## 最终答案\n{answer}"
            ),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.1,
        max_tokens=512,
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
            "reason": result.get("reason", ""),
            "suggestions": result.get("suggestions", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("QC output parse failed: %s, treating as passed", e)
        return {"passed": True, "reason": "QC parse failed, default pass", "suggestions": ""}


# ── Main entry ────────────────────────────────────────────────

MAX_QC_RETRIES = 2


async def run_qa_agent(query: str, user_id: str = "default") -> dict:
    """Run the QA SubAgent with Plan & Execute + ReAct.

    Flow:
      Planner → Executor (ReAct per step) → Synthesize → QC
      → pass: return | fail: re-plan (up to 2 retries)

    Args:
        query: The user's question.
        user_id: User identifier.

    Returns:
        dict with keys:
          - answer (str): The final answer.
          - confidence ("high"|"low"): Whether QC passed.
          - plan (dict): The execution plan used.
          - sources (list[str]): Source documents referenced.
    """
    feedback = None
    all_sources = []

    for retry in range(MAX_QC_RETRIES + 1):
        # Phase 1: Plan
        plan = await _plan(query, feedback=feedback)
        logger.info("QA attempt %d/%d: plan has %d steps", retry + 1, MAX_QC_RETRIES + 1, len(plan["plan"]))

        # Phase 2: Execute
        step_results = await _execute_plan(plan, query)

        # Collect sources
        for r in step_results:
            all_sources.extend(r.get("sources", []))

        # Synthesize if needed
        if plan.get("synthesis_needed") and len(step_results) > 1:
            answer = await _synthesize(query, step_results)
        else:
            answer = step_results[-1]["answer"] if step_results else ""

        # Phase 3: QC
        qc_result = await _quality_check(query, answer)

        if qc_result["passed"]:
            logger.info("QA agent: QC passed on attempt %d", retry + 1)
            return {
                "answer": answer,
                "confidence": "high",
                "plan": plan,
                "sources": list(set(all_sources)),
            }

        # QC failed — prepare feedback for re-planning
        feedback = f"原因: {qc_result['reason']}\n建议: {qc_result['suggestions']}"
        logger.info("QA agent: QC failed on attempt %d, reason: %s", retry + 1, qc_result["reason"])

    # All retries exhausted
    logger.warning("QA agent: QC failed after %d attempts, returning with low confidence", MAX_QC_RETRIES + 1)
    return {
        "answer": answer,
        "confidence": "low",
        "plan": plan,
        "sources": list(set(all_sources)),
    }
