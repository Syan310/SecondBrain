"""Quiz SubAgent: agent-driven 10-question adaptive quiz.

Flow (agent decides tool calls):
  1. read_mastery → get user's current mastery levels
  2. If no target topics → search_concepts → weighted random select by (1 - effective_mastery)
  3. Loop 10 rounds:
     a. rag_search(topic) → get content as question material
     b. LLM generates question based on retrieved content
     c. Wait for user answer (via message)
     d. LLM evaluates answer quality (high/medium/low)
     e. update_mastery(topic, quality) → high: +0.1, medium: +0.05, low: -0.15
  4. Return session summary

Tools: rag_search, read_mastery, update_mastery, search_concepts
"""

import json
import logging

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.rag_search import rag_search as _rag_search
from app.agent.tools.read_mastery import read_mastery as _read_mastery
from app.agent.tools.update_mastery import update_mastery as _update_mastery
from app.agent.tools.search_concepts import search_concepts as _search_concepts
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────

QUIZ_SYSTEM_PROMPT = """\
你是一个自适应复习助手（Quiz Agent）。你的任务是根据用户的知识掌握情况出题测试，帮助用户巩固知识。

## 可用工具

1. **rag_search(query, top_k)** — 从知识库检索相关文档片段，作为出题素材。
2. **read_mastery(user_id, topics)** — 读取用户的知识掌握度（含时间衰减）。topics 为可选的 topic 列表过滤。
3. **update_mastery(user_id, topic, is_correct)** — 更新用户掌握度。答对 +0.1，答错 -0.15。
4. **search_concepts(scope, keyword)** — 查询知识库中的概念列表。scope 按来源文档过滤，keyword 按主题关键词过滤。均为可选参数。

## 工作流程

1. **准备阶段**：
   - 调用 `read_mastery` 获取用户当前掌握度
   - 如果用户指定了复习主题，使用该主题
   - 如果没有指定主题，调用 `search_concepts` 获取知识库全部主题，优先选择 effective_mastery 低的主题

2. **出题阶段**（共 10 题）：
   - 选择一个主题，调用 `rag_search` 获取该主题的内容
   - 基于检索内容出一道问答题（简答题或概念解释题）
   - 题目应该考察理解而非死记硬背
   - 明确标注这是第几题（如"第 3/10 题"）

3. **判分阶段**：
   - 收到用户回答后，根据检索到的知识内容评估回答质量
   - 将回答分为三个质量等级，并调用 `update_mastery` 更新掌握度：
     - **高质量 (high)**：核心要点准确且表述完整 → mastery +0.1
     - **中等质量 (medium)**：方向正确但不完整，或部分准确 → mastery +0.05
     - **低质量 (low)**：完全答错、答非所问、严重偏差 → mastery -0.15

4. **结束阶段**：
   - 10 题答完后，给出总结：各质量等级分布、薄弱主题、进步建议

## 出题要求

- 题目类型：简答题、概念解释、对比分析
- 难度适中，聚焦核心概念
- 每题必须基于知识库中的实际内容
- 不同题目尽量覆盖不同主题

## 判分要求

- 语义判分：用户不必逐字匹配，考察理解程度
- 按三档评估，给出明确判定和质量标签：
  - 🟢 **高质量 (high)** — 核心要点准确，表述完整清晰
  - 🟡 **中等质量 (medium)** — 理解方向正确，但有遗漏或不够精确
  - 🔴 **低质量 (low)** — 答错、答非所问、或存在严重理解偏差
- 附上简短解析（指出回答的优缺点 + 正确答案要点）
"""

# ── Tool schema for LLM function calling ─────────────────────

QUIZ_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the knowledge base for relevant document chunks as question material.",
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
            "name": "read_mastery",
            "description": "Read user's knowledge mastery levels with time decay applied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of topics to filter. Omit to get all topics.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_mastery",
            "description": "Update mastery score after quiz answer. high: +0.1, medium: +0.05, low: -0.15.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic being tested.",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Answer quality: 'high' (accurate & complete), 'medium' (partially correct), 'low' (wrong or off-topic).",
                    },
                },
                "required": ["user_id", "topic", "quality"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_concepts",
            "description": "Query knowledge base concept index. Use to discover available topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Filter by source document name (exact match). Optional.",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Filter by topic keyword (case-insensitive substring match). Optional.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Tool execution ───────────────────────────────────────────


def _build_tool_executor(session: AsyncSession, user_id: str):
    """Build a tool executor closure with bound DB session.

    Returns an async function that takes a tool_call dict and returns result string.
    """

    async def execute(tool_call: dict) -> str:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        try:
            if name == "rag_search":
                results = _rag_search(
                    query=args["query"],
                    top_k=args.get("top_k", 5),
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

            elif name == "read_mastery":
                records = await _read_mastery(
                    session=session,
                    user_id=args.get("user_id", user_id),
                    topics=args.get("topics"),
                )
                return json.dumps(
                    [
                        {
                            "topic": r.topic,
                            "effective_mastery": r.effective_mastery,
                            "mastery_score": r.mastery_score,
                            "days_since_review": r.days_since_review,
                            "total_attempts": r.total_attempts,
                        }
                        for r in records
                    ],
                    ensure_ascii=False,
                )

            elif name == "update_mastery":
                result = await _update_mastery(
                    session=session,
                    user_id=args.get("user_id", user_id),
                    topic=args["topic"],
                    quality=args["quality"],
                )
                return json.dumps(
                    {
                        "topic": result.topic,
                        "old_effective_mastery": result.old_effective_mastery,
                        "new_mastery_score": result.new_mastery_score,
                        "quality": result.quality,
                    },
                    ensure_ascii=False,
                )

            elif name == "search_concepts":
                records = await _search_concepts(
                    session=session,
                    scope=args.get("scope"),
                    keyword=args.get("keyword"),
                )
                return json.dumps(
                    [
                        {
                            "topic": r.topic,
                            "source_doc": r.source_doc,
                            "heading_path": r.heading_path,
                        }
                        for r in records
                    ],
                    ensure_ascii=False,
                )

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            logger.error("Quiz tool %s execution failed: %s", name, e)
            return json.dumps({"error": str(e)})

    return execute


# ── Agent execution ──────────────────────────────────────────

MAX_ITERATIONS = 30  # quiz has more back-and-forth (10 questions × tool calls)


async def run_quiz_agent(
    session: AsyncSession,
    user_id: str,
    user_message: str,
    message_history: list[dict] | None = None,
) -> dict:
    """Run the Quiz SubAgent for one turn of interaction.

    This function handles a single turn: the agent may call tools and produce
    a response (either a question or evaluation). For multi-turn quiz sessions,
    the caller should maintain message_history across calls.

    Args:
        session: Async DB session for mastery read/write.
        user_id: User identifier.
        user_message: The user's current message (e.g., "开始复习" or an answer).
        message_history: Previous conversation messages. If None, starts a new session.

    Returns:
        dict with keys:
          - response (str): The agent's response text.
          - message_history (list): Updated message history for next turn.
          - is_complete (bool): Whether the quiz session is finished.
    """
    execute_tool = _build_tool_executor(session, user_id)

    if message_history is None:
        messages = [
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"当前用户 ID: {user_id}。请开始 10 题复习流程。",
            },
        ]
    else:
        messages = list(message_history)

    # Add user message
    messages.append({"role": "user", "content": user_message})

    # ReAct loop: LLM reasons, calls tools, until it produces a text response
    for iteration in range(MAX_ITERATIONS):
        response = await llm_client.chat(
            messages=messages,
            tools=QUIZ_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.5,
            max_tokens=4096,
        )

        tool_calls = response.get("tool_calls")

        if not tool_calls:
            # LLM produced a text response (question or evaluation)
            agent_response = response["content"] or ""
            messages.append({"role": "assistant", "content": agent_response})

            # Heuristic: check if quiz is complete
            is_complete = _check_quiz_complete(agent_response)

            logger.info(
                "Quiz agent responded in %d iterations, complete=%s",
                iteration + 1,
                is_complete,
            )
            return {
                "response": agent_response,
                "message_history": messages,
                "is_complete": is_complete,
            }

        # Execute tool calls
        messages.append(response)
        for tc in tool_calls:
            result_str = await execute_tool(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                }
            )

    # Safety limit
    logger.warning("Quiz agent hit max iterations (%d)", MAX_ITERATIONS)
    return {
        "response": "抱歉，出题流程出现异常，请重新开始。",
        "message_history": messages,
        "is_complete": True,
    }


def _check_quiz_complete(response: str) -> bool:
    """Heuristic check if the quiz session is complete."""
    completion_signals = ["10/10", "总结", "复习结束", "本次测试", "答题结束", "总成绩"]
    return any(signal in response for signal in completion_signals)
