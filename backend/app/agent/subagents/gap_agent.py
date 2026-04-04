"""Gap Analysis SubAgent: knowledge gap detection and diagnosis.

Flow (agent decides tool calls):
  1. read_mastery → get all user mastery records
  2. search_concepts → get all knowledge base topics
  3. Cross compare (agent reasoning):
     - In concepts but no mastery record → "未接触盲区" (high priority)
     - effective_mastery < 0.3 → "薄弱盲区" (high priority)
     - effective_mastery 0.3~0.5 → "需巩固" (medium priority)
     - effective_mastery >= 0.5 → "掌握中/已掌握" (low priority)
  4. Generate diagnosis report with recommended review order

Tools: read_mastery, search_concepts
"""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.read_mastery import read_mastery as _read_mastery
from app.agent.tools.search_concepts import search_concepts as _search_concepts
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────

GAP_SYSTEM_PROMPT = """\
你是一个知识盲区诊断助手（Gap Analysis Agent）。你的任务是分析用户的知识掌握情况，找出薄弱环节，并给出复习建议。

## 可用工具

1. **read_mastery(user_id, topics)** — 读取用户的知识掌握度（含时间衰减）。topics 为可选过滤。
2. **search_concepts(scope, keyword)** — 查询知识库中的概念列表。scope 按来源文档过滤，keyword 按关键词过滤。均可选。

## 工作流程

1. 调用 `search_concepts` 获取知识库中的所有概念
2. 调用 `read_mastery` 获取用户的所有掌握度记录
3. 交叉对比，按以下规则分类：

   | 情况 | 判定 | 优先级 |
   |------|------|--------|
   | 知识库有，但用户无掌握记录 | **未接触盲区** | 🔴 高 |
   | 有记录，effective_mastery < 0.3 | **薄弱盲区** | 🔴 高 |
   | 有记录，effective_mastery 0.3~0.5 | **需巩固** | 🟡 中 |
   | effective_mastery ≥ 0.5 | **掌握中/已掌握** | 🟢 低 |

4. 生成诊断报告：
   - 按优先级从高到低列出所有主题
   - 给出推荐复习顺序（优先复习高优先级主题）
   - 附上整体掌握度概览（已掌握/需巩固/盲区 的比例）

## 输出格式

生成结构化的诊断报告，包含：
- 📊 整体概览（各类别数量和占比）
- 🔴 高优先级（未接触盲区 + 薄弱盲区）
- 🟡 中优先级（需巩固）
- 🟢 已掌握
- 📋 推荐复习顺序

如果用户指定了范围（如某个文档），只分析该范围内的知识点。
"""

# ── Tool schema for LLM function calling ─────────────────────

GAP_TOOL_SCHEMAS = [
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
                        "description": "Optional list of topics to filter.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_concepts",
            "description": "Query knowledge base concept index to discover all available topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Filter by source document name (exact match). Optional.",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Filter by topic keyword (case-insensitive substring). Optional.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Tool execution ───────────────────────────────────────────


def _build_tool_executor(session: AsyncSession, user_id: str):
    """Build a tool executor closure with bound DB session."""

    async def execute(tool_call: dict) -> str:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        try:
            if name == "read_mastery":
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
            logger.error("Gap tool %s execution failed: %s", name, e)
            return json.dumps({"error": str(e)})

    return execute


# ── Agent execution ──────────────────────────────────────────

MAX_ITERATIONS = 10


async def run_gap_agent(
    session: AsyncSession,
    user_id: str,
    scope: str | None = None,
) -> dict:
    """Run the Gap Analysis SubAgent.

    Args:
        session: Async DB session for mastery/concepts queries.
        user_id: User identifier.
        scope: Optional source_doc filter to limit analysis scope.

    Returns:
        dict with keys:
          - report (str): The diagnosis report text.
    """
    execute_tool = _build_tool_executor(session, user_id)

    user_prompt = "请对我的知识掌握情况进行全面诊断分析。"
    if scope:
        user_prompt += f"只分析来源文档为「{scope}」的知识点。"

    messages = [
        {"role": "system", "content": GAP_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"当前用户 ID: {user_id}",
        },
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(MAX_ITERATIONS):
        response = await llm_client.chat(
            messages=messages,
            tools=GAP_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=4096,
        )

        tool_calls = response.get("tool_calls")

        if not tool_calls:
            report = response["content"] or ""
            logger.info(
                "Gap agent finished in %d iterations for user: %s",
                iteration + 1,
                user_id,
            )
            return {"report": report}

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
    logger.warning("Gap agent hit max iterations (%d)", MAX_ITERATIONS)
    return {"report": "诊断分析未能在有限步骤内完成，请重试。"}
