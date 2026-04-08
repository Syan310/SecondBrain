from __future__ import annotations

"""Gap Analysis SubAgent: knowledge gap detection and diagnosis.

Architecture (Python classification + LLM report):
  1. search_concepts → get all knowledge base topics (Python)
  2. read_mastery → get all user mastery records (Python)
  3. Deterministic classification (Python):
     - In concepts but no mastery record → "未接触盲区" (high priority)
     - effective_mastery < 0.3 → "薄弱盲区" (high priority)
     - effective_mastery 0.3~0.5 → "需巩固" (medium priority)
     - effective_mastery >= 0.5 → "掌握中/已掌握" (low priority)
  4. LLM generates diagnosis report from classified data

Tools used: read_mastery, search_concepts (called directly, not via LLM)
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.read_mastery import read_mastery as _read_mastery
from app.agent.tools.search_concepts import search_concepts as _search_concepts
from app.services.llm_client import llm_client
from app.services.tracing import observe

logger = logging.getLogger(__name__)

# ── Report generation prompt ────────────────────────────────

GAP_REPORT_PROMPT = """\
你是一个知识盲区诊断助手。请根据以下分类数据生成诊断报告。

## 输出格式

- 📊 整体概览（各类别数量和占比）
- 🔴 高优先级（未接触盲区 + 薄弱盲区）— 列出具体主题
- 🟡 中优先级（需巩固）— 列出具体主题
- 🟢 已掌握 — 概括即可，不需逐一列出
- 📋 推荐复习顺序（优先复习高优先级主题）

如果某个类别为空，简要说明即可。
"""


# ── Deterministic classification ──────────────────────────────


@observe(name="gap_classify")
async def _classify_topics(
    session: AsyncSession,
    user_id: str,
    scope: str | None = None,
    keyword: str | None = None,
) -> tuple[dict[str, list], int]:
    """Classify all topics by mastery level using deterministic rules.

    Returns:
        (categories dict, total concept count)
    """
    concepts = await _search_concepts(session=session, scope=scope, keyword=keyword)
    mastery_records = await _read_mastery(session=session, user_id=user_id, topics=None)

    mastery_map = {r.topic: r for r in mastery_records}

    categories: dict[str, list] = {
        "未接触盲区": [],
        "薄弱盲区": [],
        "需巩固": [],
        "已掌握": [],
    }

    for concept in concepts:
        topic = concept.topic
        record = mastery_map.get(topic)
        if record is None:
            categories["未接触盲区"].append(topic)
        elif record.effective_mastery < 0.3:
            categories["薄弱盲区"].append(
                {"topic": topic, "mastery": round(record.effective_mastery, 3)}
            )
        elif record.effective_mastery < 0.5:
            categories["需巩固"].append(
                {"topic": topic, "mastery": round(record.effective_mastery, 3)}
            )
        else:
            categories["已掌握"].append(
                {"topic": topic, "mastery": round(record.effective_mastery, 3)}
            )

    return categories, len(concepts)


# ── Report generation ─────────────────────────────────────────


@observe(name="gap_report")
async def _generate_report(
    categories: dict[str, list],
    total: int,
    scope: str | None = None,
    keyword: str | None = None,
) -> str:
    """Generate a human-readable diagnosis report from classified data."""
    lines: list[str] = []

    lines.append("## 分类统计")
    lines.append(f"- 知识点总数: {total}")
    for cat_name in ("未接触盲区", "薄弱盲区", "需巩固", "已掌握"):
        count = len(categories[cat_name])
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
        lines.append(f"- {cat_name}: {count} ({pct})")

    for cat_name in ("未接触盲区", "薄弱盲区", "需巩固", "已掌握"):
        items = categories[cat_name]
        if not items:
            continue
        lines.append(f"\n## {cat_name}")
        if cat_name == "未接触盲区":
            lines.append(", ".join(items))
        else:
            lines.append(
                ", ".join(
                    f"{item['topic']}({item['mastery']:.2f})" for item in items
                )
            )

    scope_desc = ""
    if scope:
        scope_desc += f"分析范围: 来源文档「{scope}」\n"
    if keyword:
        scope_desc += f"过滤关键词: 「{keyword}」\n"

    messages = [
        {"role": "system", "content": GAP_REPORT_PROMPT},
        {
            "role": "user",
            "content": f"{scope_desc}请根据以下数据生成诊断报告：\n\n" + "\n".join(lines),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    return response["content"] or ""


# ── Main entry point ──────────────────────────────────────────


@observe(name="gap_agent")
async def run_gap_agent(
    session: AsyncSession,
    user_id: str,
    scope: str | None = None,
    keyword: str | None = None,
) -> dict:
    """Run the Gap Analysis SubAgent.

    Architecture: Python deterministic classification + LLM report generation.
    Classification rules are hardcoded to avoid LLM reasoning over large topic sets.

    Args:
        session: Async DB session for mastery/concepts queries.
        user_id: User identifier.
        scope: Optional source_doc filter to limit analysis scope.
        keyword: Optional keyword filter to limit analysis to matching topics.

    Returns:
        dict with keys:
          - report (str): The diagnosis report text.
    """
    categories, total = await _classify_topics(session, user_id, scope, keyword)

    if total == 0:
        logger.info("Gap agent: no concepts found for scope=%s keyword=%s", scope, keyword)
        return {"report": "知识库中未找到相关概念，无法进行盲区分析。"}

    logger.info(
        "Gap agent classified %d topics: 未接触=%d, 薄弱=%d, 需巩固=%d, 已掌握=%d",
        total,
        len(categories["未接触盲区"]),
        len(categories["薄弱盲区"]),
        len(categories["需巩固"]),
        len(categories["已掌握"]),
    )

    report = await _generate_report(categories, total, scope, keyword)
    return {"report": report}
