from __future__ import annotations

"""Quiz SubAgent: generate 2 multiple-choice questions based on RAG retrieval.

New architecture (Plan & Execute step):
  1. Determine topics:
     - If prior QA context provided → quiz on that topic
     - If no specific topic → read_mastery + search_concepts → pick weakest topics
  2. rag_search per topic → retrieve content as question material
  3. LLM generates 2 multiple-choice questions (4 choices each)
  4. Return questions (grading happens separately in API layer)

Tools used: rag_search, read_mastery, search_concepts
"""

import json
import logging
from dataclasses import dataclass, asdict

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.rag_search import rag_search as _rag_search
from app.agent.tools.read_mastery import read_mastery as _read_mastery
from app.agent.tools.search_concepts import search_concepts as _search_concepts
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────


@dataclass
class QuizQuestion:
    """A single multiple-choice question."""

    topic: str
    question: str
    choices: list[str]  # 4 choices: ["A. ...", "B. ...", "C. ...", "D. ..."]
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str  # brief explanation of correct answer
    source_chunks: list[str]  # source texts used to generate this question


# ── Question generation prompt ──────────────────────────────

GENERATE_QUESTIONS_PROMPT = """\
你是一个出题助手。根据提供的知识库内容，生成 {num_questions} 道选择题（每题 4 个选项）。

## 要求

- 每道题考察对核心概念的理解，而非死记硬背
- 4 个选项中只有 1 个正确，其余 3 个应该具有一定迷惑性
- 题目应基于提供的知识内容，不要超出范围
- 难度适中

## 输出格式

你必须输出一个 JSON 数组（不要输出其他内容）：

```json
[
  {{
    "topic": "该题考察的主题",
    "question": "题目内容",
    "choices": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
    "correct_answer": "A",
    "explanation": "简短解析：为什么正确答案是 A"
  }}
]
```
"""


# ── Topic selection ─────────────────────────────────────────


async def _select_topics_by_mastery(
    session: AsyncSession,
    user_id: str,
    num_topics: int = 3,
) -> list[str]:
    """Select weakest topics based on mastery scores.

    Falls back to search_concepts if no mastery records exist.
    """
    # Get all mastery records
    mastery_records = await _read_mastery(session=session, user_id=user_id, topics=None)

    if mastery_records:
        # Sort by effective_mastery ascending (weakest first)
        sorted_records = sorted(mastery_records, key=lambda r: r.effective_mastery)
        topics = [r.topic for r in sorted_records[:num_topics]]
        logger.info("Selected weakest topics from mastery: %s", topics)
        return topics

    # No mastery records — pick from knowledge_concepts
    concepts = await _search_concepts(session=session, scope=None, keyword=None)
    if concepts:
        topics = [c.topic for c in concepts[:num_topics]]
        logger.info("Selected topics from concepts (no mastery data): %s", topics)
        return topics

    return []


# ── RAG retrieval for question material ─────────────────────


async def _retrieve_question_material(topics: list[str]) -> dict[str, str]:
    """Retrieve knowledge content for each topic via RAG search.

    Returns:
        dict mapping topic → concatenated chunk texts.
    """
    materials: dict[str, str] = {}

    for topic in topics:
        results = _rag_search(query=topic, top_k=3)
        if results:
            chunks_text = "\n\n".join(
                f"[来源: {r.source_doc} | {r.heading_path}]\n{r.text}"
                for r in results
            )
            materials[topic] = chunks_text
        else:
            logger.warning("No RAG results for topic: %s", topic)

    return materials


# ── Question generation ─────────────────────────────────────


def _parse_json(text: str) -> list | dict:
    """Extract and parse JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


async def _generate_questions(
    materials: dict[str, str],
    num_questions: int = 2,
) -> list[QuizQuestion]:
    """Use LLM to generate multiple-choice questions from retrieved materials."""
    # Build material context
    material_text = "\n\n".join(
        f"### 主题: {topic}\n{content}"
        for topic, content in materials.items()
    )

    messages = [
        {
            "role": "system",
            "content": GENERATE_QUESTIONS_PROMPT.format(num_questions=num_questions),
        },
        {
            "role": "user",
            "content": f"请根据以下知识内容出 {num_questions} 道选择题：\n\n{material_text}",
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )

    try:
        raw_questions = _parse_json(response["content"])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse quiz questions: %s", e)
        return []

    questions = []
    for q in raw_questions:
        topic = q.get("topic", "")
        # Find source chunks for this topic
        source_chunks = []
        for t, content in materials.items():
            if t == topic or topic in content:
                source_chunks.append(content)
                break

        questions.append(
            QuizQuestion(
                topic=q.get("topic", ""),
                question=q.get("question", ""),
                choices=q.get("choices", []),
                correct_answer=q.get("correct_answer", ""),
                explanation=q.get("explanation", ""),
                source_chunks=source_chunks,
            )
        )

    return questions


# ── Main entry point ────────────────────────────────────────


async def run_quiz_agent(
    session: AsyncSession,
    user_id: str,
    topics: list[str] | None = None,
    prior_context: str | None = None,
    num_questions: int = 3,
) -> dict:
    """Generate quiz questions as an orchestrator step.

    Args:
        session: Async DB session.
        user_id: User identifier.
        topics: Explicit topics to quiz on (from QA step or user request).
                If None, selects weakest topics from mastery.
        prior_context: Context from prior steps (e.g., QA answer) to inform question generation.
        num_questions: Number of questions to generate (default 3).

    Returns:
        dict with keys:
          - questions: list of QuizQuestion dicts
          - topics: list of topics covered
    """
    # Step 1: Determine topics
    if topics:
        selected_topics = topics[:num_questions]
        logger.info("Using provided topics: %s", selected_topics)
    else:
        selected_topics = await _select_topics_by_mastery(
            session, user_id, num_topics=num_questions
        )

    if not selected_topics:
        logger.warning("No topics available for quiz generation")
        return {
            "questions": [],
            "topics": [],
        }

    # Step 2: Retrieve question material via RAG
    materials = await _retrieve_question_material(selected_topics)

    if not materials:
        logger.warning("No materials retrieved for topics: %s", selected_topics)
        return {
            "questions": [],
            "topics": selected_topics,
        }

    # If prior context provided, append it to materials for richer question generation
    if prior_context:
        for topic in materials:
            materials[topic] += f"\n\n### 补充上下文（来自前序步骤）\n{prior_context}"

    # Step 3: Generate questions
    questions = await _generate_questions(materials, num_questions=num_questions)

    logger.info(
        "Quiz agent generated %d questions for topics: %s",
        len(questions),
        selected_topics,
    )

    return {
        "questions": [asdict(q) for q in questions],
        "topics": selected_topics,
    }
