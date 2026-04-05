"""Semi-automatic evaluation dataset generator.

Reads chunks from Milvus, uses Qwen LLM to generate candidate QA pairs,
intent queries, and quiz/gap scenarios. Outputs JSON files for human review.

Usage:
    cd backend
    python -m evaluation.generate_dataset

Requires Milvus and LLM API to be available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from pathlib import Path

from app.config import settings
from app.services.llm_client import llm_client
from app.services.milvus_client import milvus_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "datasets"


# ── Milvus chunk extraction ────────────────────────────────────


def fetch_all_chunks() -> list[dict]:
    """Query all chunks from Milvus, return as list of dicts."""
    col = milvus_service.collection
    # Paginated query to get all chunks (no vector needed)
    all_chunks = []
    offset = 0
    batch_size = 200
    while True:
        results = col.query(
            expr="child_id != ''",
            output_fields=["child_id", "parent_id", "source_doc", "heading_path", "chunk_index", "text"],
            limit=batch_size,
            offset=offset,
        )
        if not results:
            break
        all_chunks.extend(results)
        if len(results) < batch_size:
            break
        offset += batch_size

    logger.info("Fetched %d chunks from Milvus.", len(all_chunks))
    return all_chunks


def group_by_source(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group chunks by source_doc."""
    groups: dict[str, list[dict]] = {}
    for c in chunks:
        src = c.get("source_doc", "unknown")
        groups.setdefault(src, []).append(c)
    for src in groups:
        groups[src].sort(key=lambda x: (x.get("heading_path", ""), x.get("chunk_index", 0)))
    return groups


# ── QA dataset generation ──────────────────────────────────────


SINGLE_HOP_PROMPT = """\
你是一个评估数据集生成器。根据以下文档片段，生成{n}个单跳问答对。

要求：
1. 问题应该只需要这些片段中的信息就能回答
2. 问题类型多样化（事实型、概念解释型、对比型、应用型）
3. ground_truth 答案必须准确，直接基于文档内容
4. 每个问答对注明来源文档

文档片段（来源: {source_doc}）：
{chunks_text}

输出格式（JSON数组）：
[
  {{
    "question": "问题文本",
    "ground_truth": "标准答案",
    "source_doc": "{source_doc}",
    "question_type": "factual|conceptual|comparison|application",
    "difficulty": "easy|medium|hard"
  }}
]

只输出JSON数组，不要其他文字。"""

MULTI_HOP_PROMPT = """\
你是一个评估数据集生成器。根据以下来自不同文档的片段，生成{n}个多跳问答对。

要求：
1. 每个问题必须需要综合多个片段的信息才能回答（不能只看一个片段）
2. 问题应涉及跨文档/跨章节的对比、关联、因果推理
3. ground_truth 答案必须综合多个片段的信息
4. 标注所有涉及的来源文档

文档片段：
{chunks_text}

输出格式（JSON数组）：
[
  {{
    "question": "需要跨文档/跨概念综合才能回答的问题",
    "ground_truth": "综合多个来源的标准答案",
    "source_docs": ["来源1", "来源2"],
    "question_type": "multi-hop",
    "difficulty": "hard"
  }}
]

只输出JSON数组，不要其他文字。"""


async def generate_single_hop_qa(
    source_doc: str, chunks: list[dict], n: int = 5
) -> list[dict]:
    """Generate single-hop QA pairs from one source doc's chunks."""
    # Sample chunks if too many (keep context manageable)
    sampled = chunks if len(chunks) <= 15 else random.sample(chunks, 15)
    chunks_text = "\n\n".join(
        f"[{c['heading_path']}] {c['text']}" for c in sampled
    )

    prompt = SINGLE_HOP_PROMPT.format(
        n=n, source_doc=source_doc, chunks_text=chunks_text
    )
    resp = await llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
    )
    return _parse_json_array(resp.get("content", ""), source_doc)


async def generate_multi_hop_qa(
    grouped: dict[str, list[dict]], n: int = 10
) -> list[dict]:
    """Generate multi-hop QA pairs across multiple source docs."""
    # Pick chunks from different sources
    all_sources = list(grouped.keys())
    if len(all_sources) < 2:
        logger.warning("Need at least 2 source docs for multi-hop, got %d.", len(all_sources))
        # Still generate with different sections of the same doc
        src = all_sources[0]
        sampled = random.sample(grouped[src], min(20, len(grouped[src])))
    else:
        sampled = []
        for src in all_sources:
            sample_n = min(5, len(grouped[src]))
            sampled.extend(random.sample(grouped[src], sample_n))

    chunks_text = "\n\n".join(
        f"[来源: {c['source_doc']} | {c['heading_path']}] {c['text']}" for c in sampled
    )

    prompt = MULTI_HOP_PROMPT.format(n=n, chunks_text=chunks_text)
    resp = await llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
    )
    return _parse_json_array(resp.get("content", ""), "multi-hop")


# ── Intent dataset generation ──────────────────────────────────


INTENT_PROMPT = """\
你是一个评估数据集生成器。根据以下文档主题，为每种意图各生成10条用户查询。

文档主题: {topics}

三种意图：
1. **qa** — 知识问答（问事实、概念、原理）
2. **quiz** — 要求测验/复习/出题
3. **gap_analysis** — 要求分析知识盲区/薄弱点

要求：
- 表述自然多样，包含口语化和正式写法
- qa 类涵盖简单和复杂问题
- quiz 类包含指定范围和不指定范围的
- gap_analysis 类包含全局分析和指定文档分析的

输出格式（JSON数组）：
[
  {{"query": "用户查询", "expected_intent": "qa|quiz|gap_analysis"}}
]

只输出JSON数组，不要其他文字。"""


async def generate_intent_dataset(source_docs: list[str]) -> list[dict]:
    """Generate intent classification test set."""
    topics = ", ".join(source_docs[:10])
    prompt = INTENT_PROMPT.format(topics=topics)
    resp = await llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4096,
    )
    return _parse_json_array(resp.get("content", ""), "intent")


# ── Helpers ────────────────────────────────────────────────────


def _parse_json_array(text: str, label: str) -> list[dict]:
    """Extract JSON array from LLM response, with fallback."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse JSON array for %s. Raw text: %s", label, text[:200])
    return []


def save_dataset(data: list[dict], name: str) -> Path:
    """Save dataset to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d items to %s", len(data), path)
    return path


# ── Main ───────────────────────────────────────────────────────


async def main():
    # Connect to Milvus
    logger.info("Connecting to Milvus at %s:%s ...", settings.milvus_host, settings.milvus_port)
    milvus_service.connect()
    milvus_service.init_collection()

    # Fetch all chunks
    chunks = fetch_all_chunks()
    if not chunks:
        logger.error("No chunks found in Milvus. Upload documents first.")
        sys.exit(1)

    grouped = group_by_source(chunks)
    source_docs = list(grouped.keys())
    logger.info("Found %d source docs: %s", len(source_docs), source_docs)

    # 1. Generate single-hop QA (5 per source, max 30)
    logger.info("=== Generating single-hop QA ===")
    single_hop_qa = []
    per_doc_n = max(1, min(5, 30 // len(source_docs)))
    for src, src_chunks in grouped.items():
        logger.info("  Generating %d QA for: %s (%d chunks)", per_doc_n, src, len(src_chunks))
        qa_items = await generate_single_hop_qa(src, src_chunks, n=per_doc_n)
        single_hop_qa.extend(qa_items)

    # 2. Generate multi-hop QA (10 items)
    logger.info("=== Generating multi-hop QA ===")
    multi_hop_qa = await generate_multi_hop_qa(grouped, n=10)

    # 3. Merge QA dataset
    qa_dataset = single_hop_qa + multi_hop_qa
    save_dataset(qa_dataset, "qa_dataset")
    logger.info("QA dataset: %d single-hop + %d multi-hop = %d total",
                len(single_hop_qa), len(multi_hop_qa), len(qa_dataset))

    # 4. Generate intent dataset
    logger.info("=== Generating intent dataset ===")
    intent_dataset = await generate_intent_dataset(source_docs)
    save_dataset(intent_dataset, "intent_dataset")
    logger.info("Intent dataset: %d items", len(intent_dataset))

    # 5. Summary
    logger.info("=== Dataset generation complete ===")
    logger.info("Output directory: %s", OUTPUT_DIR.resolve())
    logger.info("Files: qa_dataset.json, intent_dataset.json")
    logger.info("Please review and edit these files before running evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
