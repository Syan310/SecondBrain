"""LLM-based concept extraction from document chunks.

Replaces the old heading-based extraction. For each parent chunk, calls LLM
to extract meaningful knowledge concepts as a flat list of standardized terms.
"""

import json
import logging

from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是一个知识概念提取助手。从给定的文本段落中提取所有核心知识概念。

## 提取规则

1. 提取文本中涉及的所有有意义的知识概念
2. 概念名称应当足够具体，能作为一个可测试的知识单元（即可以就此出一道quiz题）
3. 概念名称应简洁、标准化，优先使用该领域的通用术语
4. 例如："Self-Attention机制"、"Transformer位置编码"、"L1正则化"、"交叉熵损失"、"梯度消失问题"
5. 以下类型不是知识概念，不要提取：
   - 结构性标题（"实践结论"、"核心要点"、"注意事项"、"常见误区"）
   - 过程性描述（"实现步骤"、"使用方法"）
   - 元信息（"参考文献"、"前置知识"）
   - 过于宽泛的词（"深度学习"、"机器学习"，除非文本确实在系统讲解该大主题本身）
6. 如果文本内容不包含任何有意义的知识概念（例如目录、前言、致谢等），返回空数组

## 已有概念

以下概念已在知识库中存在。如果文本涉及相同概念，直接复用已有名称，不要创造含义相同但措辞不同的新概念：
{existing_concepts}

## 输出

仅输出 JSON 数组，不要输出其他内容。示例：

["Self-Attention机制", "Transformer位置编码", "L1正则化"]
"""


async def extract_concepts(
    text: str,
    existing_concepts: list[str],
) -> list[str]:
    """Extract knowledge concepts from a parent chunk using LLM.

    Args:
        text: The parent chunk text content.
        existing_concepts: Already-known concept topics for deduplication.

    Returns:
        List of concept topic strings extracted from the text.
    """
    if not text.strip():
        return []

    # Format existing concepts for prompt
    if existing_concepts:
        concepts_str = json.dumps(existing_concepts, ensure_ascii=False)
    else:
        concepts_str = "（暂无已有概念）"

    system = _SYSTEM_PROMPT.replace("{existing_concepts}", concepts_str)

    response = await llm_client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    content = (response.get("content") or "").strip()
    return _parse_response(content)


def _parse_response(content: str) -> list[str]:
    """Parse LLM response into a list of concept strings."""
    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse concept extraction response: %s", content[:200])
        return []

    if not isinstance(result, list):
        logger.warning("Concept extraction returned non-list: %s", type(result))
        return []

    # Filter to strings only, deduplicate, preserve order
    seen: set[str] = set()
    concepts: list[str] = []
    for item in result:
        if isinstance(item, str) and item.strip():
            normalized = item.strip()
            if normalized not in seen:
                seen.add(normalized)
                concepts.append(normalized)

    return concepts
