"""Query Decompose tool: LLM-based complex query decomposition.

Takes a complex question and decomposes it into 2-4 simpler sub-queries
that can be independently retrieved and answered.

Used by QA SubAgent for multi-hop reasoning.
"""

import json
import logging

from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM_PROMPT = """\
你是一个查询分解助手。你的任务是将一个复杂问题拆解为多个简单的、可独立检索的子查询。

规则:
1. 每个子查询应该聚焦一个具体方面，能够通过单次向量检索获得答案
2. 子查询数量控制在 2-4 个
3. 子查询之间应该覆盖原始问题的各个方面，合起来能完整回答原始问题
4. 保持子查询的语言与原始问题一致（中文问题输出中文子查询）
5. 只输出 JSON 数组，不要输出其他内容

示例:
输入: "Transformer中self-attention和cross-attention的区别是什么？它们分别在encoder和decoder中如何使用？"
输出: ["Self-attention的工作原理和特点", "Cross-attention的工作原理和特点", "Transformer encoder中attention的使用方式", "Transformer decoder中attention的使用方式"]
"""


async def query_decompose(original_query: str) -> list[str]:
    """Decompose a complex query into simpler sub-queries.

    Args:
        original_query: The complex user question.

    Returns:
        List of 2-4 sub-query strings.
    """
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": original_query},
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.3,
        max_tokens=512,
    )

    content = response["content"].strip()

    # Parse JSON array from response
    try:
        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list) or not all(
            isinstance(q, str) for q in sub_queries
        ):
            raise ValueError("Expected a JSON array of strings")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "query_decompose: failed to parse LLM output, "
            "falling back to original query. error=%s output=%s",
            e,
            content[:200],
        )
        return [original_query]

    # Ensure 2-4 sub-queries
    if len(sub_queries) < 1:
        return [original_query]
    if len(sub_queries) > 4:
        sub_queries = sub_queries[:4]

    logger.info(
        "query_decompose: query=%s | sub_queries=%d: %s",
        original_query[:50],
        len(sub_queries),
        sub_queries,
    )
    return sub_queries
