"""RAGAS evaluation helpers aligned with the official async evaluation flow.

This module keeps the RAGAS-specific setup isolated from ``eval_rag.py`` so the
overall evaluation script can stay focused on the project's evaluation logic:
dataset loading, running the RAG pipeline, computing classic IR metrics, and
persisting reports.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
)


def build_ragas_judge_llm_kwargs(judge_model: str) -> dict[str, Any]:
    """Build Kimi K2.5-specific kwargs for ragas.llm_factory().

    Kimi K2.5 does not accept temperature, top_p, frequency_penalty,
    presence_penalty, or n — these are fixed by the model and must be omitted.
    """
    return {
        "extra_body": {"thinking": {"type": "disabled"}},
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
    }


def build_ragas_samples(
    qa_subset: list[dict[str, Any]],
    rag_results_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build RAGAS input samples from the project eval dataset + RAG outputs."""
    ragas_samples: list[dict[str, Any]] = []
    for item in qa_subset:
        rag_result = rag_results_map.get(item["question"])
        if not rag_result:
            continue
        ragas_samples.append(
            {
                "user_input": item["question"],
                "response": rag_result["answer"],
                "retrieved_contexts": rag_result["context_texts"],
                "reference": item.get("ground_truth", ""),
            }
        )
    return ragas_samples


def compute_ragas_means(ragas_results: dict[str, list[Any]]) -> dict[str, float]:
    """Compute mean score per metric from the row-wise RAGAS output."""
    ragas_means: dict[str, float] = {}
    for key, values in ragas_results.items():
        if key in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        numeric = [v for v in values if isinstance(v, (int, float)) and v == v]
        if numeric:
            ragas_means[key] = sum(numeric) / len(numeric)
    return ragas_means


async def run_ragas_evaluation(
    samples: list[dict[str, Any]],
    *,
    judge_model: str,
    judge_api_key: str,
    judge_base_url: str,
    embeddings: Any,
) -> dict[str, Any]:
    """Run RAGAS using the official async evaluation flow.

    Official pattern used here:
    - ``AsyncOpenAI`` client
    - ``ragas.llms.llm_factory(...)`` for judge initialization
    - ``ragas.aevaluate(...)`` for async-safe production evaluation
    """
    from openai import AsyncOpenAI
    from ragas import EvaluationDataset, aevaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import llm_factory
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    ragas_dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
                reference=sample["reference"],
            )
            for sample in samples
        ]
    )

    merged_results: dict[str, list[Any]] = {
        "user_input": [sample["user_input"] for sample in samples],
        "response": [sample["response"] for sample in samples],
        "retrieved_contexts": [sample["retrieved_contexts"] for sample in samples],
        "reference": [sample["reference"] for sample in samples],
    }
    metric_status: dict[str, dict[str, Any]] = {}

    try:
        judge_client = AsyncOpenAI(
            api_key=judge_api_key,
            base_url=judge_base_url,
        )
        judge_llm = llm_factory(
            judge_model,
            client=judge_client,
            **build_ragas_judge_llm_kwargs(judge_model),
        )
    except Exception as exc:
        logger.exception("Failed to initialize RAGAS judge: %s", exc)
        for metric_name in RAGAS_METRIC_NAMES:
            merged_results[metric_name] = [None] * len(samples)
            metric_status[metric_name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return {"results": merged_results, "metric_status": metric_status}

    metric_builders = [
        ("faithfulness", Faithfulness),
        ("answer_relevancy", AnswerRelevancy),
        ("llm_context_precision_with_reference", LLMContextPrecisionWithReference),
        ("context_recall", LLMContextRecall),
    ]

    logger.info(
        "Running RAGAS evaluation with %d samples via aevaluate() and llm_factory().",
        len(samples),
    )

    for metric_name, metric_cls in metric_builders:
        try:
            metric = metric_cls()
            logger.info(
                "Running RAGAS metric=%s with judge model=%s using official aevaluate() flow.",
                metric_name,
                judge_model,
            )
            result = await aevaluate(
                dataset=ragas_dataset,
                metrics=[metric],
                llm=judge_llm,
                embeddings=embeddings,
                raise_exceptions=True,
                show_progress=False,
            )
            metric_values = result.to_pandas().to_dict(orient="list").get(metric_name, [])
            merged_results[metric_name] = metric_values
            valid_count = sum(
                1 for value in metric_values if isinstance(value, (int, float)) and value == value
            )
            metric_status[metric_name] = {
                "status": "ok",
                "valid_count": valid_count,
                "sample_count": len(metric_values),
            }
        except Exception as exc:
            logger.exception("RAGAS metric failed: %s", metric_name)
            merged_results[metric_name] = [None] * len(samples)
            metric_status[metric_name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    return {"results": merged_results, "metric_status": metric_status}
