"""Shared evaluation profile presets and deterministic subset builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable


VALID_EVAL_PROFILES = ("smoke", "standard", "full")


@dataclass(frozen=True)
class EvalProfile:
    name: str
    rag_single_hop_limit: int | None
    rag_multi_hop_limit: int | None
    rag_refusal_limit: int | None  # refusal questions (answerable=false) in RAG eval
    rag_answerable_types: tuple[str, ...] | None  # restrict answerable pool; None = all
    agent_intent_limit: int | None
    agent_qa_single_hop_limit: int | None
    agent_qa_multi_hop_limit: int | None
    agent_quiz_limit: int | None
    agent_gap_limit: int | None
    concurrency: int  # max parallel QA E2E cases


EVAL_PROFILES: dict[str, EvalProfile] = {
    "smoke": EvalProfile(
        name="smoke",
        rag_single_hop_limit=4,
        rag_multi_hop_limit=2,
        rag_refusal_limit=2,
        rag_answerable_types=("original",),
        agent_intent_limit=6,
        agent_qa_single_hop_limit=3,
        agent_qa_multi_hop_limit=1,
        agent_quiz_limit=3,
        agent_gap_limit=2,
        concurrency=3,
    ),
    "standard": EvalProfile(
        name="standard",
        rag_single_hop_limit=10,
        rag_multi_hop_limit=5,
        rag_refusal_limit=3,
        rag_answerable_types=None,
        agent_intent_limit=15,
        agent_qa_single_hop_limit=10,
        agent_qa_multi_hop_limit=5,
        agent_quiz_limit=6,
        agent_gap_limit=4,
        concurrency=5,
    ),
    "full": EvalProfile(
        name="full",
        rag_single_hop_limit=None,
        rag_multi_hop_limit=None,
        rag_refusal_limit=None,
        rag_answerable_types=None,
        agent_intent_limit=None,
        agent_qa_single_hop_limit=None,
        agent_qa_multi_hop_limit=None,
        agent_quiz_limit=None,
        agent_gap_limit=None,
        concurrency=8,
    ),
}


def normalize_eval_profile(raw: str | None) -> str:
    """Return a valid profile name, falling back to standard."""
    if not raw:
        return "standard"
    profile = raw.strip().lower()
    return profile if profile in EVAL_PROFILES else "standard"


def get_eval_profile(raw: str | None) -> EvalProfile:
    """Resolve an EvalProfile from a user/env supplied value."""
    return EVAL_PROFILES[normalize_eval_profile(raw)]


def limit_items(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Return the full list when limit is None, else the first N items."""
    if limit is None:
        return list(items)
    return list(items[:limit])


def _round_robin_take(
    grouped_items: dict[Hashable, list[dict[str, Any]]],
    ordered_keys: list[Hashable],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Take items in a stable round-robin order across groups.

    This keeps the subset deterministic while avoiding the head-of-file bias that
    happens when datasets are grouped by source document or intent.
    """
    if limit is None:
        limit = sum(len(items) for items in grouped_items.values())

    positions = {key: 0 for key in ordered_keys}
    selected: list[dict[str, Any]] = []

    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            items = grouped_items[key]
            pos = positions[key]
            if pos >= len(items):
                continue
            selected.append(items[pos])
            positions[key] = pos + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return selected


def build_balanced_subset(
    dataset: list[dict[str, Any]],
    limit: int | None,
    group_key: Callable[[dict[str, Any]], Hashable],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, balanced subset across logical groups.

    Items remain stable within each group, and groups are traversed in first-seen
    order to keep the output reproducible without random seeds.
    """
    grouped: dict[Hashable, list[dict[str, Any]]] = {}
    ordered_keys: list[Hashable] = []

    for item in dataset:
        key = group_key(item)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(item)

    selected = _round_robin_take(grouped, ordered_keys, limit)
    return selected, {
        "limit": limit,
        "selected": len(selected),
        "group_counts": {str(key): len(grouped[key]) for key in ordered_keys},
    }


def _qa_bucket_group_key(item: dict[str, Any]) -> Hashable:
    """Group QA items by source coverage so sampling spans more documents."""
    if item.get("question_type") == "multi-hop":
        source_docs = item.get("source_docs", [])
        return tuple(source_docs) if source_docs else ("__unknown_multi_hop__",)
    return item.get("source_doc", "__unknown_single_hop__")


def build_intent_eval_subset(
    intent_dataset: list[dict[str, Any]],
    intent_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a deterministic balanced subset across intent labels."""
    subset, meta = build_balanced_subset(
        intent_dataset,
        intent_limit,
        group_key=lambda item: item.get("expected_intent", "__unknown__"),
    )
    return subset, {
        "intent_limit": intent_limit,
        "intent_selected": len(subset),
        "intent_group_counts": meta["group_counts"],
    }


def build_stratified_subset(
    dataset: list[dict[str, Any]],
    single_hop_limit: int | None,
    multi_hop_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split by question_type and return a stable, source-balanced subset."""
    single_hop = [item for item in dataset if item.get("question_type") != "multi-hop"]
    multi_hop = [item for item in dataset if item.get("question_type") == "multi-hop"]

    selected_single, single_meta = build_balanced_subset(
        single_hop,
        single_hop_limit,
        group_key=_qa_bucket_group_key,
    )
    selected_multi, multi_meta = build_balanced_subset(
        multi_hop,
        multi_hop_limit,
        group_key=_qa_bucket_group_key,
    )
    subset = selected_single + selected_multi

    return subset, {
        "single_hop_limit": single_hop_limit,
        "multi_hop_limit": multi_hop_limit,
        "single_hop_selected": len(selected_single),
        "multi_hop_selected": len(selected_multi),
        "total_selected": len(subset),
        "single_hop_group_counts": single_meta["group_counts"],
        "multi_hop_group_counts": multi_meta["group_counts"],
    }
