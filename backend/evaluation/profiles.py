"""Shared evaluation profile presets for smoke / standard / full runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_EVAL_PROFILES = ("smoke", "standard", "full")


@dataclass(frozen=True)
class EvalProfile:
    name: str
    rag_single_hop_limit: int | None
    rag_multi_hop_limit: int | None
    agent_intent_limit: int | None
    agent_qa_single_hop_limit: int | None
    agent_qa_multi_hop_limit: int | None


EVAL_PROFILES: dict[str, EvalProfile] = {
    "smoke": EvalProfile(
        name="smoke",
        rag_single_hop_limit=4,
        rag_multi_hop_limit=2,
        agent_intent_limit=6,
        agent_qa_single_hop_limit=3,
        agent_qa_multi_hop_limit=1,
    ),
    "standard": EvalProfile(
        name="standard",
        rag_single_hop_limit=10,
        rag_multi_hop_limit=5,
        agent_intent_limit=15,
        agent_qa_single_hop_limit=10,
        agent_qa_multi_hop_limit=5,
    ),
    "full": EvalProfile(
        name="full",
        rag_single_hop_limit=None,
        rag_multi_hop_limit=None,
        agent_intent_limit=None,
        agent_qa_single_hop_limit=None,
        agent_qa_multi_hop_limit=None,
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


def build_stratified_subset(
    dataset: list[dict[str, Any]],
    single_hop_limit: int | None,
    multi_hop_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split by question_type and return a stable single-hop + multi-hop subset."""
    single_hop = [item for item in dataset if item.get("question_type") != "multi-hop"]
    multi_hop = [item for item in dataset if item.get("question_type") == "multi-hop"]

    selected_single = limit_items(single_hop, single_hop_limit)
    selected_multi = limit_items(multi_hop, multi_hop_limit)
    subset = selected_single + selected_multi

    return subset, {
        "single_hop_limit": single_hop_limit,
        "multi_hop_limit": multi_hop_limit,
        "single_hop_selected": len(selected_single),
        "multi_hop_selected": len(selected_multi),
        "total_selected": len(subset),
    }
