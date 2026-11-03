from evaluation.judge_client import build_judge_request_kwargs
from evaluation.profiles import (
    build_balanced_subset,
    build_intent_eval_subset,
    build_stratified_subset,
    get_eval_profile,
    normalize_eval_profile,
)


def _make_dataset(single_hop_count: int = 20, multi_hop_count: int = 10) -> list[dict]:
    single = []
    for i in range(single_hop_count):
        doc_idx = i % 4
        single.append(
            {
                "question": f"single-{i}",
                "question_type": "factual",
                "source_doc": f"doc-{doc_idx}.md",
            }
        )

    multi = []
    for i in range(multi_hop_count):
        multi.append(
            {
                "question": f"multi-{i}",
                "question_type": "multi-hop",
                "source_docs": [f"doc-{i % 3}.md", f"doc-{(i + 1) % 3}.md"],
            }
        )

    return single + multi


def _make_intent_dataset() -> list[dict]:
    dataset = []
    for intent in ("qa", "quiz", "gap_analysis"):
        for i in range(4):
            dataset.append({"query": f"{intent}-{i}", "expected_intent": intent})
    return dataset


def test_normalize_eval_profile_defaults_to_standard():
    assert normalize_eval_profile(None) == "standard"
    assert normalize_eval_profile("unknown") == "standard"
    assert normalize_eval_profile("SMOKE") == "smoke"


def test_build_balanced_subset_round_robins_across_groups():
    dataset = [
        {"id": "a1", "group": "a"},
        {"id": "a2", "group": "a"},
        {"id": "b1", "group": "b"},
        {"id": "b2", "group": "b"},
        {"id": "c1", "group": "c"},
    ]

    subset, meta = build_balanced_subset(dataset, 4, group_key=lambda item: item["group"])

    assert [item["id"] for item in subset] == ["a1", "b1", "c1", "a2"]
    assert meta["group_counts"] == {"a": 2, "b": 2, "c": 1}


def test_smoke_profile_limits_match_expected_sizes_and_span_docs():
    profile = get_eval_profile("smoke")
    dataset = _make_dataset()

    subset, meta = build_stratified_subset(
        dataset,
        profile.rag_single_hop_limit,
        profile.rag_multi_hop_limit,
    )

    assert meta["single_hop_selected"] == 4
    assert meta["multi_hop_selected"] == 2
    assert meta["total_selected"] == 6
    assert [item["source_doc"] for item in subset[:4]] == [
        "doc-0.md",
        "doc-1.md",
        "doc-2.md",
        "doc-3.md",
    ]


def test_standard_profile_matches_current_agent_subset_with_balanced_docs():
    profile = get_eval_profile("standard")
    dataset = _make_dataset()

    subset, meta = build_stratified_subset(
        dataset,
        profile.agent_qa_single_hop_limit,
        profile.agent_qa_multi_hop_limit,
    )

    assert meta["single_hop_selected"] == 10
    assert meta["multi_hop_selected"] == 5
    assert meta["total_selected"] == 15
    assert len({item["source_doc"] for item in subset[:4]}) == 4


def test_full_profile_selects_all_items():
    profile = get_eval_profile("full")
    dataset = _make_dataset(single_hop_count=7, multi_hop_count=3)

    subset, meta = build_stratified_subset(
        dataset,
        profile.rag_single_hop_limit,
        profile.rag_multi_hop_limit,
    )

    assert meta["single_hop_limit"] is None
    assert meta["multi_hop_limit"] is None
    assert meta["single_hop_selected"] == 7
    assert meta["multi_hop_selected"] == 3
    assert meta["total_selected"] == 10
    assert subset == dataset


def test_intent_subset_balances_labels_instead_of_head_slice():
    dataset = _make_intent_dataset()

    subset, meta = build_intent_eval_subset(dataset, 6)

    assert [item["expected_intent"] for item in subset] == [
        "qa",
        "quiz",
        "gap_analysis",
        "qa",
        "quiz",
        "gap_analysis",
    ]
    assert meta["intent_group_counts"] == {
        "qa": 4,
        "quiz": 4,
        "gap_analysis": 4,
    }


def test_judge_request_kwargs_send_no_sampling_params():
    kwargs = build_judge_request_kwargs(
        model="anthropic/claude-sonnet-5.5",
        messages=[{"role": "user", "content": "judge this"}],
    )

    assert kwargs == {
        "model": "anthropic/claude-sonnet-5.5",
        "messages": [{"role": "user", "content": "judge this"}],
    }


def test_judge_request_kwargs_pass_extra_body_and_max_completion_tokens():
    routing = {"provider": {"order": ["anthropic"], "allow_fallbacks": False}}
    kwargs = build_judge_request_kwargs(
        model="anthropic/claude-sonnet-5.5",
        messages=[{"role": "user", "content": "judge this"}],
        max_completion_tokens=512,
        extra_body=routing,
    )

    assert kwargs == {
        "model": "anthropic/claude-sonnet-5.5",
        "messages": [{"role": "user", "content": "judge this"}],
        "extra_body": routing,
        "max_completion_tokens": 512,
    }
