from evaluation.profiles import build_stratified_subset, get_eval_profile, normalize_eval_profile


def _make_dataset(single_hop_count: int = 20, multi_hop_count: int = 10) -> list[dict]:
    single = [
        {"question": f"single-{i}", "question_type": "factual"}
        for i in range(single_hop_count)
    ]
    multi = [
        {"question": f"multi-{i}", "question_type": "multi-hop"}
        for i in range(multi_hop_count)
    ]
    return single + multi


def test_normalize_eval_profile_defaults_to_standard():
    assert normalize_eval_profile(None) == "standard"
    assert normalize_eval_profile("unknown") == "standard"
    assert normalize_eval_profile("SMOKE") == "smoke"


def test_smoke_profile_limits_match_expected_sizes():
    profile = get_eval_profile("smoke")
    dataset = _make_dataset()

    subset, meta = build_stratified_subset(
        dataset,
        profile.rag_single_hop_limit,
        profile.rag_multi_hop_limit,
    )

    assert meta == {
        "single_hop_limit": 4,
        "multi_hop_limit": 2,
        "single_hop_selected": 4,
        "multi_hop_selected": 2,
        "total_selected": 6,
    }
    assert [item["question"] for item in subset[:4]] == ["single-0", "single-1", "single-2", "single-3"]
    assert [item["question"] for item in subset[4:]] == ["multi-0", "multi-1"]


def test_standard_profile_matches_current_agent_subset():
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
    assert len(subset) == 15


def test_full_profile_selects_all_items():
    profile = get_eval_profile("full")
    dataset = _make_dataset(single_hop_count=7, multi_hop_count=3)

    subset, meta = build_stratified_subset(
        dataset,
        profile.rag_single_hop_limit,
        profile.rag_multi_hop_limit,
    )

    assert meta == {
        "single_hop_limit": None,
        "multi_hop_limit": None,
        "single_hop_selected": 7,
        "multi_hop_selected": 3,
        "total_selected": 10,
    }
    assert subset == dataset
