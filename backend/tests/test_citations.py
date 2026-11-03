"""Tests for citation enforcement (apply_citations). No external services."""

import pytest

from app.services.citations import (
    CITATION_RULES,
    REFUSAL_SENTENCE,
    apply_citations,
)

IDS = {1, 2, 3}


# ── Cases from the task doc, section 5.6 ─────────────────────


def test_marker_after_punctuation_belongs_to_previous_sentence():
    r = apply_citations("A 是 B。[1]C 是 D。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"
    assert (r.units_total, r.units_kept) == (2, 1)
    assert not r.fully_refused


def test_marker_before_punctuation_is_kept():
    r = apply_citations("A 是 B[1]。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"
    assert (r.units_total, r.units_kept) == (1, 1)


def test_multiple_markers_all_stripped():
    r = apply_citations("A 是 B[1][3]。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"
    assert r.units_kept == 1


def test_unknown_id_is_dropped_and_counted():
    r = apply_citations("A 是 B[9]。", {1, 2}, "filter")
    assert r.invalid_citations == 1
    assert r.units_kept == 0
    assert r.clean_answer == REFUSAL_SENTENCE
    assert r.fully_refused


def test_refusal_sentence_alone_is_kept():
    r = apply_citations(REFUSAL_SENTENCE, IDS, "filter")
    assert r.clean_answer == REFUSAL_SENTENCE
    assert (r.units_total, r.units_kept) == (1, 1)
    assert not r.fully_refused


def test_all_dropped_returns_refusal():
    r = apply_citations("A 是 B。C 是 D。", IDS, "filter")
    assert r.clean_answer == REFUSAL_SENTENCE
    assert (r.units_total, r.units_kept) == (2, 0)
    assert r.fully_refused


def test_list_items():
    r = apply_citations("- A[1]\n- B", IDS, "filter")
    assert r.clean_answer == "- A"
    assert (r.units_total, r.units_kept) == (2, 1)


def test_prompt_mode_strips_markers_only():
    r = apply_citations("A[1]。B。", IDS, "prompt")
    assert r.clean_answer == "A。B。"
    assert (r.units_total, r.units_kept) == (2, 2)
    assert not r.fully_refused


def test_english_sentences():
    r = apply_citations("A is B [1]. C is D.", IDS, "filter")
    assert r.clean_answer == "A is B."
    assert (r.units_total, r.units_kept) == (2, 1)


def test_comma_separated_ids():
    r = apply_citations("A 是 B[1,3]。C 是 D[1, 2]。", IDS, "filter")
    assert r.clean_answer == "A 是 B。C 是 D。"
    assert (r.units_total, r.units_kept) == (2, 2)
    assert r.invalid_citations == 0


def test_enumeration_comma_and_fullwidth_comma():
    r = apply_citations("A 是 B[1、2]。C 是 D[1，2]。", IDS, "filter")
    assert r.clean_answer == "A 是 B。C 是 D。"
    assert (r.units_total, r.units_kept) == (2, 2)


def test_fullwidth_brackets():
    r = apply_citations("A 是 B【1】。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"
    assert r.units_kept == 1
    assert not r.fully_refused


def test_mixed_valid_and_invalid_ids_in_one_marker():
    r = apply_citations("A 是 B[1,9]。", {1, 2}, "filter")
    assert r.clean_answer == "A 是 B。"
    assert r.units_kept == 1
    assert r.invalid_citations == 1


# ── Extra cases ──────────────────────────────────────────────


def test_off_mode_returns_answer_unchanged():
    answer = "A 是 B[1]。C 是 D。"
    r = apply_citations(answer, IDS, "off")
    assert r.clean_answer == answer
    assert r.invalid_citations == 0
    assert not r.fully_refused


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        apply_citations("A。", IDS, "strict")


def test_format_lines_are_dropped_and_not_counted():
    r = apply_citations("## 结论\n---\nA 是 B[1]。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"
    assert (r.units_total, r.units_kept) == (1, 1)


def test_valid_and_invalid_markers_in_one_sentence():
    r = apply_citations("A 是 B[2][7]。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"
    assert r.invalid_citations == 1


def test_invalid_markers_counted_in_prompt_mode():
    r = apply_citations("A[1]。B[8]。", IDS, "prompt")
    assert r.clean_answer == "A。B。"
    assert r.invalid_citations == 1


def test_numbered_list_item_is_one_sentence():
    r = apply_citations("1. 先切块。再向量化[1]。\n2. 存入数据库。", IDS, "filter")
    assert r.clean_answer == "1. 先切块。再向量化。"
    assert (r.units_total, r.units_kept) == (2, 1)


def test_multiline_keeps_cited_lines():
    answer = "Pre-Norm 更稳定[1]。\n\n原因是残差路径通畅[2]。没有依据的话。"
    r = apply_citations(answer, IDS, "filter")
    assert r.clean_answer == "Pre-Norm 更稳定。\n\n原因是残差路径通畅。"
    assert (r.units_total, r.units_kept) == (3, 2)


def test_decimal_point_does_not_split():
    r = apply_citations("准确率从 0.72 提升到 0.90[1]。", IDS, "filter")
    assert r.clean_answer == "准确率从 0.72 提升到 0.90。"
    assert r.units_total == 1


def test_comma_ids_counted_as_invalid_when_unknown():
    # Before v1.5 these were dropped silently with invalid_citations=0.
    r = apply_citations("A 是 B[1,3]。C 是 D[2]。", {2}, "filter")
    assert r.clean_answer == "C 是 D。"
    assert r.invalid_citations == 2


def test_fullwidth_brackets_unknown_id_counted():
    r = apply_citations("A 是 B【7】。", IDS, "filter")
    assert r.fully_refused
    assert r.invalid_citations == 1


def test_fullwidth_marker_after_punctuation():
    r = apply_citations("A 是 B。【2】C 是 D。", IDS, "filter")
    assert r.clean_answer == "A 是 B。"


def test_new_formats_stripped_in_prompt_mode():
    r = apply_citations("A[1, 3]。B【2】。C[1、2]。", IDS, "prompt")
    assert r.clean_answer == "A。B。C。"


def test_parenthesized_marker_removed_with_parentheses():
    r = apply_citations("A 是 B（[1]）。C 是 D ([2][3]).", IDS, "filter")
    assert r.clean_answer == "A 是 B。C 是 D."


def test_parentheses_with_other_text_are_kept():
    r = apply_citations("A 是 B（见下文[1]）。", IDS, "filter")
    assert r.clean_answer == "A 是 B（见下文）。"


def test_eg_does_not_end_a_sentence():
    r = apply_citations("Use a reranker, e.g. bge-reranker [1]. Other text.", IDS, "filter")
    assert r.clean_answer == "Use a reranker, e.g. bge-reranker."
    assert r.units_total == 2


def test_empty_answer_in_filter_mode_is_refusal():
    r = apply_citations("", IDS, "filter")
    assert r.clean_answer == REFUSAL_SENTENCE
    assert r.fully_refused


def test_citation_rules_wording():
    lines = CITATION_RULES.split("\n")
    assert len(lines) == 4
    assert all(line.startswith("- ") for line in lines)
    assert lines[-1].endswith(REFUSAL_SENTENCE)


# ── fallback mode (task-fallback-answer.md §1) ──────────────


def test_fallback_rules_replace_only_the_last_rule():
    from app.services.citations import FALLBACK_RULES

    old, new = CITATION_RULES.split("\n"), FALLBACK_RULES.split("\n")
    assert new[:-1] == old[:-1]
    assert new[-1] == (
        "- 如果片段里没有答案，第一句写：笔记里没有找到相关内容。"
        "然后根据你自己的知识回答，这部分不要标注引用。"
    )
    assert REFUSAL_SENTENCE not in FALLBACK_RULES


def test_rules_for_mode():
    from app.services.citations import CITATION_MODES, FALLBACK_RULES, rules_for_mode

    assert "fallback" in CITATION_MODES
    assert rules_for_mode("off") == ""
    assert rules_for_mode("prompt") == rules_for_mode("filter") == CITATION_RULES
    assert rules_for_mode("fallback") == FALLBACK_RULES
    with pytest.raises(ValueError):
        rules_for_mode("strict")


@pytest.mark.parametrize("answer, expected", [
    ("笔记里没有找到相关内容。RAPTOR 是一种……", True),
    ("  \n笔记里没有找到相关内容。\n\n根据我的知识……", True),
    ("A 是 B。笔记里没有找到相关内容。", False),
    ("笔记里没有找到相关内容", False),  # no full stop: not the fixed sentence
    ("", False),
    (None, False),
])
def test_is_not_found(answer, expected):
    from app.services.citations import is_not_found

    assert is_not_found(answer) is expected


def test_fallback_mode_keeps_uncited_sentences_and_strips_markers():
    answer = "笔记里没有找到相关内容。\n根据我的知识，RAPTOR 递归聚类。A 是 B[1]。"
    r = apply_citations(answer, IDS, "fallback")
    assert r.clean_answer == "笔记里没有找到相关内容。\n根据我的知识，RAPTOR 递归聚类。A 是 B。"
    assert r.units_kept == r.units_total == 3
    assert not r.fully_refused
