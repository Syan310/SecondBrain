"""Citation enforcement: shared by the eval pipeline and the QA SubAgent.

The answer prompt asks for a chunk id marker ([1] or [1][3]) at the end of
every sentence. apply_citations() then post-processes the answer:

  - "off":    return the answer unchanged
  - "prompt": strip the markers, keep every sentence
  - "filter": drop sentences without a valid marker, then strip the markers;
              if nothing is left, return REFUSAL_SENTENCE
  - "fallback": like "prompt" (strip the markers, keep every sentence); the
              prompt uses FALLBACK_RULES, so when the chunks have no answer the
              model starts with NOT_FOUND_SENTENCE and answers from its own knowledge
"""

import re
from dataclasses import dataclass

REFUSAL_SENTENCE = "知识库中没有足够的信息回答这个问题。"
NOT_FOUND_SENTENCE = "笔记里没有找到相关内容。"

# Appended after the existing 回答要求 in both prompts. Keep the wording identical.
CITATION_RULES = (
    "- 每句话结尾标注它依据的片段编号，格式为 [1] 或 [1][3]\n"
    "- 只能引用上面给出的片段编号\n"
    "- 找不到依据的句子不要写\n"
    f"- 如果片段不足以回答问题，只回答：{REFUSAL_SENTENCE}"
)

# CITATION_RULES with the last rule replaced: answer from own knowledge instead of refusing.
FALLBACK_RULES = (
    CITATION_RULES.rsplit("\n", 1)[0]
    + f"\n- 如果片段里没有答案，第一句写：{NOT_FOUND_SENTENCE}然后根据你自己的知识回答，这部分不要标注引用。"
)

CITATION_MODES = ("off", "prompt", "filter", "fallback")


def rules_for_mode(mode: str) -> str:
    """Prompt rules appended for a citation mode ("" for off)."""
    if mode not in CITATION_MODES:
        raise ValueError(f"Unknown citation_mode: {mode!r}")
    if mode == "off":
        return ""
    return FALLBACK_RULES if mode == "fallback" else CITATION_RULES


def is_not_found(answer: str) -> bool:
    """True if the answer (leading whitespace ignored) starts with NOT_FOUND_SENTENCE."""
    return (answer or "").lstrip().startswith(NOT_FOUND_SENTENCE)

# One marker: [1], [1,3], [1, 3], [1，3], [1、3] or 【1】 (each id is checked).
_MARKER = r"[\[【][ \t]*\d+(?:[ \t]*[,，、][ \t]*\d+)*[ \t]*[\]】]"
# Markers wrapped in otherwise empty parentheses, e.g. （[1]）; removed with them.
_WRAPPED = rf"[（(][ \t]*{_MARKER}(?:[ \t]*{_MARKER})*[ \t]*[）)]"
_MARKER_RE = re.compile(_MARKER)
_ID_RE = re.compile(r"\d+")
_STRIP_MARKER_RE = re.compile(rf"[ \t]*(?:{_WRAPPED}|{_MARKER})")
# Sentence end: Chinese/English terminal punctuation (plus closing quotes),
# or an English period followed by whitespace (not the one in "e.g." / "i.e.").
_SENT_END_RE = re.compile(r"[。！？!?]+[”’」』）)]*|(?<!e\.g)(?<!i\.e)\.(?=\s)")
_TRAILING_MARKERS_RE = re.compile(rf"(?:[ \t]*(?:{_WRAPPED}|{_MARKER}))+")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)、])\s+")
_FORMAT_LINE_RE = re.compile(r"^\s*(?:#{1,6}(?:\s.*)?|(?:[-*_]\s*){3,})$")


@dataclass
class CitationResult:
    clean_answer: str        # text for the user and RAGAS, without markers
    units_total: int         # sentences after splitting
    units_kept: int          # sentences kept
    invalid_citations: int   # markers whose id is not in valid_ids
    fully_refused: bool      # every sentence dropped, REFUSAL_SENTENCE returned


def _split_line(line: str) -> list[str]:
    """Split one line into sentences. Markers after the punctuation stay with
    the sentence before them. A Markdown list item is one sentence."""
    if _LIST_ITEM_RE.match(line):
        return [line]
    units: list[str] = []
    pos = 0
    while pos < len(line):
        m = _SENT_END_RE.search(line, pos)
        end = m.end() if m else len(line)
        trailing = _TRAILING_MARKERS_RE.match(line, end)
        if trailing:
            end = trailing.end()
        unit = line[pos:end]
        if unit.strip():
            units.append(unit)
        elif units:
            units[-1] += unit
        pos = end
    return units


def _cited_ids(text: str) -> list[int]:
    """Every id in every marker of the text, in order."""
    return [int(n) for m in _MARKER_RE.findall(text) for n in _ID_RE.findall(m)]


def _strip_markers(text: str) -> str:
    return _STRIP_MARKER_RE.sub("", text)


def _is_refusal(unit: str) -> bool:
    core = _strip_markers(unit).strip()
    return core.rstrip("。.") == REFUSAL_SENTENCE.rstrip("。")


def _join(lines: list[str]) -> str:
    """Join lines, collapsing runs of blank lines."""
    out: list[str] = []
    for line in lines:
        line = line.rstrip()
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    return "\n".join(out).strip()


def apply_citations(answer: str, valid_ids: set[int], mode: str) -> CitationResult:
    """Post-process an answer according to citation_mode."""
    if mode not in CITATION_MODES:
        raise ValueError(f"Unknown citation_mode: {mode!r}")

    lines = answer.split("\n")
    line_units = [
        None if _FORMAT_LINE_RE.match(line) else _split_line(line)
        for line in lines
    ]
    units_total = sum(len(u) for u in line_units if u)

    if mode == "off":
        return CitationResult(answer, units_total, units_total, 0, False)

    invalid = sum(1 for n in _cited_ids(answer) if n not in valid_ids)

    if mode in ("prompt", "fallback"):
        return CitationResult(_strip_markers(answer).strip(), units_total, units_total, invalid, False)

    kept_lines: list[str] = []
    units_kept = 0
    for line, units in zip(lines, line_units):
        if units is None:          # pure format line: drop
            continue
        if not units:              # blank line: keep layout
            kept_lines.append("")
            continue
        kept = [
            u for u in units
            if _is_refusal(u) or any(n in valid_ids for n in _cited_ids(u))
        ]
        units_kept += len(kept)
        if kept:
            text = _strip_markers("".join(kept))
            kept_lines.append(text if _LIST_ITEM_RE.match(line) else text.strip())

    clean = _join(kept_lines)
    if units_kept == 0 or not clean:
        return CitationResult(REFUSAL_SENTENCE, units_total, 0, invalid, True)
    return CitationResult(clean, units_total, units_kept, invalid, False)
