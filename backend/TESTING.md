# Testing Log

## 2026-04-06 — Evaluation sampling and GPT-5.4 judge compatibility

- Scope:
  - `evaluation/profiles.py`
  - `evaluation/eval_agent.py`
  - `evaluation/eval_rag.py`
  - `evaluation/judge_client.py`
  - `tests/test_eval_profiles.py`

- Commands:
  - `python -m py_compile backend/evaluation/eval_rag.py backend/evaluation/eval_agent.py backend/evaluation/profiles.py backend/evaluation/judge_client.py backend/tests/test_eval_profiles.py`
  - `pytest -q backend/tests/test_eval_profiles.py`
  - `pytest -q backend/tests/test_orchestrator.py`

- Results:
  - `py_compile` passed for all modified evaluation files.
  - `backend/tests/test_eval_profiles.py`: `7 passed`
  - `backend/tests/test_orchestrator.py`: `42 passed`

- Problems found:
  - Intent evaluation used dataset head slicing, so `smoke` only covered `qa`.
  - QA evaluation profiles were ordered by dataset layout, not by representative source coverage.
  - Judge request compatibility was brittle for GPT-5 reasoning models.

- Fixes applied:
  - Replaced head slicing with deterministic round-robin balanced sampling.
  - Balanced intent subsets across `qa / quiz / gap_analysis`.
  - Switched judge request construction to GPT-5.4-compatible `max_completion_tokens` path and aligned config default to `gpt-5.4`.
