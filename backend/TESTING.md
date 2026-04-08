# Testing Log

## 2026-04-07 — reranker default switched to off

- Scope:
  - `app/config.py`
  - `tests/test_rag_search.py`
  - `docs/tools.md`
  - `TASKS.md`

- Commands:
  - `PYTHONDONTWRITEBYTECODE=1 /home/syan/miniconda3/envs/llm/bin/python -m pytest -q backend/tests/test_rag_search.py`

- Results:
  - `backend/tests/test_rag_search.py`: `6 passed`

- Problems found:
  - Existing `rag_search` tests assumed reranker was enabled by default, so changing the default config caused two test failures even though runtime behavior was correct.

- Fixes applied:
  - Changed `enable_reranker` default from `True` to `False`.
  - Updated reranker-on unit tests to patch `settings.enable_reranker = True` explicitly, so each test now declares the mode it depends on.
  - Updated tool docs and task tracking to reflect the new default strategy.

## 2026-04-07 — rag_search supports disabling reranker without changing final chunk count

- Scope:
  - `app/config.py`
  - `app/agent/tools/rag_search.py`
  - `tests/test_rag_search.py`
  - `docs/tools.md`

- Commands:
  - `PYTHONDONTWRITEBYTECODE=1 /home/syan/miniconda3/envs/llm/bin/python - <<'PY' ... compile(...) ... PY`
  - `/home/syan/miniconda3/envs/llm/bin/python -m pytest -q backend/tests/test_rag_search.py`

- Results:
  - Syntax check passed for modified config/tool/test files.
  - `backend/tests/test_rag_search.py`: `6 passed`

- Problems found:
  - The default system `python3` environment did not include `pytest`.
  - Writing `.pyc` files under `backend/app/__pycache__` hit a local permission issue, so syntax validation was run with `PYTHONDONTWRITEBYTECODE=1`.

- Fixes applied:
  - Added `enable_reranker` config to gate reranker usage explicitly.
  - Updated `rag_search` so disabling reranker returns Milvus original ranking capped at `top_k`, keeping the downstream LLM context size unchanged for fair A/B evaluation.
  - Added a regression test to verify reranker-off mode does not call the reranker and still returns exactly `top_k` chunks.

## 2026-04-06 — Judge switched to Kimi K2.5 non-thinking mode

- Scope:
  - `.env`
  - `app/config.py`
  - `evaluation/judge_client.py`
  - `evaluation/ragas_runner.py`
  - `tests/test_eval_profiles.py`
  - `tests/test_ragas_runner.py`

- Commands:
  - `python -m py_compile backend/app/config.py backend/evaluation/judge_client.py backend/evaluation/ragas_runner.py backend/evaluation/eval_agent.py backend/evaluation/eval_rag.py backend/tests/test_eval_profiles.py backend/tests/test_ragas_runner.py`
  - `pytest -q backend/tests/test_eval_profiles.py backend/tests/test_ragas_runner.py`

- Results:
  - `py_compile` passed for all modified judge/eval files.
  - `backend/tests/test_eval_profiles.py backend/tests/test_ragas_runner.py`: `15 passed`

- Problems found:
  - The OpenAI judge path was still blocked by provider/model compatibility issues around `max_tokens`.
  - `backend/.env` still overrode the code defaults with the old OpenAI judge path (`https://api.openai.com/v1` + `gpt-5.4`).
  - Kimi K2.5 has model-specific request constraints: non-thinking mode requires `thinking={"type": "disabled"}`, fixed `temperature=0.6`, fixed `top_p=0.95`, and defaults `max_tokens` to 32k.
  - `ragas.llm_factory()` injects its own defaults (`temperature=0.01`, `top_p=0.1`, `max_tokens=1024`), so the Kimi path must explicitly override them.

- Fixes applied:
  - Changed the default evaluation judge to `kimi-k2.5` with `judge_base_url=https://api.moonshot.ai/v1`.
  - Updated the shared judge client to send `thinking={"type": "disabled"}` for Kimi and keep the request on the non-thinking path.
  - Updated the RAGAS judge setup to pass Kimi-compatible kwargs into `llm_factory(...)`: `thinking={"type":"disabled"}`, `temperature=0.6`, `top_p=0.95`, `max_tokens=32768`.
  - Aligned both `eval_rag` and `eval_agent` onto the same Kimi non-thinking judge configuration.

## 2026-04-06 — Judge switched from GPT-5.4-mini to GPT-4.1

- Scope:
  - `app/config.py`
  - `evaluation/judge_client.py`
  - `evaluation/eval_agent.py`
  - `tests/test_eval_profiles.py`

- Commands:
  - `python -m py_compile backend/app/config.py backend/evaluation/judge_client.py backend/tests/test_eval_profiles.py`
  - `pytest -q backend/tests/test_eval_profiles.py`

- Results:
  - `py_compile` passed for all modified judge config/client files.
  - `backend/tests/test_eval_profiles.py`: `9 passed`

- Problems found:
  - `ragas 0.4.3` does not correctly recognize dotted GPT-5 model names such as `gpt-5.4-mini` as reasoning models, so its internal `max_tokens` -> `max_completion_tokens` mapping does not trigger.
  - Keeping GPT-5-specific request fields in the shared judge client would leave unnecessary incompatibility risk after switching to a non-reasoning judge model.

- Fixes applied:
  - Changed the default judge model to `gpt-4.1`.
  - Updated `build_judge_request_kwargs()` to use a plain Chat Completions path for non-GPT-5 models and only attach `reasoning={"effort": "none"}` for GPT-5 models.
  - Mapped optional token limits to `max_tokens` for `gpt-4.1`, while retaining `max_completion_tokens` only for GPT-5 paths.
  - Updated `eval_agent.py` result metadata and project task docs to stop referring to the old GPT-5 judge path.

## 2026-04-06 — eval_rag refactored back to official ragas flow

- Scope:
  - `evaluation/eval_rag.py`
  - `evaluation/ragas_runner.py`
  - `evaluation/judge_client.py`
  - `requirements.txt`
  - `tests/test_eval_profiles.py`
  - `tests/test_ragas_runner.py`

- Commands:
  - `python -m py_compile backend/evaluation/eval_rag.py backend/evaluation/ragas_runner.py backend/evaluation/judge_client.py backend/tests/test_eval_profiles.py backend/tests/test_ragas_runner.py`
  - `pytest -q backend/tests/test_eval_profiles.py backend/tests/test_ragas_runner.py`

- Results:
  - `py_compile` passed for all modified evaluation files.
  - `backend/tests/test_eval_profiles.py backend/tests/test_ragas_runner.py`: `11 passed`

- Problems found:
  - `eval_rag.py` had accumulated custom RAGAS wiring, making the judge setup hard to reason about and fragile to library changes.
  - The RAGAS path depended on project-specific proxy logic instead of the official `ragas` evaluation flow.
  - The dependency floor (`ragas>=0.2.5`) no longer matched the latest public docs/API shape used by the refactor.

- Fixes applied:
  - Extracted RAGAS-specific orchestration into `evaluation/ragas_runner.py`.
  - Reworked the RAGAS judge path to follow the documented async flow: `AsyncOpenAI` + `llm_factory(...)` + `aevaluate(...)`.
  - Removed the RAGAS-specific OpenAI compat proxy from the evaluation path and kept `eval_rag.py` focused on project eval logic only.
  - Added unit tests for RAGAS sample building, mean aggregation, and multi-hop lift calculation.
  - Bumped the dependency floor to `ragas>=0.4.3` to align runtime expectations with the official docs used for this refactor.

## 2026-04-06 — Judge switched to gpt-5.4-mini with reasoning disabled

- Scope:
  - `app/config.py`
  - `evaluation/judge_client.py`
  - `evaluation/eval_agent.py`
  - `evaluation/eval_rag.py`
  - `tests/test_eval_profiles.py`

- Commands:
  - `python -m py_compile backend/evaluation/eval_rag.py backend/evaluation/eval_agent.py backend/evaluation/judge_client.py backend/app/config.py backend/tests/test_eval_profiles.py`
  - `pytest -q backend/tests/test_eval_profiles.py`

- Results:
  - `py_compile` passed for all modified evaluation files.
  - `backend/tests/test_eval_profiles.py`: `9 passed`

- Problems found:
  - Agent judge still sent an explicit max token cap even when not required.
  - RAGAS judge path could still inherit `max_tokens` from third-party defaults, triggering OpenAI GPT-5 parameter errors.
  - Importing `evaluation.eval_rag` in tests pulled heavyweight embedding dependencies and broke isolated unit tests.

- Fixes applied:
  - Changed the default judge model to `gpt-5.4-mini`.
  - Set judge requests to `reasoning={"effort": "none"}` and removed explicit max token caps from the direct judge path.
  - Added an OpenAI compatibility proxy for the RAGAS path that rewrites legacy `max_tokens` to `max_completion_tokens` and strips incompatible sampling kwargs.
  - Applied the compatibility proxy after `ragas.llm_factory(...)` returns, so `instructor.from_openai(...)` still receives a real OpenAI client and does not produce a broken `judge_llm.client`.
  - Moved the compatibility helper into `evaluation/judge_client.py` so unit tests can validate the mapping logic without loading embedding dependencies.

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
