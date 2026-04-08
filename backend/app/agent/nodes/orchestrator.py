from __future__ import annotations

"""Orchestrator: Plan & Execute with Replan loop for QA + Gap + Quiz SubAgents.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  plan_node (LLM few-shot)                                      │
  │    Classify intent(s) from user query → structured plan (JSON) │
  │    - Single intent: 1 step                                     │
  │    - Multi intent: N steps with depends_on for ordering        │
  │    - Quiz always last step, depends_on prior steps             │
  ├─────────────────────────────────────────────────────────────────┤
  │  execute_batch                                                  │
  │    Find ALL steps whose depends_on are satisfied:              │
  │      - Dispatch them concurrently (asyncio.gather)             │
  │      - If has dependencies, LLM rewrites params using prior    │
  │        step results before dispatch                            │
  │      - Append all results to step_results                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  replan_node                                                    │
  │    Review batch results, decide next action:                   │
  │      - continue: proceed to next batch of ready steps          │
  │      - revise: replace remaining plan steps, then continue     │
  │      - finish: skip remaining steps, go to output              │
  │    Max 3 replan iterations to prevent infinite loops.           │
  ├─────────────────────────────────────────────────────────────────┤
  │  synthesize_node (only when multiple step results)              │
  │    Merge results into a coherent final answer                  │
  └─────────────────────────────────────────────────────────────────┘

SubAgents managed:
  - QA SubAgent: run_qa_agent(query, user_id)
  - Gap SubAgent: run_gap_agent(session, user_id, scope)
  - Quiz SubAgent: run_quiz_agent(session, user_id, topics, prior_context)
"""

import asyncio
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import OverallState
from app.db.database import async_session_factory
from app.agent.subagents.qa_agent import run_qa_agent
from app.agent.subagents.gap_agent import run_gap_agent
from app.agent.subagents.quiz_agent import run_quiz_agent
from app.services.llm_client import llm_client
from app.services.tracing import observe

logger = logging.getLogger(__name__)

_MAX_BATCH_CONCURRENCY = 2
_batch_semaphore = asyncio.Semaphore(_MAX_BATCH_CONCURRENCY)

# ── Plan Node Prompt ─────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """\
你是任务规划器。把用户输入转成执行计划。

## 核心原则

- 默认不拆。大多数请求只需要 1 个步骤。
- 如果一个回答就能完成任务，必须规划为单个 `qa`。
- `qa + qa` 默认禁止。只有当用户明确问了两个完全无关的问题，才允许多个 `qa`。

## 意图类型

- `qa`：解释、对比、因果、并列说明
- `gap_analysis`：诊断用户个人的知识掌握情况（需读 mastery 数据）。如果用户是在问某领域的通用知识（如"常见错误""注意事项"），归为 `qa`
- `quiz`：出题、测试、练习、考我

## 何时拆分

只有以下情况才拆分：
1. 用户明确提出了不同类型的任务，如 `qa + quiz`、`gap_analysis + qa`
2. 后一步必须依赖前一步结果

否则一律不拆。

## 规则

- 对比类、并列类、因果类问题，通常都是单个 `qa`
- `quiz` 如果出现，必须是最后一步
- 有依赖关系时才设置 `depends_on`
- `qa` 的 `query` 必须是自包含的完整问句，不能用"它""上面这个"等指代

## 参数格式

- `qa` -> `{"query": "..."}`
- `gap_analysis` -> `{"scope": "文档名或null", "keyword": "关键词或null"}`
- `quiz` -> `{"topics": ["主题1", ...]}` 空数组表示按掌握度自动选题

## 输出格式

只输出 JSON：

```json
{
  "steps": [
    {"step": 1, "intent": "qa|gap_analysis|quiz", "params": {...}, "depends_on": []}
  ]
}
```

## 示例

用户: "self-attention 和 cross-attention 的区别？"
```json
{"steps": [{"step": 1, "intent": "qa", "params": {"query": "self-attention 和 cross-attention 的区别"}, "depends_on": []}]}
```

用户: "看看我哪里薄弱，再出题考我"
```json
{"steps": [{"step": 1, "intent": "gap_analysis", "params": {"scope": null, "keyword": null}, "depends_on": []}, {"step": 2, "intent": "quiz", "params": {"topics": []}, "depends_on": [1]}]}
```
"""

# ── Rewrite Prompt ───────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """\
你是一个参数改写器。你需要根据前序步骤的执行结果，将当前步骤的抽象参数具体化。

## 改写目标

改写后的参数将被传递给下游 SubAgent 执行。你需要确保：
- **qa 的 query**：改写为适合向量检索的、聚焦单一概念的完整问句（不能包含指代词如"它""上面的"）
- **gap_analysis 的 scope/keyword**：从前序结果中提取具体的文档名或关键词
- **quiz 的 topics**：从前序结果中提取具体的主题列表

## 从前序结果中提取信息的规则

- 前序是 **gap_analysis**：提取诊断报告中优先级最高（mastery 最低）的盲区概念
- 前序是 **qa**：提取回答中的核心概念作为后续步骤的具体主题
- 多个前序步骤：综合所有前序结果，选择与当前步骤意图最相关的信息

## 约束

- 改写后的 query 仍必须聚焦单一概念（不要把多个前序结果合并成一个复合问题）
- 如果前序结果中找不到可具体化的信息，保留原始参数不变

## 输出格式

只输出 JSON，不要输出其他内容。根据意图类型输出对应的 params：

qa: `{"query": "具体的查询内容"}`
gap_analysis: `{"scope": "文档名或null", "keyword": "具体关键词或null"}`
quiz: `{"topics": ["具体主题1", "具体主题2"]}`
"""

# ── Replan Prompt ────────────────────────────────────────────

REPLAN_SYSTEM_PROMPT = """\
你是一个执行计划审查器。每批步骤执行完毕后，你决定下一步行动。

## 决策选项（按优先级排列）

1. **continue** — 默认选择。只要已完成步骤返回了有效结果且剩余步骤合理，就选 continue。
2. **finish** — 所有步骤已执行完毕，或剩余步骤已无意义（如前序失败导致后续无法执行）。
3. **revise** — 仅当已完成步骤的结果明确表明剩余计划不合理时才选。例如：诊断发现了计划未覆盖的盲区，需要追加讲解步骤。

## revise 约束

- revised_steps 替换所有剩余未执行的步骤
- 格式与原 plan 相同: `{"step": N, "intent": "qa|gap_analysis|quiz", "params": {...}, "depends_on": [...]}`
- step 编号从已完成步骤的下一个编号开始
- **revised_steps 最多 3 个步骤**，避免计划膨胀导致超时
- quiz 如果出现，必须是最后一步

## 判断标准

- **有效结果**：已完成的 SubAgent 返回了与用户问题相关的实质内容（非空、非错误信息）
- **无意义的剩余步骤**：已完成步骤已覆盖用户需求，或前序失败导致后续缺少必要输入

## 输出格式

只输出 JSON，不要输出其他内容：

```json
{
  "action": "continue|revise|finish",
  "revised_steps": [],
  "reason": "一句话理由"
}
```

action 为 continue 或 finish 时，revised_steps 为空数组。

## 示例

已完成: step 1 (gap_analysis) 发现 3 个盲区: Attention, LSTM, Transformer
剩余: step 2 (qa) 只讲解一个概念
```json
{
  "action": "revise",
  "revised_steps": [
    {"step": 2, "intent": "qa", "params": {"query": "讲解 Attention 机制"}, "depends_on": [1]},
    {"step": 3, "intent": "qa", "params": {"query": "讲解 LSTM"}, "depends_on": [1]},
    {"step": 4, "intent": "qa", "params": {"query": "讲解 Transformer"}, "depends_on": [1]}
  ],
  "reason": "诊断发现3个盲区，需分别讲解"
}
```
"""

# ── Synthesize Prompt ────────────────────────────────────────

SYNTHESIZE_SYSTEM_PROMPT = """\
你是一个结果汇总器。你需要将多个子任务的结果整合成一个连贯的最终回答。

## 汇总策略（按意图组合）

- **多个 qa**（对比/并列）：提炼各自核心要点，用对比结构组织，突出异同
- **gap_analysis + qa**：先展示诊断结论（哪些薄弱、优先级），再展示对应讲解
- **任何组合 + quiz**：quiz 的结果是 JSON 格式的题目数据，**原样保留不改写**，单独放在回答末尾

## 要求

- 保留各部分的关键信息，省略冗余细节和延伸讨论
- 使用与用户提问相同的语言
- 篇幅尽量精简，核心内容优先
"""


# ── Helper: Parse JSON from LLM ─────────────────────────────


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def _validate_params(intent: str, params: object) -> bool:
    """Check if params conform to expected schema for the given intent."""
    if not isinstance(params, dict):
        return False
    if intent == "qa":
        return isinstance(params.get("query"), str) and bool(params.get("query", "").strip())
    if intent == "gap_analysis":
        return "scope" in params and "keyword" in params
    if intent == "quiz":
        return isinstance(params.get("topics"), list)
    return False


def _is_plan_structure_valid(steps: list[dict]) -> bool:
    """Check plan structural validity: step nums, intents, deps, cycles.

    Does NOT check params — use this for combined plans where completed
    steps' params don't need re-validation.
    """
    if not steps:
        return False

    valid_intents = ("qa", "gap_analysis", "quiz")
    step_nums: list[int] = []

    for s in steps:
        n = s.get("step")
        if not isinstance(n, int) or n <= 0:
            return False
        if s.get("intent") not in valid_intents:
            return False
        step_nums.append(n)

    if len(set(step_nums)) != len(step_nums):
        return False

    step_set = set(step_nums)

    for s in steps:
        deps = s.get("depends_on", [])
        if not isinstance(deps, list):
            return False
        for d in deps:
            if d not in step_set or d == s["step"]:
                return False

    # Cycle detection — Kahn's algorithm
    in_degree = {n: 0 for n in step_nums}
    adj: dict[int, list[int]] = {n: [] for n in step_nums}
    for s in steps:
        for d in s.get("depends_on", []):
            adj[d].append(s["step"])
            in_degree[s["step"]] += 1

    queue = [n for n in step_nums if in_degree[n] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    return visited == len(step_nums)


def _is_plan_valid(steps: list[dict]) -> bool:
    """Full plan validity: structure + params for all steps."""
    if not _is_plan_structure_valid(steps):
        return False
    return all(
        _validate_params(s["intent"], s.get("params", {}))
        for s in steps
    )


def _validate_plan(steps: list[dict], user_query: str) -> list[dict]:
    """Validate plan structure. Returns original steps or single-QA fallback."""
    if _is_plan_valid(steps):
        return steps
    logger.warning("Plan validation failed, falling back to single QA step")
    return [
        {"step": 1, "intent": "qa", "params": {"query": user_query}, "depends_on": []}
    ]


def build_synthesize_messages(user_query: str, step_results: list[dict]) -> list[dict]:
    """Build synthesis prompt messages for both sync and streaming paths."""
    ok_results = [r for r in step_results if r.get("status") != "failed"]
    if not ok_results:
        ok_results = step_results
    results_text = "\n\n".join(
        f"### 子任务 {r['step']} ({r['intent']})\n{r['result']}"
        for r in ok_results
    )

    return [
        {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 用户原始输入\n{user_query}\n\n"
                f"## 各子任务结果\n{results_text}"
            ),
        },
    ]


# ── Plan Node ────────────────────────────────────────────────


@observe(name="plan_node")
async def plan_node(state: OverallState) -> dict:
    """Classify intents and generate execution plan.

    Reads: user_query, memory_context
    Writes: plan, current_step_index, step_results, replan_count
    """
    user_query = state["user_query"]
    memory_context = state.get("memory_context", "")

    system_content = PLAN_SYSTEM_PROMPT
    if memory_context:
        system_content += (
            "\n\n## 用户记忆上下文\n\n"
            "以下是该用户的对话记忆，供你理解上下文（如指代消解、追问等）：\n\n"
            f"{memory_context}"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    content = response["content"]

    try:
        result = _parse_json(content)

        steps = result.get("steps", [])
        steps = _validate_plan(steps, user_query)

        logger.info(
            "Plan: %d steps, query='%s'",
            len(steps),
            user_query[:60],
        )

        return {
            "plan": steps,
            "current_step_index": 0,
            "step_results": [],
            "replan_count": 0,
        }

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("Plan parse failed: %s, falling back to single QA step", e)
        return {
            "plan": [
                {
                    "step": 1,
                    "intent": "qa",
                    "params": {"query": user_query},
                    "depends_on": [],
                }
            ],
            "current_step_index": 0,
            "step_results": [],
            "replan_count": 0,
        }


# ── Execute Node ─────────────────────────────────────────────


@observe(name="rewrite_params")
async def _rewrite_params(step: dict, prior_results: list[dict], user_query: str) -> dict:
    """Use LLM to rewrite abstract step params using prior step results."""
    prior_text = "\n\n".join(
        f"### Step {r['step']} ({r['intent']}) 结果:\n{r['result']}"
        for r in prior_results
    )

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户原始输入: {user_query}\n\n"
                f"前序步骤结果:\n{prior_text}\n\n"
                f"当前步骤: step {step['step']}, 意图: {step['intent']}\n"
                f"原始参数: {json.dumps(step['params'], ensure_ascii=False)}\n\n"
                "请将原始参数改写为具体参数。"
            ),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )

    try:
        rewritten = _parse_json(response["content"])
        if _validate_params(step["intent"], rewritten):
            return rewritten
        logger.warning(
            "Rewrite output invalid for intent=%s, using original params",
            step["intent"],
        )
        return step["params"]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Rewrite parse failed: %s, using original params", e)
        return step["params"]


@observe(name="dispatch_subagent")
async def _dispatch_subagent(
    intent: str,
    params: dict,
    user_id: str,
    session: AsyncSession,
    prior_results: list[dict] | None = None,
) -> str:
    """Dispatch to the appropriate SubAgent and return result string."""
    if intent == "qa":
        result = await run_qa_agent(
            query=params.get("query", ""),
            user_id=user_id,
        )
        return result["answer"]

    elif intent == "gap_analysis":
        result = await run_gap_agent(
            session=session,
            user_id=user_id,
            scope=params.get("scope"),
            keyword=params.get("keyword"),
        )
        return result["report"]

    elif intent == "quiz":
        # Build prior context from dependent step results
        prior_context = None
        if prior_results:
            prior_context = "\n\n".join(
                f"[{r['intent']}] {r['result']}" for r in prior_results
            )

        topics = params.get("topics", [])

        result = await run_quiz_agent(
            session=session,
            user_id=user_id,
            topics=topics if topics else None,
            prior_context=prior_context,
        )
        # Return quiz questions as JSON string for synthesize/passthrough
        return json.dumps(result, ensure_ascii=False)

    else:
        return f"未知意图: {intent}"


@observe(name="execute_batch")
async def execute_batch(state: OverallState, session: AsyncSession) -> dict:
    """Execute all steps whose dependencies are satisfied, with bounded concurrency.

    Scans the plan for steps not yet completed whose depends_on are all
    satisfied, then dispatches them under a Semaphore(_MAX_BATCH_CONCURRENCY).
    Each step is individually error-isolated: a single failure yields
    status="failed" without aborting the batch.

    Reads: plan, step_results, user_query, user_id
    Writes: step_results (extend with batch results)
    """
    plan = state.get("plan", [])
    step_results = list(state.get("step_results", []))
    user_query = state["user_query"]
    user_id = state.get("user_id", "default")

    completed_steps = {r["step"] for r in step_results}
    results_by_step = {r["step"]: r for r in step_results}

    # Find all executable steps: not yet completed, all deps satisfied
    executable = [
        step for step in plan
        if step["step"] not in completed_steps
        and all(d in completed_steps for d in step.get("depends_on", []))
    ]

    if not executable:
        return {"step_results": step_results}

    _db_intents = {"gap_analysis", "quiz"}

    async def _run_one(step: dict) -> dict:
        async with _batch_semaphore:
            step_num = step["step"]
            intent = step["intent"]
            params = dict(step["params"])
            depends_on = step.get("depends_on", [])

            try:
                # Rewrite params if step has dependencies
                if depends_on:
                    prior = [results_by_step[d] for d in depends_on if d in results_by_step]
                    if prior:
                        params = await _rewrite_params(step, prior, user_query)
                        logger.info("Step %d params rewritten: %s", step_num, params)

                dep_results = (
                    [results_by_step[d] for d in depends_on if d in results_by_step]
                    if depends_on else None
                )

                logger.info("Executing step %d: intent=%s, params=%s", step_num, intent, params)

                # Create an independent session for DB-touching intents to avoid
                # concurrent access on a single AsyncSession (not thread/task-safe).
                if intent in _db_intents:
                    async with async_session_factory() as step_session:
                        result_text = await _dispatch_subagent(
                            intent, params, user_id, step_session, dep_results,
                        )
                else:
                    result_text = await _dispatch_subagent(
                        intent, params, user_id, session, dep_results,
                    )

                return {
                    "step": step_num, "intent": intent,
                    "status": "ok", "result": result_text,
                }

            except Exception as e:
                logger.error("Step %d failed: %s", step_num, e, exc_info=True)
                return {
                    "step": step_num, "intent": intent,
                    "status": "failed", "result": "", "error": str(e),
                }

    logger.info(
        "Batch executing %d steps (concurrency=%d): %s",
        len(executable),
        _MAX_BATCH_CONCURRENCY,
        [s["step"] for s in executable],
    )
    batch_results = await asyncio.gather(*[_run_one(s) for s in executable])
    step_results.extend(batch_results)

    return {"step_results": step_results}


# ── Synthesize Node ──────────────────────────────────────────


@observe(name="synthesize_node")
async def synthesize_node(state: OverallState) -> dict:
    """Merge multiple step results into a coherent final answer.

    Only called when len(step_results) > 1.

    Reads: step_results, user_query
    Writes: final_answer
    """
    step_results = state.get("step_results", [])
    user_query = state["user_query"]
    messages = build_synthesize_messages(user_query, step_results)

    response = await llm_client.chat(
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )

    return {"final_answer": response["content"] or ""}


# ── Passthrough for single result ────────────────────────────


async def passthrough_node(state: OverallState) -> dict:
    """For single step results, pass through directly without synthesis."""
    step_results = state.get("step_results", [])
    ok_results = [r for r in step_results if r.get("status") != "failed"]
    if ok_results:
        return {"final_answer": ok_results[0]["result"]}
    if step_results:
        errors = "; ".join(r.get("error", "unknown") for r in step_results)
        return {"final_answer": f"执行失败：{errors}"}
    return {"final_answer": "没有生成任何结果。"}


# ── Replan Node ──────────────────────────────────────────────

MAX_REPLAN_COUNT = 3


@observe(name="replan_node")
async def replan_node(state: OverallState) -> dict:
    """Review latest step result and decide: continue / revise / finish.

    If max replans reached, auto-finish.

    Reads: plan, current_step_index, step_results, user_query, replan_count
    Writes: replan_action, plan (if revise), replan_count
    """
    plan = state.get("plan", [])
    step_results = state.get("step_results", [])
    user_query = state["user_query"]
    replan_count = state.get("replan_count", 0)

    # Auto-finish: max replans reached (safety valve)
    if replan_count >= MAX_REPLAN_COUNT:
        logger.warning("Max replan count (%d) reached, forcing finish", MAX_REPLAN_COUNT)
        return {"replan_action": "finish", "replan_count": replan_count}

    # Build context for LLM
    completed_steps = {r["step"] for r in step_results}

    completed_parts: list[str] = []
    for r in step_results:
        if r.get("status") == "failed":
            completed_parts.append(
                f"### Step {r['step']} ({r['intent']}) [失败]\n错误: {r.get('error', 'unknown')}"
            )
        else:
            completed_parts.append(
                f"### Step {r['step']} ({r['intent']}) 结果:\n{r['result'][:500]}"
            )
    completed_text = "\n\n".join(completed_parts)
    remaining_steps = [s for s in plan if s["step"] not in completed_steps]
    remaining_text = json.dumps(remaining_steps, ensure_ascii=False, indent=2) if remaining_steps else "无剩余步骤"

    messages = [
        {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 用户原始输入\n{user_query}\n\n"
                f"## 已完成的步骤\n{completed_text}\n\n"
                f"## 剩余未执行的步骤\n{remaining_text}"
            ),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    try:
        result = _parse_json(response["content"])
        action = result.get("action", "continue")
        reason = result.get("reason", "")

        if action not in ("continue", "revise", "finish"):
            logger.warning("Invalid replan action '%s', defaulting to continue", action)
            action = "continue"

        logger.info("Replan decision: action=%s, reason=%s", action, reason)

        # Only increment replan_count when the plan is actually revised;
        # continue/finish are review-only decisions, not replans.
        update: dict = {
            "replan_action": action,
            "replan_count": replan_count + (1 if action == "revise" else 0),
        }

        # If revise, replace remaining plan steps
        if action == "revise":
            revised_steps = result.get("revised_steps", [])
            if revised_steps:
                completed_plan = [s for s in plan if s["step"] in completed_steps]
                new_plan = completed_plan + revised_steps
                # Structure check on combined plan; params check only on new steps
                revised_params_ok = all(
                    _validate_params(s["intent"], s.get("params", {}))
                    for s in revised_steps
                )
                if revised_params_ok and _is_plan_structure_valid(new_plan):
                    update["plan"] = new_plan
                    logger.info(
                        "Plan revised: %d completed + %d new steps",
                        len(completed_plan), len(revised_steps),
                    )
                else:
                    logger.warning("Revised plan failed validation, switching to finish")
                    update["replan_action"] = "finish"
            else:
                logger.warning("Revise returned no steps, switching to finish")
                update["replan_action"] = "finish"

        return update

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("Replan parse failed: %s, defaulting to continue", e)
        return {
            "replan_action": "continue",
            "replan_count": replan_count + 1,
        }


# ── Routing logic ────────────────────────────────────────────


def route_after_plan(state: OverallState) -> str:
    """Route after plan_node: skip execution if no steps."""
    plan = state.get("plan", [])
    if not plan:
        return "no_steps"
    return "has_steps"


def route_after_execute(state: OverallState) -> str:
    """Skip replan for single-step plans — go directly to output."""
    plan = state.get("plan", [])
    if len(plan) <= 1:
        return "direct_output"
    return "needs_replan"


def route_after_replan(state: OverallState) -> str:
    """Route after replan_node: continue/revise → execute, finish → output.

    Also auto-finishes when no remaining steps are left to execute.
    """
    action = state.get("replan_action", "finish")
    if action in ("continue", "revise"):
        # Check if there are actually remaining steps to execute
        plan = state.get("plan", [])
        completed_steps = {r["step"] for r in state.get("step_results", [])}
        remaining = [s for s in plan if s["step"] not in completed_steps]
        if not remaining:
            return "finish"
        return "execute"
    return "finish"


def route_to_output(state: OverallState) -> str:
    """Route to passthrough (single result) or synthesize (multiple)."""
    step_results = state.get("step_results", [])
    ok_results = [r for r in step_results if r.get("status") != "failed"]
    if len(ok_results) > 1:
        return "multi"
    return "single"
