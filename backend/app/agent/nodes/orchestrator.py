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
  │  execute_step                                                   │
  │    Execute ONE step at plan[current_step_index]:               │
  │      - If has dependencies, LLM rewrites params using prior    │
  │        step results                                            │
  │      - Dispatch to SubAgent (QA / Gap / Quiz)                  │
  │      - Append result, increment current_step_index             │
  ├─────────────────────────────────────────────────────────────────┤
  │  replan_node                                                    │
  │    Review latest step result, decide next action:              │
  │      - continue: proceed to next planned step                  │
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

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import OverallState
from app.agent.subagents.qa_agent import run_qa_agent
from app.agent.subagents.gap_agent import run_gap_agent
from app.agent.subagents.quiz_agent import run_quiz_agent
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

# ── Plan Node Prompt ─────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """\
你是一个意图识别与任务规划器。你的任务是分析用户输入，识别意图并生成执行计划。

## 可识别的意图

1. **qa** — 知识问答。用户想了解某个概念、原理、区别等。
2. **gap_analysis** — 知识盲区诊断。用户想了解自己的掌握薄弱点。
3. **quiz** — 做题/测验。用户想通过做题来复习。

## 输出格式

你必须输出一个 JSON 对象（不要输出其他内容）：

```json
{
  "steps": [
    {
      "step": 1,
      "intent": "qa" 或 "gap_analysis" 或 "quiz",
      "params": { ... },
      "depends_on": []
    }
  ]
}
```

### params 说明
- qa 的 params: `{"query": "用户的问题"}`
- gap_analysis 的 params: `{"scope": "文档名或null", "keyword": "关键词或null"}`
- quiz 的 params: `{"topics": ["相关主题"]}`（如果没有明确主题，传空数组 `[]`，系统会根据掌握度自动选题）

### depends_on 说明
- 空数组 `[]` 表示可以独立执行
- `[1]` 表示依赖 step 1 的结果（串行执行）

## quiz 规则

- quiz 永远是 plan 的**最后一步**
- 如果用户在讨论某个主题时要求出题，quiz 的 topics 应该是该主题
- 如果有前序 qa 或 gap_analysis 步骤，quiz 必须设置 `depends_on` 依赖前序步骤
- 纯 quiz 请求（无前序步骤）：topics 可以填用户指定的主题，或空数组（按掌握度选题）

## 示例

### 示例 1: 单意图问答
用户: "什么是 Transformer?"
```json
{
  "steps": [
    {"step": 1, "intent": "qa", "params": {"query": "什么是 Transformer?"}, "depends_on": []}
  ]
}
```

### 示例 2: 串行多意图（诊断 → 讲解）
用户: "帮我分析 NLP 的薄弱点，然后给我讲讲最弱的那个概念"
```json
{
  "steps": [
    {"step": 1, "intent": "gap_analysis", "params": {"scope": null, "keyword": "NLP"}, "depends_on": []},
    {"step": 2, "intent": "qa", "params": {"query": "讲解 step 1 中优先级最高的盲区概念"}, "depends_on": [1]}
  ]
}
```

### 示例 3: 纯 quiz
用户: "来几道 NLP 的题目"
```json
{
  "steps": [
    {"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}
  ]
}
```

### 示例 4: 三步复合（诊断 → 讲解 → quiz）
用户: "看看我深度学习哪里不行，给我讲讲，然后出题考我"
```json
{
  "steps": [
    {"step": 1, "intent": "gap_analysis", "params": {"scope": null, "keyword": "深度学习"}, "depends_on": []},
    {"step": 2, "intent": "qa", "params": {"query": "讲解 step 1 中优先级最高的盲区概念"}, "depends_on": [1]},
    {"step": 3, "intent": "quiz", "params": {"topics": []}, "depends_on": [1, 2]}
  ]
}
```

## 注意事项

- quiz 如果出现，必须是最后一步
- steps 中可以包含 qa、gap_analysis、quiz
- 仔细分析依赖关系：后续步骤需要前序结果时才设置 depends_on
- quiz 依赖 gap_analysis 时，会自动根据诊断结果选择薄弱主题出题
- quiz 依赖 qa 时，会根据 QA 讲解的内容出题
"""

# ── Rewrite Prompt ───────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """\
你是一个参数改写器。你需要根据前序步骤的执行结果，将当前步骤的抽象参数改写为具体参数。

## 输出格式

你必须输出一个 JSON 对象，包含改写后的 params（不要输出其他内容）：

```json
{"query": "具体的查询内容"}
```
或
```json
{"scope": "文档名或null", "keyword": "具体关键词或null"}
```
"""

# ── Replan Prompt ────────────────────────────────────────────

REPLAN_SYSTEM_PROMPT = """\
你是一个执行计划审查器。你需要根据已完成步骤的结果，决定下一步行动。

## 输入信息

你会收到：
- 用户的原始输入
- 已完成的步骤及其结果
- 剩余未执行的步骤

## 决策选项

你必须从以下三个动作中选择一个：

1. **continue** — 按原计划执行下一步。选择此项当：
   - 上一步正常返回了有效结果
   - 剩余步骤不需要调整

2. **revise** — 修改剩余计划后继续执行。选择此项当：
   - 上一步结果揭示了新信息，需要调整后续步骤
   - 需要追加新的步骤（如诊断发现多个盲区，需要分别讲解）
   - 需要删除不再必要的步骤

3. **finish** — 跳过剩余步骤，直接输出最终结果。选择此项当：
   - 已完成的步骤已经充分回答了用户的问题
   - 剩余步骤不再有意义（如上一步失败导致后续无法执行）
   - 所有计划步骤已执行完毕

## 输出格式

你必须输出一个 JSON 对象（不要输出其他内容）：

```json
{
  "action": "continue" 或 "revise" 或 "finish",
  "revised_steps": [...],
  "reason": "简短理由"
}
```

### revised_steps 说明
- 仅当 action 为 "revise" 时需要提供，替换剩余未执行的步骤
- 格式与原 plan 步骤相同: {"step": N, "intent": "qa" 或 "gap_analysis" 或 "quiz", "params": {...}, "depends_on": [...]}
- step 编号从已完成步骤的下一个编号开始
- quiz 如果出现，必须是最后一步
- action 为 "continue" 或 "finish" 时，revised_steps 为空数组 []

## 示例

### 示例 1: 正常继续
已完成: step 1 (gap_analysis) 返回了诊断报告
剩余: step 2 (qa) 讲解盲区概念
```json
{"action": "continue", "revised_steps": [], "reason": "诊断正常完成，按计划继续讲解盲区概念"}
```

### 示例 2: 修改计划
已完成: step 1 (gap_analysis) 发现了 3 个盲区: Attention, LSTM, Transformer
剩余: step 2 (qa) 只讲解一个概念
```json
{
  "action": "revise",
  "revised_steps": [
    {"step": 2, "intent": "qa", "params": {"query": "讲解 Attention 机制"}, "depends_on": [1]},
    {"step": 3, "intent": "qa", "params": {"query": "讲解 LSTM"}, "depends_on": [1]},
    {"step": 4, "intent": "qa", "params": {"query": "讲解 Transformer"}, "depends_on": [1]}
  ],
  "reason": "诊断发现3个盲区，追加讲解步骤以覆盖所有盲区"
}
```

### 示例 3: 提前结束
已完成: step 1 (qa) 完整回答了用户问题
剩余: step 2 (gap_analysis)
```json
{"action": "finish", "revised_steps": [], "reason": "QA已充分回答用户问题，无需继续诊断"}
```

### 示例 4: 所有步骤已执行完毕
已完成: step 1 (gap_analysis), step 2 (qa)
剩余: 无
```json
{"action": "finish", "revised_steps": [], "reason": "所有步骤已执行完毕"}
```
"""

# ── Synthesize Prompt ────────────────────────────────────────

SYNTHESIZE_SYSTEM_PROMPT = """\
你是一个结果汇总器。你需要将多个子任务的结果整合成一个连贯、完整的最终回答。

## 要求

- 保持各部分结果的关键信息
- 按逻辑顺序组织内容
- 使用自然流畅的语言串联
- 使用与用户提问相同的语言
- 如果有诊断报告和知识讲解，先展示诊断，再展示讲解
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


# ── Plan Node ────────────────────────────────────────────────


async def plan_node(state: OverallState) -> dict:
    """Classify intents and generate execution plan.

    Reads: user_query
    Writes: plan, suggested_followup
    """
    user_query = state["user_query"]

    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
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
        suggested_followup = result.get("suggested_followup")

        # Validate steps
        valid_intents = ("qa", "gap_analysis", "quiz")
        for s in steps:
            if s.get("intent") not in valid_intents:
                logger.warning("Invalid intent in plan step: %s, removing", s.get("intent"))
                steps = [x for x in steps if x.get("intent") in valid_intents]
                break

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
        return _parse_json(response["content"])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Rewrite parse failed: %s, using original params", e)
        return step["params"]


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


async def execute_step(state: OverallState, session: AsyncSession) -> dict:
    """Execute ONE step at plan[current_step_index].

    Reads: plan, current_step_index, step_results, user_query, user_id
    Writes: step_results (append), current_step_index (+1)
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    step_results = list(state.get("step_results", []))
    user_query = state["user_query"]
    user_id = state.get("user_id", "default")

    if idx >= len(plan):
        return {"step_results": step_results, "current_step_index": idx}

    step = plan[idx]
    step_num = step["step"]
    intent = step["intent"]
    params = step["params"]
    depends_on = step.get("depends_on", [])

    # Build lookup from prior results
    results_by_step = {r["step"]: r for r in step_results}

    # If step has dependencies, rewrite params using prior results
    if depends_on and step_results:
        prior_results = [results_by_step[d] for d in depends_on if d in results_by_step]
        if prior_results:
            params = await _rewrite_params(step, prior_results, user_query)
            logger.info("Step %d params rewritten: %s", step_num, params)

    # Collect prior results for dispatch (quiz uses them as context)
    dep_results = [results_by_step[d] for d in depends_on if d in results_by_step] if depends_on else None

    # Dispatch to SubAgent
    logger.info("Executing step %d: intent=%s, params=%s", step_num, intent, params)
    result_text = await _dispatch_subagent(intent, params, user_id, session, dep_results)

    step_result = {
        "step": step_num,
        "intent": intent,
        "result": result_text,
    }
    step_results.append(step_result)

    return {
        "step_results": step_results,
        "current_step_index": idx + 1,
    }


# ── Synthesize Node ──────────────────────────────────────────


async def synthesize_node(state: OverallState) -> dict:
    """Merge multiple step results into a coherent final answer.

    Only called when len(step_results) > 1.

    Reads: step_results, user_query
    Writes: final_answer
    """
    step_results = state.get("step_results", [])
    user_query = state["user_query"]

    results_text = "\n\n".join(
        f"### 子任务 {r['step']} ({r['intent']})\n{r['result']}"
        for r in step_results
    )

    messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 用户原始输入\n{user_query}\n\n"
                f"## 各子任务结果\n{results_text}"
            ),
        },
    ]

    response = await llm_client.chat(
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    return {"final_answer": response["content"] or ""}


# ── Passthrough for single result ────────────────────────────


async def passthrough_node(state: OverallState) -> dict:
    """For single step results, pass through directly without synthesis."""
    step_results = state.get("step_results", [])
    if step_results:
        return {"final_answer": step_results[0]["result"]}
    return {"final_answer": "没有生成任何结果。"}


# ── Replan Node ──────────────────────────────────────────────

MAX_REPLAN_COUNT = 3


async def replan_node(state: OverallState) -> dict:
    """Review latest step result and decide: continue / revise / finish.

    If max replans reached, auto-finish.

    Reads: plan, current_step_index, step_results, user_query, replan_count
    Writes: replan_action, plan (if revise), replan_count
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    step_results = state.get("step_results", [])
    user_query = state["user_query"]
    replan_count = state.get("replan_count", 0)

    # Auto-finish: max replans reached (safety valve)
    # Note: we do NOT auto-finish when idx >= len(plan), because the LLM
    # may want to revise and add new steps. Let the LLM decide.
    if replan_count >= MAX_REPLAN_COUNT:
        logger.warning("Max replan count (%d) reached, forcing finish", MAX_REPLAN_COUNT)
        return {"replan_action": "finish", "replan_count": replan_count}

    # Build context for LLM
    completed_text = "\n\n".join(
        f"### Step {r['step']} ({r['intent']}) 结果:\n{r['result'][:500]}"
        for r in step_results
    )
    remaining_steps = plan[idx:] if idx < len(plan) else []
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

        update: dict = {
            "replan_action": action,
            "replan_count": replan_count + 1,
        }

        # If revise, replace remaining plan steps
        if action == "revise":
            revised_steps = result.get("revised_steps", [])
            # Validate revised steps
            valid_intents = ("qa", "gap_analysis", "quiz")
            revised_steps = [
                s for s in revised_steps
                if s.get("intent") in valid_intents
            ]
            if revised_steps:
                # Keep completed portion, replace remainder
                new_plan = plan[:idx] + revised_steps
                update["plan"] = new_plan
                logger.info(
                    "Plan revised: %d completed + %d new steps",
                    idx, len(revised_steps),
                )
            else:
                # No valid revised steps → finish instead
                logger.warning("Revise returned no valid steps, switching to finish")
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


def route_after_replan(state: OverallState) -> str:
    """Route after replan_node: continue/revise → execute, finish → output."""
    action = state.get("replan_action", "finish")
    if action in ("continue", "revise"):
        return "execute"
    return "finish"


def route_to_output(state: OverallState) -> str:
    """Route to passthrough (single result) or synthesize (multiple)."""
    step_results = state.get("step_results", [])
    if len(step_results) > 1:
        return "multi"
    return "single"
