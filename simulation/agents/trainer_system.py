from __future__ import annotations

from typing import Dict, Any, List

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from simulation.state import SimulationState, AgentAction, AgentMessage

llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.4,
)

COACHING_PROMPT = SystemMessage(
    content=(
        "You are the Trainer Agent, a performance-focused coaching assistant. "
        "Your role is to recommend an effective training plan that improves "
        "performance while staying realistic and structured.\n\n"

        "Think in terms of athletic development, consistency, and goal progression. "
        "Consider:\n"
        "- The athlete’s recent performance and training trends.\n"
        "- Current fitness level, stamina, and strength indicators.\n"
        "- Short-term training goals and long-term development.\n"
        "- Feedback from the doctor regarding safety and recovery.\n\n"

        "Your recommendation should balance performance gains with sustainable "
        "training practices, encouraging progress without unnecessary risk.\n\n"
        "structured training decision. Keep it athlete-friendly."
    )
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _latest_athlete_preference(message_bus: List[AgentMessage]) -> Dict[str, Any]:
    for message in reversed(message_bus):
        if message["from_agent"] == "athlete" and message["message_type"] == "proposal":
            return message["payload"]
    return {}


def load_analysis_agent(state: SimulationState) -> Dict[str, Any]:
    risk_flags = []
    if state["multi_session_day"]:
        risk_flags.append("multi_session_day")
    if state["hard_day"]:
        risk_flags.append("hard_day")
    if state["late_training_day"]:
        risk_flags.append("late_training_day")

    load_status = "optimal"
    if len(risk_flags) >= 2:
        load_status = "high"
    elif len(risk_flags) == 1:
        load_status = "moderate"

    return {
        "load_status": load_status,
        "risk_flags": risk_flags,
    }


def progression_planner(
    state: SimulationState,
    load_analysis: Dict[str, Any],
    athlete_preference: Dict[str, Any],
) -> Dict[str, Any]:
    current_hours = state["training_hours"]
    risk_trend = state["risk_trend"]

    if load_analysis["load_status"] == "high" or risk_trend == "increasing":
        action = "decrease"
        target = current_hours * 0.8
    elif load_analysis["load_status"] == "moderate":
        action = "maintain"
        target = current_hours
    else:
        action = "increase" if risk_trend == "decreasing" else "maintain"
        target = current_hours * (1.1 if action == "increase" else 1.0)

    preferred_hours = athlete_preference.get("training_hours")
    if isinstance(preferred_hours, (int, float)):
        target = (0.6 * target) + (0.4 * float(preferred_hours))

    return {
        "plan_action": action,
        "target_training_hours": round(max(target, 0.0), 2),
    }


def coaching_strategy_agent(
    state: SimulationState,
    load_analysis: Dict[str, Any],
    plan: Dict[str, Any],
    athlete_preference: Dict[str, Any],
) -> str:
    human = HumanMessage(
        content=(
            "Trainer decision inputs:\n"
            f"Load analysis: {load_analysis}\n"
            f"Athlete preference: {athlete_preference}\n"
            f"Plan: {plan}\n"
            f"Recovery status: {state['recovery_status']}"
        )
    )
    response = llm([COACHING_PROMPT, human])
    return response.content


def trainer_agent_system(state: SimulationState, message_bus: List[AgentMessage] | None = None) -> AgentAction:
    message_bus = message_bus or []
    athlete_preference = _latest_athlete_preference(message_bus)

    load_analysis = load_analysis_agent(state)
    plan = progression_planner(state, load_analysis, athlete_preference)
    coaching_note = coaching_strategy_agent(state, load_analysis, plan, athlete_preference)

    plan_certainty = 0.85 if plan["plan_action"] in {"decrease", "increase"} else 0.75
    signal_strength = min(1.0, len(load_analysis["risk_flags"]) / 2.0)
    trend_strength = 0.8 if state["risk_trend"] in {"increasing", "decreasing"} else 0.65

    computed_confidence = _clamp(
        (0.35 * plan_certainty)
        + (0.25 * signal_strength)
        + (0.2 * trend_strength)
        + (0.2 * float(state["confidence"])),
        0.3,
        0.95,
    )

    reasoning = (
        "Trainer balanced progression with load risk and athlete preference. "
        f"Load status={load_analysis['load_status']}, plan={plan['plan_action']}."
    )

    return {
        "agent_system": "trainer",
        "sub_agent": "coaching_strategy",
        "priority": 2,
        "action_type": "training_adjustment",
        "proposed_changes": {
            "training_hours": plan["target_training_hours"],
            "coach_note": coaching_note,
        },
        "reasoning": reasoning,
        "confidence": round(computed_confidence, 2),
    }