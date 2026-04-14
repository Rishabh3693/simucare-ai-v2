from __future__ import annotations

from typing import Dict, Any, List

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from simulation.state import SimulationState, AgentAction, AgentMessage

llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.3,
)

CLINICAL_PROMPT = SystemMessage(
    content=(
        "You are the Doctor Agent, a medical safety assistant for the athlete. "
        "Your role is to evaluate the athlete’s condition and recommend a safe "
        "training approach while preventing injury and overexertion.\n\n"

        "Think in terms of medical risk, recovery, and long-term health. "
        "Consider:\n"
        "- Current fatigue, soreness, or injury indicators.\n"
        "- Recent training load and recovery time.\n"
        "- Stress, sleep, and overall physical condition.\n"
        "- Whether the athlete is fit for low, moderate, or high intensity.\n\n"

        "Your recommendation should reflect a careful, health-first mindset, "
        "even if it means reducing intensity or training hours.\n\n"
        "Provide a concise medical explanation for your recommendation, "
    )
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _latest_trainer_plan(message_bus: List[AgentMessage]) -> Dict[str, Any]:
    for message in reversed(message_bus):
        if message["from_agent"] == "trainer" and message["message_type"] == "proposal":
            return message["payload"]
    return {}


def physiological_risk_agent(state: SimulationState) -> Dict[str, Any]:
    flags = []
    severity = "low"

    if state["hrv_deviation"] <= -10:
        flags.append("hrv_suppression")
    if state["rhr_deviation"] >= 5:
        flags.append("rhr_spike")
    if state["injury_risk"] in {"high", "very_high"}:
        flags.append("elevated_risk_level")

    if len(flags) >= 2:
        severity = "high"
    elif len(flags) == 1:
        severity = "moderate"

    return {
        "risk_flags": flags,
        "severity": severity,
    }


def medical_policy_agent(risk: Dict[str, Any], trainer_plan: Dict[str, Any]) -> Dict[str, Any]:
    plan_hours = trainer_plan.get("training_hours")

    if risk["severity"] == "high":
        return {
            "mandatory_rest": True,
            "max_training_hours": 0.0,
        }
    if risk["severity"] == "moderate":
        safe_cap = 0.5
        if isinstance(plan_hours, (int, float)):
            safe_cap = min(safe_cap, float(plan_hours))
        return {
            "mandatory_rest": False,
            "max_training_hours": safe_cap,
        }
    return {
        "mandatory_rest": False,
        "max_training_hours": float(plan_hours) if isinstance(plan_hours, (int, float)) else None,
    }


def clinical_explanation_agent(state: SimulationState, risk: Dict[str, Any], trainer_plan: Dict[str, Any]) -> str:
    human = HumanMessage(
        content=(
            "Provide a short medical safety explanation based on:"
            f"\nRisk: {risk}"
            f"\nTrainer plan: {trainer_plan}"
            f"\nRecovery status: {state['recovery_status']}"
        )
    )
    response = llm([CLINICAL_PROMPT, human])
    return response.content


def _doctor_confidence(state: SimulationState, risk: Dict[str, Any], policy: Dict[str, Any]) -> float:
    severity_score = {"low": 0.5, "moderate": 0.75, "high": 0.95}[risk["severity"]]
    signal_strength = min(1.0, len(risk["risk_flags"]) / 3.0)
    decisiveness = 0.95 if policy["mandatory_rest"] else (0.85 if policy["max_training_hours"] is not None else 0.7)

    return round(
        _clamp(
            (0.35 * severity_score)
            + (0.25 * signal_strength)
            + (0.2 * decisiveness)
            + (0.2 * float(state["confidence"])),
            0.35,
            0.99,
        ),
        2,
    )


def doctor_agent_system(state: SimulationState, message_bus: List[AgentMessage] | None = None) -> AgentAction:
    message_bus = message_bus or []
    trainer_plan = _latest_trainer_plan(message_bus)

    risk = physiological_risk_agent(state)
    policy = medical_policy_agent(risk, trainer_plan)
    explanation = clinical_explanation_agent(state, risk, trainer_plan)

    proposed_changes: Dict[str, Any] = {
        "clinical_note": explanation,
        "medical_constraints": {
            "mandatory_rest": policy["mandatory_rest"],
            "max_training_hours": policy["max_training_hours"],
        },
    }
    action_type = "medical_clearance"

    if policy["mandatory_rest"]:
        action_type = "mandatory_rest"
        proposed_changes["training_hours"] = 0.0
    elif policy["max_training_hours"] is not None:
        action_type = "training_cap"
        proposed_changes["training_hours"] = policy["max_training_hours"]

    reasoning = (
        "Doctor safety decision based on physiological signals and trainer plan. "
        f"severity={risk['severity']}, flags={risk['risk_flags']}"
    )

    return {
        "agent_system": "doctor",
        "sub_agent": "clinical_explanation",
        "priority": 3,
        "action_type": action_type,
        "proposed_changes": proposed_changes,
        "reasoning": reasoning,
        "confidence": _doctor_confidence(state, risk, policy),
    }