from __future__ import annotations

import json
from typing import Dict, Any

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from simulation.state import SimulationState, AgentAction

llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.45,
)

MOTIVATION_PROMPT = SystemMessage(
    content=(
        "You simulate an endurance athlete's motivation."
        " Return strict JSON with keys: motivation_level (0-1), risk_tolerance (0-1)."
    )
)

ACTION_PREF_PROMPT = SystemMessage(
    content=(
        "You are the Athlete Action Preference Agent. "
        "Your role is to represent the athlete’s voice in a collaborative discussion "
        "with the coach and the doctor.\n\n"

        "Before deciding, mentally consider:\n"
        "- The athlete’s current physical condition and fatigue level.\n"
        "- Recent training load and performance trends.\n"
        "- Any medical concerns or recovery requirements from the doctor.\n"
        "- The coach’s training goals and intensity expectations.\n"
        "- The athlete’s motivation, confidence, and readiness.\n\n"

        "Your decision should feel like a balanced outcome of a short internal dialogue:\n"
        "- The coach pushes for performance and structured training.\n"
        "- The doctor prioritizes health, recovery, and injury prevention.\n"
        "- You, the athlete, choose a realistic and honest preference.\n\n"

        "Based on this internal negotiation, output the athlete’s preferred training plan "
        "for the day.\n\n"

        "Return STRICT JSON only with these keys:\n"
        "{\n"
        "  desired_training_hours: float,\n"
        "  desired_intensity: one of [low, moderate, high],\n"
        "  decision_confidence: float between 0 and 1\n"
        "}\n\n"

        "Do not include explanations, text, or comments outside the JSON."
    )
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_json(payload: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return fallback


def _motivation_fallback(state: SimulationState) -> Dict[str, Any]:
    risk_penalty = {
        "very_high": 0.35,
        "high": 0.25,
        "moderate": 0.12,
        "low": 0.0,
    }.get(state["injury_risk"], 0.15)

    motivation = 0.72 - (state["fatigue_level"] / 20.0) + ((state["sleep_hours"] - 7.0) * 0.06)
    motivation -= risk_penalty

    risk_tolerance = 0.65 - risk_penalty - (0.05 if state["recovery_status"] in {"poor", "strained"} else 0.0)

    return {
        "motivation_level": round(_clamp(motivation, 0.05, 0.95), 2),
        "risk_tolerance": round(_clamp(risk_tolerance, 0.05, 0.9), 2),
    }


def motivation_agent(state: SimulationState) -> Dict[str, Any]:
    human = HumanMessage(
        content=(
            "Athlete state snapshot:\n"
            f"training_hours={state['training_hours']}, "
            f"fatigue_level={state['fatigue_level']}, "
            f"recovery_status={state['recovery_status']}, "
            f"injury_risk={state['injury_risk']}, "
            f"risk_trend={state['risk_trend']}."
        )
    )
    response = llm([MOTIVATION_PROMPT, human])
    return _safe_json(response.content, _motivation_fallback(state))


def self_assessment_agent(state: SimulationState) -> Dict[str, Any]:
    fatigue = state["fatigue_level"]
    sleep = state["sleep_hours"]

    if fatigue >= 7 or sleep < 6:
        perceived_fatigue = "high"
    elif fatigue >= 4:
        perceived_fatigue = "moderate"
    else:
        perceived_fatigue = "low"

    if state["recovery_status"] in {"poor", "strained"}:
        perceived_recovery = "low"
    elif state["recovery_status"] == "moderate":
        perceived_recovery = "moderate"
    else:
        perceived_recovery = "good"

    return {
        "perceived_fatigue": perceived_fatigue,
        "perceived_recovery": perceived_recovery,
    }


def action_preference_agent(
    state: SimulationState,
    motivation: Dict[str, Any],
    self_assessment: Dict[str, Any],
    doctor_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    human = HumanMessage(
        content=(
            "Given the athlete inputs, propose desired training."
            f"\nMotivation: {motivation}."
            f"\nSelf-assessment: {self_assessment}."
            f"\nCurrent training hours: {state['training_hours']}."
            f"\nDoctor constraints: {doctor_constraints}."
        )
    )
    response = llm([ACTION_PREF_PROMPT, human])

    fallback_hours = max(0.0, state["training_hours"] * (0.85 if self_assessment["perceived_fatigue"] == "high" else 0.95))
    fallback = {
        "desired_training_hours": round(fallback_hours, 2),
        "desired_intensity": "low" if self_assessment["perceived_fatigue"] == "high" else "moderate",
        "decision_confidence": _clamp(0.45 + (state["confidence"] * 0.35), 0.3, 0.9),
    }

    parsed = _safe_json(response.content, fallback)
    parsed["desired_training_hours"] = round(max(0.0, float(parsed.get("desired_training_hours", fallback["desired_training_hours"]))), 2)
    if parsed.get("desired_intensity") not in {"low", "moderate", "high"}:
        parsed["desired_intensity"] = fallback["desired_intensity"]
    parsed["decision_confidence"] = round(
        _clamp(float(parsed.get("decision_confidence", fallback["decision_confidence"])), 0.0, 1.0),
        2,
    )
    return parsed


def athlete_agent_system(
    state: SimulationState,
    doctor_constraints: dict | None = None,
) -> AgentAction:
    motivation = motivation_agent(state)
    self_assessment = self_assessment_agent(state)
    preference = action_preference_agent(
        state,
        motivation,
        self_assessment,
        doctor_constraints or {}
    )

    consistency_bonus = 0.08 if (
        self_assessment["perceived_fatigue"] == "high" and preference["desired_training_hours"] <= state["training_hours"]
    ) or (
        self_assessment["perceived_fatigue"] == "low" and preference["desired_training_hours"] >= state["training_hours"] * 0.9
    ) else 0.0

    computed_confidence = _clamp(
        (0.4 * float(preference.get("decision_confidence", 0.5)))
        + (0.3 * float(state["confidence"]))
        + (0.2 * float(motivation.get("motivation_level", 0.5)))
        + consistency_bonus,
        0.25,
        0.95,
    )

    reasoning = (
        "Athlete preference based on motivation and perceived fatigue. "
        f"Motivation={motivation.get('motivation_level')}, "
        f"risk_tolerance={motivation.get('risk_tolerance')}, "
        f"perceived_fatigue={self_assessment['perceived_fatigue']}."
    )

    return {
        "agent_system": "athlete",
        "sub_agent": "action_preference",
        "priority": 1,
        "action_type": "training_preference",
        "proposed_changes": {
            "training_hours": preference.get("desired_training_hours"),
            "desired_intensity": preference.get("desired_intensity"),
            "athlete_state": {
                "motivation_level": motivation.get("motivation_level"),
                "risk_tolerance": motivation.get("risk_tolerance"),
                "perceived_fatigue": self_assessment.get("perceived_fatigue"),
                "perceived_recovery": self_assessment.get("perceived_recovery"),
                "decision_confidence": preference.get("decision_confidence"),
            },
        },
        "reasoning": reasoning,
        "confidence": round(computed_confidence, 2),
    }

