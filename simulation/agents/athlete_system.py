from __future__ import annotations

import json
from typing import Dict, Any

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from simulation.state import SimulationState

# ----------------------------------------
# LLM CONFIG
# ----------------------------------------
llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.6,
)

# ----------------------------------------
# UTILS
# ----------------------------------------
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_json(payload: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(payload)
    except:
        return fallback


# ----------------------------------------
# MOTIVATION AGENT (LLM + fallback)
# ----------------------------------------
def _motivation_fallback(state: SimulationState) -> Dict[str, Any]:
    fatigue = state.get("fatigue_level", 5)
    sleep = state.get("sleep_hours", 7)

    motivation = 0.7 - (fatigue / 20.0) + ((sleep - 7) * 0.05)
    risk_tolerance = 0.6 - (fatigue / 25.0)

    return {
        "motivation_level": round(_clamp(motivation, 0.05, 0.95), 2),
        "risk_tolerance": round(_clamp(risk_tolerance, 0.05, 0.9), 2),
    }


def motivation_agent(state: SimulationState) -> Dict[str, Any]:
    prompt = HumanMessage(
        content=(
            f"fatigue={state.get('fatigue_level')}, "
            f"sleep={state.get('sleep_hours')}, "
            f"risk={state.get('injury_risk')}"
        )
    )

    system = SystemMessage(
        content="""
Simulate athlete motivation.
Return JSON:
{ motivation_level: float (0-1), risk_tolerance: float (0-1) }
"""
    )

    response = llm([system, prompt])
    return _safe_json(response.content, _motivation_fallback(state))


# ----------------------------------------
# SELF PERCEPTION
# ----------------------------------------
def self_assessment_agent(state: SimulationState) -> Dict[str, Any]:
    fatigue = state.get("fatigue_level", 5)
    sleep = state.get("sleep_hours", 7)

    if fatigue >= 7 or sleep < 6:
        perceived_fatigue = "high"
    elif fatigue >= 4:
        perceived_fatigue = "moderate"
    else:
        perceived_fatigue = "low"

    recovery_status = state.get("recovery_status")

    if recovery_status in {"poor", "strained"}:
        perceived_recovery = "low"
    elif recovery_status == "moderate":
        perceived_recovery = "moderate"
    else:
        perceived_recovery = "good"

    return {
        "perceived_fatigue": perceived_fatigue,
        "perceived_recovery": perceived_recovery,
    }


# ----------------------------------------
# INTERNAL STATE (USED BY DIALOGUE)
# ----------------------------------------
def athlete_internal_state(state: SimulationState) -> Dict[str, Any]:
    return {
        "motivation": motivation_agent(state),
        "perception": self_assessment_agent(state),
    }


# ----------------------------------------
# 🔥 STRONG DATA CONTEXT BUILDER
# ----------------------------------------
def build_context_text(state, analysis, graph_context):
    return f"""
ATHLETE METRICS:
- Training Hours: {state.get('training_hours')}
- Sleep Hours: {state.get('sleep_hours')}
- Fatigue Level: {state.get('fatigue_level')}
- HRV Deviation: {state.get('hrv_deviation')}
- RHR Deviation: {state.get('rhr_deviation')}
- Sleep Debt: {state.get('sleep_debt')}

ANALYSIS:
- Load Status: {analysis['training_load'].get('load_status')}
- Recovery Status: {analysis['recovery'].get('recovery_status')}
- Risk Level: {analysis['risk'].get('risk_level')}

GRAPH RELATIONSHIPS:
{graph_context}
"""
# ----------------------------------------
# MEMORY BUILDER 
# ----------------------------------------
def build_memory_text(history):
    if not history:
        return "No recent history."

    lines = []
    for i, day in enumerate(history[-3:]):
        lines.append(
            f"Day-{i+1}: fatigue={day.get('fatigue_level')}, "
            f"recovery={day.get('recovery_status')}, "
            f"risk={day.get('injury_risk')}"
        )

    return "\n".join(lines)
# ----------------------------------------
# MODERATOR AGENT (ADD THIS)
# ----------------------------------------
def moderator_agent(state, analysis, history):

    memory_text = build_memory_text(history)

    prompt = f"""
Current:
fatigue={state.get('fatigue_level')}
risk={analysis['risk'].get('risk_level')}

History:
{memory_text}

Decide if intervention is needed.

Rules:
- interrupt ONLY if trend or risk justifies
- return JSON ONLY

Output:
{{
  "interrupt": true/false,
  "who": "coach" or "doctor",
  "reason": "short reason"
}}
"""

    try:
        res = llm.invoke([
            SystemMessage(content="You are a decision agent."),
            HumanMessage(content=prompt)
        ])

        return json.loads(res.content)

    except:
        return {
            "interrupt": False,
            "who": None,
            "reason": ""
        }
# ----------------------------------------
# 🤖 MULTI-AGENT DIALOGUE (REAL AI)
# ----------------------------------------
def athlete_dialogue_agent(state, analysis, graph_context, history):

    memory_text = build_memory_text(history)

    base_context = f"""
fatigue={state.get('fatigue_level')}
sleep={state.get('sleep_hours')}
load={analysis['training_load'].get('load_status')}
recovery={analysis['recovery'].get('recovery_status')}
risk={analysis['risk'].get('risk_level')}

History:
{memory_text}
"""

    # 🔥 STRICT RESPONSE FORMAT
    SHORT_RULE = """
RULES:
- Speak in ONE short sentence
- Max 15 words
- No explanations
- No metrics
- Sound natural and human
"""

    def call_llm(messages, role):
        try:
            res = llm.invoke(messages)
            text = (res.content or "").strip()

            # 🔥 HARD TRIM SAFETY
            if len(text.split()) > 20:
                text = " ".join(text.split()[:20])

            print(f"\n--- {role} TURN ---")
            print("OUTPUT:", text)

            if not text:
                raise ValueError("Empty")

            return text

        except Exception as e:
            print(f"{role} ERROR:", e)
            return None

    # ---------------- TURN 1: ATHLETE ----------------
    athlete_msg = call_llm([
        SystemMessage(content=f"""
You are the athlete.
{SHORT_RULE}
Speak about how you feel.
"""),
        HumanMessage(content=base_context)
    ], "ATHLETE")

    # ---------------- TURN 2: COACH ----------------
    coach_msg = call_llm([
        SystemMessage(content=f"""
You are the coach.
{SHORT_RULE}
Give training advice.
"""),
        HumanMessage(content=f"""
Athlete: "{athlete_msg}"
{base_context}
""")
    ], "COACH")

    # ---------------- MODERATOR ----------------
    decision = moderator_agent(state, analysis, history)

    interruption = None

    if decision["interrupt"]:

        if decision["who"] == "doctor":
            interruption = call_llm([
                SystemMessage(content=f"""
You are a doctor.
{SHORT_RULE}
Intervene clearly and firmly.
"""),
                HumanMessage(content=f"""
Athlete: "{athlete_msg}"
Coach: "{coach_msg}"
Reason: {decision['reason']}
""")
            ], "DOCTOR INTERRUPTION")

        elif decision["who"] == "coach":
            interruption = call_llm([
                SystemMessage(content=f"""
You are a coach.
{SHORT_RULE}
Override the plan.
"""),
                HumanMessage(content=f"""
Athlete: "{athlete_msg}"
Reason: {decision['reason']}
""")
            ], "COACH INTERRUPTION")

    # ---------------- TURN 3: DOCTOR ----------------
    doctor_msg = call_llm([
        SystemMessage(content=f"""
You are the sports doctor.
{SHORT_RULE}
Give a risk statement.
"""),
        HumanMessage(content=f"""
Athlete: "{athlete_msg}"
Coach: "{coach_msg}"
""")
    ], "DOCTOR")

    # ---------------- TURN 4: ATHLETE REPLY ----------------
    athlete_reply = call_llm([
        SystemMessage(content=f"""
You are the athlete.
{SHORT_RULE}
Respond to coach and doctor.
"""),
        HumanMessage(content=f"""
Coach: "{coach_msg}"
Doctor: "{doctor_msg}"
""")
    ], "ATHLETE REPLY")

    # ---------------- FINAL OUTPUT ----------------
    conversation = [
        {"role": "athlete", "text": athlete_msg or "I feel tired."},
        {"role": "coach", "text": coach_msg or "Train lightly today."},
    ]

    if interruption:
        conversation.append({
            "role": decision["who"],
            "text": interruption,
            "type": "interruption"
        })

    conversation.append({
        "role": "doctor",
        "text": doctor_msg or "Be cautious today."
    })

    conversation.append({
        "role": "athlete",
        "text": athlete_reply or "Okay, I’ll take it easy."
    })

    return {
        "conversation": conversation,
        "interruption_meta": decision
    }