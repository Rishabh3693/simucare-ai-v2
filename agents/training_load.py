from typing import Dict, Any

def training_load_agent(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates training load and fatigue risk using deterministic rules.
    Input: selected_features dict (from Feature Selector)
    Output: structured training load analysis
    """

    training = features.get("training_load", {})
    context = features.get("context", {})

    findings = []
    risk_flags = []

    # -----------------------------
    # 1. Acute vs Chronic Load (ACR)
    # -----------------------------
    acr = training.get("acr_training_load")

    if acr is not None:
        if acr > 1.5:
            findings.append("Acute training load is significantly higher than chronic baseline.")
            risk_flags.append("high_acr")
        elif acr < 0.8:
            findings.append("Training load is below recent baseline (possible detraining).")
        else:
            findings.append("Acute to chronic training load ratio is within optimal range.")

    # -----------------------------
    # 2. Acute Training Volume
    # -----------------------------
    acute_load = training.get("acute_training_hours_7d")

    if acute_load is not None:
        if acute_load > 8:
            findings.append("High total training volume accumulated over the past 7 days.")
            risk_flags.append("high_acute_volume")
        elif acute_load < 3:
            findings.append("Low recent training volume detected.")

    # -----------------------------
    # 3. Session Density
    # -----------------------------
    sessions = training.get("session_count")

    if sessions is not None and sessions > 1:
        findings.append("Multiple training sessions completed in a single day.")
        risk_flags.append("multi_session_day")

    # -----------------------------
    # 4. Hard Session Exposure
    # -----------------------------
    hard_sessions = training.get("hard_sessions") or context.get("hard_sessions")

    if hard_sessions is not None and hard_sessions > 0:
        findings.append("High-intensity training session detected.")
        risk_flags.append("hard_session")

    # -----------------------------
    # 5. Fatigue Classification
    # -----------------------------
    if len(risk_flags) >= 3:
        load_status = "high"
        fatigue_risk = "elevated"
    elif len(risk_flags) == 2:
        load_status = "moderate"
        fatigue_risk = "moderate"
    else:
        load_status = "optimal"
        fatigue_risk = "low"

    # -----------------------------
    # 6. Confidence Estimation
    # -----------------------------
    signal_count = sum(
        1 for v in [
            acr,
            acute_load,
            sessions,
            hard_sessions
        ] if v is not None
    )

    confidence = min(0.9, 0.4 + 0.15 * signal_count)

    # -----------------------------
    # 7. Structured Output
    # -----------------------------
    return {
        "load_status": load_status,
        "fatigue_risk": fatigue_risk,
        "findings": findings,
        "risk_flags": risk_flags,
        "signals_used": signal_count,
        "confidence": round(confidence, 2)
    }
