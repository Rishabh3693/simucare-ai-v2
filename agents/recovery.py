from typing import Dict, Any

def recovery_agent(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates athlete recovery state using sleep and physiological signals.
    Input: selected_features dict (from Feature Selector)
    Output: structured recovery analysis
    """

    sleep = features.get("sleep", {})
    recovery = features.get("recovery", {})
    context = features.get("context", {})
    lagged = features.get("lagged", {})

    findings = []
    risk_flags = []
    primary_limiter = None

    # -----------------------------
    # 1. Sleep Quantity & Debt
    # -----------------------------
    sleep_hours = sleep.get("total_sleep_hours")
    sleep_debt = sleep.get("sleep_debt")

    if sleep_hours is not None:
        if sleep_hours < 6:
            findings.append("Severely reduced sleep duration detected.")
            risk_flags.append("low_sleep_duration")
        elif sleep_hours < 7:
            findings.append("Sleep duration below optimal range.")
            risk_flags.append("suboptimal_sleep")

    if sleep_debt is not None and sleep_debt < -1:
        findings.append("Accumulated sleep debt relative to baseline.")
        risk_flags.append("sleep_debt")

    # -----------------------------
    # 2. Sleep Quality & Regularity
    # -----------------------------
    sleep_score = sleep.get("avg_sleep_score")
    sleep_regularity = sleep.get("sleep_regularity")

    if sleep_score is not None and sleep_score < 65:
        findings.append("Reduced sleep quality score observed.")
        risk_flags.append("poor_sleep_quality")

    if sleep_regularity is not None and sleep_regularity < 70:
        findings.append("Irregular sleep pattern detected.")
        risk_flags.append("irregular_sleep")

    # -----------------------------
    # 3. HRV Deviation (Key signal)
    # -----------------------------
    hrv_dev = recovery.get("hrv_deviation")

    if hrv_dev is not None:
        if hrv_dev < -10:
            findings.append("Significant suppression in HRV detected.")
            risk_flags.append("hrv_suppression")
        elif hrv_dev < -5:
            findings.append("Moderate HRV suppression observed.")
            risk_flags.append("mild_hrv_suppression")

    # -----------------------------
    # 4. Resting Heart Rate Elevation
    # -----------------------------
    rhr_dev = recovery.get("rhr_deviation")

    if rhr_dev is not None and rhr_dev > 5:
        findings.append("Elevated resting heart rate relative to baseline.")
        risk_flags.append("elevated_rhr")

    # -----------------------------
    # 5. Late Training Context
    # -----------------------------
    if context.get("late_training_day"):
        findings.append("Evening training may have impaired recovery.")
        risk_flags.append("late_training")

    # -----------------------------
    # 6. Primary Recovery Limiter
    # -----------------------------
    if "hrv_suppression" in risk_flags:
        primary_limiter = "autonomic_stress"
    elif "sleep_debt" in risk_flags or "low_sleep_duration" in risk_flags:
        primary_limiter = "sleep_deprivation"
    elif "elevated_rhr" in risk_flags:
        primary_limiter = "systemic_stress"
    else:
        primary_limiter = "none_detected"

    # -----------------------------
    # 7. Recovery Status Classification
    # -----------------------------
    if len(risk_flags) >= 4:
        recovery_status = "poor"
    elif len(risk_flags) >= 2:
        recovery_status = "compromised"
    else:
        recovery_status = "adequate"

    # -----------------------------
    # 8. Confidence Estimation
    # -----------------------------
    signal_count = sum(
        1 for v in [
            sleep_hours,
            sleep_debt,
            sleep_score,
            hrv_dev,
            rhr_dev
        ] if v is not None
    )

    confidence = min(0.9, 0.4 + 0.12 * signal_count)

    # -----------------------------
    # 9. Structured Output
    # -----------------------------
    return {
        "recovery_status": recovery_status,
        "primary_limiter": primary_limiter,
        "findings": findings,
        "risk_flags": risk_flags,
        "signals_used": signal_count,
        "confidence": round(confidence, 2)
    }
