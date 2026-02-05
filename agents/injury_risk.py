from typing import Dict, Any

def injury_risk_agent(
    load: Dict[str, Any],
    recovery: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assesses injury and illness risk by combining training load and recovery state.
    This agent produces risk levels, not diagnoses.
    """

    risk_factors = []
    contributors = []

    load_status = load.get("load_status")
    fatigue_risk = load.get("fatigue_risk")
    load_flags = load.get("risk_flags", [])

    recovery_status = recovery.get("recovery_status")
    recovery_flags = recovery.get("risk_flags", [])

    # -----------------------------
    # 1. Load-related risk factors
    # -----------------------------
    if load_status == "high":
        risk_factors.append("High external training load")
        contributors.append("training_load")

    if "high_acr" in load_flags:
        risk_factors.append("Acute-to-chronic load spike detected")
        contributors.append("load_spike")

    if "multi_session_day" in load_flags:
        risk_factors.append("Multiple sessions in a single day")
        contributors.append("session_density")

    # -----------------------------
    # 2. Recovery-related risk factors
    # -----------------------------
    if recovery_status in ["poor", "compromised"]:
        risk_factors.append("Insufficient physiological recovery")
        contributors.append("recovery_deficit")

    if "hrv_suppression" in recovery_flags:
        risk_factors.append("Suppressed autonomic recovery (HRV)")
        contributors.append("autonomic_stress")

    if "elevated_rhr" in recovery_flags:
        risk_factors.append("Elevated resting heart rate")
        contributors.append("systemic_stress")

    if "sleep_debt" in recovery_flags:
        risk_factors.append("Accumulated sleep debt")
        contributors.append("sleep_deprivation")

    # -----------------------------
    # 3. Compounding risk logic
    # -----------------------------
    compounding = False

    if load_status == "high" and recovery_status in ["poor", "compromised"]:
        compounding = True
        risk_factors.append(
            "High training load combined with inadequate recovery"
        )
        contributors.append("load_recovery_mismatch")

    # -----------------------------
    # 4. Risk classification
    # -----------------------------
    unique_contributors = set(contributors)

    if compounding and len(unique_contributors) >= 3:
        risk_level = "high"
    elif len(unique_contributors) >= 2:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # -----------------------------
    # 5. Risk type inference
    # -----------------------------
    if "autonomic_stress" in contributors and "systemic_stress" in contributors:
        risk_type = "illness_risk"
    elif "load_recovery_mismatch" in contributors:
        risk_type = "injury_risk"
    else:
        risk_type = "general_fatigue"

    # -----------------------------
    # 6. Confidence estimation
    # -----------------------------
    signal_count = (
        len(load_flags) +
        len(recovery_flags) +
        (1 if compounding else 0)
    )

    confidence = min(0.9, 0.45 + 0.1 * signal_count)

    # -----------------------------
    # 7. Structured output
    # -----------------------------
    return {
        "risk_level": risk_level,
        "risk_type": risk_type,
        "risk_factors": risk_factors,
        "contributors": list(unique_contributors),
        "confidence": round(confidence, 2),
        "disclaimer": (
            "This assessment indicates elevated risk patterns "
            "and does not constitute a medical diagnosis."
        )
    }
