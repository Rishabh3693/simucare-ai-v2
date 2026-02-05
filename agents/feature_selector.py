from typing import Dict, Any

def feature_selector_agent(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selects contextually relevant features for downstream agents.
    Input: full athlete-day feature dictionary
    Output: filtered feature set + rationale
    """

    selected = {}
    rationale = {}

    # -------------------------------
    # 1. Training load signals
    # -------------------------------
    training_keys = [
        "training_hours",
        "acute_training_hours_7d",
        "chronic_training_hours_28d",
        "acr_training_load",
        "hard_sessions",
        "session_count",
        "total_suffer_score",
        "multi_session_day",
    ]

    training_present = {
        k: features.get(k)
        for k in training_keys
        if features.get(k) is not None
    }

    if training_present:
        selected["training_load"] = training_present
        rationale["training_load"] = (
            "Training volume and intensity signals are present and "
            "are required to assess acute and cumulative load."
        )

    # -------------------------------
    # 2. Sleep signals
    # -------------------------------
    sleep_keys = [
        "total_sleep_hours",
        "sleep_baseline_7d",
        "sleep_debt",
        "avg_sleep_score",
        "sleep_regularity",
        "late_training_day",
    ]

    sleep_present = {
        k: features.get(k)
        for k in sleep_keys
        if features.get(k) is not None
    }

    if sleep_present:
        selected["sleep"] = sleep_present
        rationale["sleep"] = (
            "Sleep duration, quality, and regularity signals are "
            "available and relevant for recovery assessment."
        )

    # -------------------------------
    # 3. Recovery signals (HRV / RHR)
    # -------------------------------
    recovery_keys = [
        "hrv_balance",
        "hrv_baseline_14d",
        "hrv_deviation",
        "resting_heart_rate",
        "rhr_deviation",
        "body_temperature",
    ]

    recovery_present = {
        k: features.get(k)
        for k in recovery_keys
        if features.get(k) is not None
    }

    if recovery_present:
        selected["recovery"] = recovery_present
        rationale["recovery"] = (
            "Autonomic and physiological recovery markers "
            "are available and indicate internal stress response."
        )

    # -------------------------------
    # 4. Context / intent signals
    # -------------------------------
    intent_keys = [
        "hard_day",
        "easy_sessions",
        "evening_sessions",
        "strength_sessions",
        "long_sessions",
    ]

    intent_present = {
        k: features.get(k)
        for k in intent_keys
        if features.get(k) is not None
    }

    if intent_present:
        selected["context"] = intent_present
        rationale["context"] = (
            "Session intent and timing features provide context "
            "about training quality and behavioral patterns."
        )

    # -------------------------------
    # 5. Lagged signals (yesterday effects)
    # -------------------------------
    lag_keys = [
        "readiness_prev_day",
        "training_hours_prev_day",
        "sleep_hours_prev_day",
    ]

    lag_present = {
        k: features.get(k)
        for k in lag_keys
        if features.get(k) is not None
    }

    if lag_present:
        selected["lagged"] = lag_present
        rationale["lagged"] = (
            "Lagged features capture carryover effects "
            "from the previous day."
        )

    # -------------------------------
    # 6. Diagnostics & warnings
    # -------------------------------
    warnings = []

    if not selected:
        warnings.append(
            "No valid features available for this date; "
            "downstream analysis confidence should be reduced."
        )

    if features.get("training_hours") is None and features.get("total_sleep_hours") is None:
        warnings.append(
            "Both training and sleep data are missing; "
            "interpretations will rely primarily on recovery signals."
        )

    # -------------------------------
    # 7. Final structured output
    # -------------------------------
    return {
        "selected_features": selected,
        "rationale": rationale,
        "warnings": warnings,
        "feature_count": sum(len(v) for v in selected.values()),
    }
