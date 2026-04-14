from __future__ import annotations

from typing import Dict, Any

from simulation.state import SimulationState
from simulation.connectors import (
    fetch_daily_insight,
    fetch_weekly_summary,
    fetch_monthly_summary,
)
from data_access import load_athlete_day

RISK_SCORES = {
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}


def _risk_score(dist: Dict[str, int]) -> float:
    total = sum(dist.values())
    if total == 0:
        return 0.0
    return sum(RISK_SCORES.get(k, 2) * v for k, v in dist.items()) / total


def _trend_label(delta: float, threshold: float = 0.15) -> str:
    if delta > threshold:
        return "increasing"
    if delta < -threshold:
        return "decreasing"
    return "stable"


def build_simulation_state(user_id: str, day: str) -> SimulationState:
    features = load_athlete_day(user_id, day)
    daily_insight = fetch_daily_insight(user_id, day)

    weekly_summary = fetch_weekly_summary(user_id, day)
    monthly_summary = fetch_monthly_summary(user_id, day)

    weekly_risk_score = _risk_score(weekly_summary.get("risk_distribution", {}))
    monthly_risk_score = _risk_score(monthly_summary.get("risk_distribution", {}))

    risk_trend = _trend_label(weekly_risk_score - monthly_risk_score)
    confidence_trend = _trend_label(
        weekly_summary.get("average_confidence", 0) -
        monthly_summary.get("average_confidence", 0)
    )

    training_hours = float(features.get("training_hours", 0.0))
    sleep_hours = float(
        features.get("total_sleep_hours")
        or features.get("sleep_hours_prev_day", 0.0)
    )

    hrv_deviation = float(features.get("hrv_deviation", 0.0))
    rhr_deviation = float(features.get("rhr_deviation", 0.0))

    fatigue_level = max(
        0.0,
        min(10.0, (training_hours * 1.2) + abs(hrv_deviation) + max(0.0, 7 - sleep_hours)),
    )

    if hrv_deviation <= -8 or sleep_hours < 6:
        recovery_status = "poor"
    elif hrv_deviation <= -4 or sleep_hours < 7:
        recovery_status = "strained"
    elif hrv_deviation <= -1:
        recovery_status = "moderate"
    else:
        recovery_status = "good"

    return {
        "user_id": user_id,
        "day": day,
        "training_hours": training_hours,
        "sleep_hours": sleep_hours,
        "fatigue_level": round(fatigue_level, 2),
        "recovery_status": recovery_status,
        "injury_risk": daily_insight.get("risk_level") or features.get("risk_level", "unknown"),
        "confidence": float(daily_insight.get("confidence") or features.get("confidence", 0.0)),
        "hrv_deviation": hrv_deviation,
        "rhr_deviation": rhr_deviation,
        "hard_day": bool(features.get("hard_day", False)),
        "multi_session_day": bool(features.get("multi_session_day", False)),
        "late_training_day": bool(features.get("late_training_day", False)),
        "risk_trend": risk_trend,
        "confidence_trend": confidence_trend,
    }