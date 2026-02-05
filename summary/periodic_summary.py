import pandas as pd
from typing import Dict, Any
from agents.insight import insight_agent

def generate_periodic_summary(
    df: pd.DataFrame,
    user_id: str,
    end_day: str,
    window: int
) -> Dict[str, Any]:
    """
    Generates a weekly or monthly summary using existing agent outputs.
    """

    end_day = pd.to_datetime(end_day)
    start_day = end_day - pd.Timedelta(days=window - 1)

    window_df = df[
        (df["user_id"] == user_id) &
        (df["day"] >= start_day) &
        (df["day"] <= end_day)
    ].sort_values("day")

    if window_df.empty:
        raise ValueError("No data available for summary window")

    # -----------------------------
    # Aggregate risk & confidence
    # -----------------------------
    risk_counts = window_df["risk_level"].value_counts().to_dict()
    avg_confidence = round(window_df["confidence"].mean(), 2)

    # -----------------------------
    # Load vs recovery balance
    # -----------------------------
    load_mean = window_df["training_hours"].mean()
    recovery_mean = window_df["hrv_deviation"].mean()

    load_recovery_balance = (
        "load_dominant" if load_mean > abs(recovery_mean)
        else "recovery_dominant"
    )

    # -----------------------------
    # Build structured context
    # -----------------------------
    structured_context = {
        "window_days": window,
        "risk_distribution": risk_counts,
        "average_confidence": avg_confidence,
        "average_training_hours": round(load_mean, 2),
        "average_hrv_deviation": round(recovery_mean, 2),
        "load_recovery_balance": load_recovery_balance,
    }

    # -----------------------------
    # Generate narrative summary
    # -----------------------------
    summary_text = insight_agent(
        load={
            "load_status": load_recovery_balance,
            "risk_flags": [],
        },
        recovery={
            "recovery_status": "variable",
            "risk_flags": [],
        },
        risk={
            "risk_level": max(risk_counts, key=risk_counts.get),
            "confidence": avg_confidence,
        },
        knowledge=None
    )

    return {
        "summary_window": f"{window}-day",
        "from": start_day.date().isoformat(),
        "to": end_day.date().isoformat(),
        "risk_distribution": risk_counts,
        "average_confidence": avg_confidence,
        "load_recovery_balance": load_recovery_balance,
        "summary_text": summary_text["insight_text"],
    }
