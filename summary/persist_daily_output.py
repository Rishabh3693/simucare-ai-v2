import pandas as pd
from datetime import datetime

DATA_PATH = "data/processed/athlete_day_features.csv"
INFERENCE_VERSION = "v1.0"   # update when logic changes


def persist_daily_output(
    user_id: str,
    day: str,
    risk_level: str,
    confidence: float
) -> None:
    """
    Safely persists daily agent outputs into the processed dataset.

    Rules:
    - One row per (user_id, day)
    - Overwrites existing inference for the same day
    - Adds inference metadata
    """

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["day"],
        dayfirst=True
    )

    # Normalize dates
    df["day"] = df["day"].dt.normalize()
    day = pd.to_datetime(day, dayfirst=True).normalize()

    # -----------------------------
    # Safety check: row must exist
    # -----------------------------
    mask = (df["user_id"] == user_id) & (df["day"] == day)

    if not mask.any():
        raise ValueError(
            f"No base feature row found for user_id={user_id}, day={day.date()}. "
            "Daily inference can only be persisted after feature generation."
        )

    # -----------------------------
    # Ensure inference columns exist
    # -----------------------------
    for col in ["risk_level", "confidence", "inferred_at", "inference_version"]:
        if col not in df.columns:
            df[col] = None

    # -----------------------------
    # Overwrite-safe update
    # -----------------------------
    df.loc[mask, "risk_level"] = risk_level
    df.loc[mask, "confidence"] = round(float(confidence), 2)
    df.loc[mask, "inferred_at"] = datetime.utcnow().isoformat()
    df.loc[mask, "inference_version"] = INFERENCE_VERSION

    # -----------------------------
    # Save back to disk
    # -----------------------------
    df.to_csv(DATA_PATH, index=False)
