from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from data_access import load_athlete_day
from orchestrator import build_orchestrator
from summary.periodic_summary import generate_periodic_summary
from agents.weekly_insight import weekly_insight_agent
from agents.monthly_insight import monthly_insight_agent

DATA_PATH = "data/processed/athlete_day_features.csv"

_graph = build_orchestrator()


def fetch_daily_insight(user_id: str, day: str) -> Dict[str, Any]:
    """
    Connects to the existing daily insight pipeline used by
    /user/{user_id}/daily-insight and returns its outputs.
    """
    features = load_athlete_day(user_id, day)

    state = {
        "user_id": user_id,
        "day": day,
        "features": features,
        "warnings": [],
        "confidence": 0.0,
    }

    result = _graph.invoke(state)
    insight = result["insight_report"]

    return {
        "risk_level": insight.get("risk_level"),
        "confidence": insight.get("confidence"),
        "insight_text": insight.get("insight_text"),
    }


def fetch_weekly_summary(user_id: str, end_day: str) -> Dict[str, Any]:
    """
    Connects to the weekly summary pipeline used by /weekly-summary.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["day"], dayfirst=True)
    weekly_context = generate_periodic_summary(
        df=df,
        user_id=user_id,
        end_day=end_day,
        window=7,
    )
    weekly_insight = weekly_insight_agent(weekly_context)
    return {
        **weekly_context,
        **weekly_insight,
    }


def fetch_monthly_summary(user_id: str, end_day: str) -> Dict[str, Any]:
    """
    Connects to the monthly summary pipeline used by /monthly-summary.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["day"], dayfirst=True)
    monthly_context = generate_periodic_summary(
        df=df,
        user_id=user_id,
        end_day=end_day,
        window=30,
    )
    monthly_insight = monthly_insight_agent(monthly_context)
    return {
        **monthly_context,
        **monthly_insight,
    }