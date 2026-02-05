from fastapi import FastAPI, HTTPException
from api.schemas import InsightResponse, ChatRequest, ChatResponse
from data_access import load_athlete_day
from orchestrator import build_orchestrator
from agents.coach_chat import coach_chat_agent
import state
from summary.persist_daily_output import persist_daily_output

app = FastAPI(
    title="SimuCare API v2",
    description="AI-driven athlete insight and coaching system",
    version="2.0.3"
)

graph = build_orchestrator()

@app.get("/user/{user_id}/daily-insight", response_model=InsightResponse)
def get_daily_insight(user_id: str, day: str):
    try:
        features = load_athlete_day(user_id, day)

        state = {
            "user_id": user_id,
            "day": day,
            "features": features,
            "warnings": [],
            "confidence": 0.0
        }

        # Run pipeline ONCE
        result = graph.invoke(state)

        # Persist daily inference
        from summary.persist_daily_output import persist_daily_output
        persist_daily_output(
            user_id=user_id,
            day=day,
            risk_level=result["injury_risk_analysis"]["risk_level"],
            confidence=result["injury_risk_analysis"]["confidence"]
        )

        insight = result["insight_report"]

        return InsightResponse(
            insight_text=insight["insight_text"],
            risk_level=insight["risk_level"],
            confidence=insight["confidence"],
            disclaimer=insight["disclaimer"]
        )


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/user/{user_id}/coach-chat", response_model=ChatResponse)
def coach_chat(user_id: str, day: str, payload: ChatRequest):
    try:
        features = load_athlete_day(user_id, day)

        state = {
            "user_id": user_id,
            "day": day,
            "features": features,
            "warnings": [],
            "confidence": 0.0
        }

        result = graph.invoke(state)

        chat_response = coach_chat_agent(
            question=payload.question,
            context={
                "training_load_analysis": result["training_load_analysis"],
                "recovery_analysis": result["recovery_analysis"],
                "injury_risk_analysis": result["injury_risk_analysis"],
                "knowledge_context": result.get("knowledge_context"),
                "insight_report": result["insight_report"],
                "confidence": result["confidence"]
            }
        )

        return ChatResponse(
            answer=chat_response["answer"],
            confidence=chat_response["confidence"],
            disclaimer=chat_response["disclaimer"]
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

import state
from summary.periodic_summary import generate_periodic_summary
import pandas as pd

DATA_PATH = "data/processed/athlete_day_features.csv"

from agents.weekly_insight import weekly_insight_agent

@app.get("/user/{user_id}/weekly-summary")
def weekly_summary(user_id: str, end_day: str):
    df = pd.read_csv(DATA_PATH, parse_dates=["day"], dayfirst=True)

    # 1. Aggregate weekly context (NO LLM)
    weekly_context = generate_periodic_summary(
        df=df,
        user_id=user_id,
        end_day=end_day,
        window=7
    )

    # 2. Generate weekly narrative (LLM)
    weekly_insight = weekly_insight_agent(weekly_context)

    # 3. Combine and return
    return {
        **weekly_context,
        **weekly_insight
    }

from agents.monthly_insight import monthly_insight_agent

@app.get("/user/{user_id}/monthly-summary")
def monthly_summary(user_id: str, end_day: str):
    df = pd.read_csv(DATA_PATH, parse_dates=["day"], dayfirst=True)

    # 1. Aggregate monthly context (NO LLM here)
    monthly_context = generate_periodic_summary(
        df=df,
        user_id=user_id,
        end_day=end_day,
        window=30
    )

    # 2. Generate monthly narrative using LLM agent
    monthly_insight = monthly_insight_agent(monthly_context)

    # 3. Combine and return
    return {
        **monthly_context,
        **monthly_insight
    }
