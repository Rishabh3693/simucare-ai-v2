from pydantic import BaseModel
from typing import Optional, Dict, Any

class InsightResponse(BaseModel):
    insight_text: str
    risk_level: str
    confidence: float
    disclaimer: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    disclaimer: str
