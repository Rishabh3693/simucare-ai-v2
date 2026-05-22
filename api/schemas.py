from pydantic import BaseModel
from typing import Optional, Dict, Any, List

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


class SimulationStateResponse(BaseModel):
    user_id: str
    day: str
    training_hours: float
    sleep_hours: float
    fatigue_level: float
    recovery_status: str
    injury_risk: str
    confidence: float
    hrv_deviation: float
    rhr_deviation: float
    hard_day: bool
    multi_session_day: bool
    late_training_day: bool
    risk_trend: str
    confidence_trend: str


class AgentActionResponse(BaseModel):
    agent_system: str
    sub_agent: str
    priority: int
    action_type: str
    proposed_changes: Dict[str, Any]
    reasoning: str
    confidence: float


class DialogueTurn(BaseModel):
    role: str
    text: str


class InterruptionMeta(BaseModel):
    interrupt: bool
    who: Optional[str]
    reason: Optional[str]


class Dialogue(BaseModel):
    conversation: List[DialogueTurn]
    interruption_meta: InterruptionMeta


class SimulationResponse(BaseModel):
    updated_state: Dict[str, Any]
    training_load: Dict[str, Any]
    recovery: Dict[str, Any]
    risk: Dict[str, Any]
    insight: Dict[str, Any]
    dialogue: Dialogue 
     