from typing import TypedDict, Dict, Any, List


class SimulationState(TypedDict):
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


class AgentAction(TypedDict):
    agent_system: str
    sub_agent: str
    priority: int
    action_type: str
    proposed_changes: Dict[str, Any]
    reasoning: str
    confidence: float


class AgentMessage(TypedDict):
    round_index: int
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]


class DoctorConstraint(TypedDict):
    max_training_hours: float | None
    mandatory_rest: bool
    rationale: str


class SimulationStepResult(TypedDict):
    updated_state: SimulationState
    agent_actions: List[AgentAction]
    explanation: str


class SimulationGraphState(TypedDict):
    simulation_state: SimulationState
    agent_actions: List[AgentAction]
    selected_action: AgentAction | None
    explanation: str
    message_bus: List[AgentMessage]
    doctor_constraints: DoctorConstraint
    negotiation_round: int