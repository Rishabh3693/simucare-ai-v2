from typing import TypedDict, Optional, Dict, Any, List

class AthleteState(TypedDict):
    # Metadata
    user_id: str
    day: str

    # Input data
    features: Dict[str, Any]

    # Agent outputs
    selected_features: Optional[Dict[str, Any]]
    training_load_analysis: Optional[Dict[str, Any]]
    recovery_analysis: Optional[Dict[str, Any]]
    injury_risk_analysis: Optional[Dict[str, Any]]
    knowledge_context: Optional[Dict[str, Any]]

    # Final output
    insight_report: Optional[Dict[str, Any]]

    # Safety & diagnostics
    warnings: List[str]
    confidence: float
