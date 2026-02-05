from typing import Dict, Any
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
# LLM configuration (swap provider later if needed)
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq", temperature=0.55)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an athlete performance insight assistant.\n"
        "You do NOT diagnose injury or illness.\n\n"
        "You ONLY explain and describe the structured analyses provided.\n"
        "If confidence is low, say so explicitly.\n\n"
        "All output must be evidence-based."
    )
)

def insight_agent(
    load: Dict[str, Any],
    recovery: Dict[str, Any],
    risk: Dict[str, Any],
    knowledge: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    LLM-powered narrative agent.
    Converts structured agent outputs into athlete-friendly insights.
    """

    user_content = {
        "training_load_analysis": load,
        "recovery_analysis": recovery,
        "risk_assessment": risk,
        "knowledge_context": knowledge
    }

    human_prompt = HumanMessage(
        content=(
            "Based ONLY on the following structured analysis, generate:\n"
            "1. A detailed daily summary (5-6 sentences)\n"
            "2. State all Key contributing factors (bullet points)\n"
            "Structured analysis:\n"
            f"{user_content}"
        )
    )

    response = llm([SYSTEM_PROMPT, human_prompt])

    return {
        "insight_text": response.content,
        "confidence": risk.get("confidence", 0.6),
        "risk_level": risk.get("risk_level"),
        "disclaimer": (
            "Insights are informational and not medical advice. "
            "They are based on available data and may be incomplete."
        )
    }
