from typing import Dict, Any, List
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
# LLM configuration (swap provider later if needed)
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq", temperature=0.55)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a sports science knowledge assistant.\n"
        "You do NOT change conclusions or risk levels.\n"
        "You do NOT diagnose injury or illness.\n"
        "You explain why the provided patterns matter using "
        "general sports science principles.\n"
        "If confidence is low or data is limited, say so explicitly.\n"
    )
)

def knowledge_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM-enhanced Knowledge & Context Agent.
    Adds evidence-aligned explanations to existing analysis.
    """

    load = context.get("load", {})
    recovery = context.get("recovery", {})
    risk = context.get("risk", {})

    # Build structured input
    structured_context = {
        "training_load_analysis": {
            "load_status": load.get("load_status"),
            "risk_flags": load.get("risk_flags"),
        },
        "recovery_analysis": {
            "recovery_status": recovery.get("recovery_status"),
            "risk_flags": recovery.get("risk_flags"),
            "primary_limiter": recovery.get("primary_limiter"),
        },
        "risk_assessment": {
            "risk_level": risk.get("risk_level"),
            "risk_type": risk.get("risk_type"),
            "confidence": risk.get("confidence"),
        }
    }

    human_prompt = HumanMessage(
        content=(
            "Using ONLY the structured analysis below, provide:\n"
            "1. Explain the sports science context explaining the patterns\n"
            "2. Why these patterns are commonly monitored in athletes\n"
            "3. A short uncertainty note if confidence is limited\n\n"
            "Structured analysis:\n"
            f"{structured_context}"
        )
    )

    response = llm([SYSTEM_PROMPT, human_prompt])

    return {
        "knowledge_summary": response.content,
        "confidence": round(min(0.9, 0.5 + 0.1 * len(risk.get("risk_factors", []))), 2),# rough confidence based on risk factors
        "disclaimer": (
            "Contextual information is based on general sports science principles "
            "and is for educational purposes only."
        )
    }
