from typing import Dict, Any
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage

WEEKLY_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a performance analysis assistant generating WEEKLY athlete insights.\n\n"
        "You DO NOT generate daily advice.\n"
        "You DO NOT diagnose injury or illness.\n"
        "You DO NOT speculate beyond the provided data.\n\n"
        "Your role is to:\n"
        "- Interpret trends across multiple days\n"
        "- Discuss consistency, variability, and patterns\n"
        "- Provide strategic, high-level guidance\n\n"
        "Use phrases like:\n"
        "'Over the past week', 'Across the last several days', 'The overall pattern suggests'.\n\n"
        "Avoid words like 'today', 'this session', or 'this workout'.\n\n"
        "If confidence is moderate or low, explicitly state uncertainty.\n"
        "All output must be conservative, explainable, and non-medical."
    )
)

# LLM configuration
llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.5
)

def weekly_insight_agent(weekly_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a weekly-level insight from aggregated risk and confidence data.
    """

    human_prompt = HumanMessage(
        content=(
            "Using ONLY the following weekly summary data, generate:\n\n"
            "1. A WEEKLY performance summary (4–6 sentences)\n"
            "2. Key observed patterns (bullet points)\n"
            "3. One strategic coaching takeaway\n\n"
            "Weekly summary data:\n"
            f"{weekly_context}"
        )
    )

    response = llm([
        WEEKLY_SYSTEM_PROMPT,
        human_prompt
    ])

    return {
        "weekly_summary_text": response.content,
        "overall_risk_level": max(
            weekly_context["risk_distribution"],
            key=weekly_context["risk_distribution"].get
        ),
        "confidence": weekly_context.get("average_confidence", 0.6),
        "disclaimer": (
            "This weekly insight is informational and non-medical, "
            "based on aggregated patterns rather than individual days."
        )
    }
