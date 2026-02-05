from typing import Dict, Any
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage

MONTHLY_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a performance analysis assistant generating MONTHLY athlete insights.\n\n"
        "You DO NOT generate daily or weekly advice.\n"
        "You DO NOT diagnose injury or illness.\n"
        "You DO NOT speculate beyond the provided data.\n\n"
        "Your role is to:\n"
        "- Interpret long-term training and recovery patterns\n"
        "- Comment on consistency, sustainability, and cumulative risk\n"
        "- Provide high-level strategic guidance for the coming month\n\n"
        "Use phrases like:\n"
        "'Over the past month', 'Across the last several weeks', "
        "'The longer-term pattern suggests'.\n\n"
        "Avoid short-term language such as 'today', 'this session', or 'this workout'.\n\n"
        "If confidence is moderate or low, explicitly state uncertainty.\n"
        "All output must be conservative, explainable, and non-medical."
    )
)

# LLM configuration
llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.45
)

def monthly_insight_agent(monthly_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a monthly-level insight from aggregated risk and confidence data.
    """

    human_prompt = HumanMessage(
        content=(
            "Using ONLY the following monthly summary data, generate:\n\n"
            "1. A MONTHLY performance summary (5–7 sentences)\n"
            "2. Key long-term patterns observed (bullet points)\n"
            "3. One strategic focus area for the upcoming month\n\n"
            "Monthly summary data:\n"
            f"{monthly_context}"
        )
    )

    response = llm([
        MONTHLY_SYSTEM_PROMPT,
        human_prompt
    ])

    return {
        "monthly_summary_text": response.content,
        "overall_risk_level": max(
            monthly_context["risk_distribution"],
            key=monthly_context["risk_distribution"].get
        ),
        "confidence": monthly_context.get("average_confidence", 0.6),
        "disclaimer": (
            "This monthly insight is informational and non-medical, "
            "based on long-term aggregated patterns rather than daily observations."
        )
    }