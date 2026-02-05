from typing import Dict, Any
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
# LLM configuration (swap provider later if needed)
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq", temperature=0.7)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a virtual performance coach.\n"
        "You do NOT analyze raw data.\n"
        "You do NOT change risk levels.\n"
        "You do NOT diagnose injury or illness.\n"
        "You do NOT give medical advice.\n\n"
        "You ONLY answer questions using the provided analyses.\n"
        "If confidence is low or data is missing, say so explicitly.\n\n"
        "Tone: supportive, clear, conservative."
    )
)

def coach_chat_agent(
    question: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Conversational coach agent.
    Answers athlete questions using existing system outputs.
    """

    human_prompt = HumanMessage(
        content=(
            f"Athlete question:\n{question}\n\n"
            "Use ONLY the following system outputs to answer:\n"
            f"{context}\n\n"
            "Answer clearly and conservatively."
        )
    )

    response = llm([SYSTEM_PROMPT, human_prompt])

    return {
        "answer": response.content,
        "confidence": context.get("confidence", 0.6),
        "disclaimer": (
            "This response is informational and not medical advice. "
            "It is based on available data and may be incomplete."
        )
    }
