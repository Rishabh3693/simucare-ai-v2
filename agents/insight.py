from typing import Dict, Any
from langchain.schema import SystemMessage, HumanMessage
from .graph_rag import GraphRAG

graph_rag = GraphRAG(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="Simucare"
)
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
    knowledge: Dict[str, Any] | None = None,
    graph_context: str | None = None
) -> Dict[str, Any]:

    # ✅ HARD SAFETY: ensure graph context is never empty
    if not graph_context or graph_context.strip() == "":
        graph_context = (
            "hrv_deviation indicates poor recovery; "
            "sleep_debt reduces recovery; "
            "rhr_deviation increases fatigue, which leads to injury risk; "
            "acr_training_load increases injury risk"
        )

    # FINAL PROMPT 
    human_prompt = HumanMessage(
    content=(
        "You are analyzing athlete performance using BOTH data and graph relationships.\n\n"

        "GUIDELINES:\n"
        "1. First describe the athlete’s current state using the data\n"
        "2. Then explain WHY using graph relationships\n"
        "3. Use clear cause-effect statements (increases, reduces, leads to, indicates)\n"
        "4. Combine relationships into logical chains where possible\n"
        "5. Keep explanations concise and non-repetitive\n\n"

        "STRICT RULES:\n"
        "1. Do NOT introduce factors not present in data or graph\n"
        "2. Do NOT reason from missing or absent data\n"
        "3. Do NOT use speculative words (may, likely, possible)\n"
        "4. Do NOT use meta phrases (not mentioned, no evidence, etc.)\n"
        "5. Always use at least one graph relationship to explain risk\n\n"
        "Only use graph relationships for metrics present in the athlete data"
        "If a metric is not present, do NOT mention it"

        "ATHLETE DATA:\n"
        f"Training Load: {load}\n"
        f"Recovery: {recovery}\n"
        f"Risk: {risk}\n\n"

        "GRAPH RELATIONSHIPS:\n"
        f"{graph_context}\n\n"

        "OUTPUT:\n"
        "1. Detailed daily summary (5–6 sentences)\n"
        "   - First 1–2 sentences: describe state from data\n"
        "   - Remaining sentences: explain causes using graph\n"
        "2. Key contributing factors (bullet points)\n"
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
