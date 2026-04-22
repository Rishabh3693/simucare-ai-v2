from .graph_rag import GraphRAG

graph_rag = GraphRAG(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="Simucare"
)
from typing import Dict, Any
from langchain.schema import HumanMessage



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

    # Build structured data dynamically
    structured_data = {
        "load_status": weekly_context.get("load_recovery_balance"),
        "risk_level": max(
            weekly_context["risk_distribution"],
            key=weekly_context["risk_distribution"].get
        ),
        "recovery_status": "variable"  # or improve later
    }

    # GraphRAG pipeline
    active_metrics = graph_rag.extract_active_metrics(structured_data)
    graph_results = graph_rag.query_graph(active_metrics)
    graph_context = graph_rag.build_context(graph_results)

    print("WEEKLY GRAPH CONTEXT:", graph_context)  # debug

    human_prompt = HumanMessage(
        content=(
            "You MUST follow these strict rules:\n"
            "1. Graph relationships ARE provided below and MUST be used\n"
            "2. You are NOT allowed to say graph data is missing\n"
            "3. You MUST explain cause-effect ONLY using graph relationships\n"
            "4. You are NOT allowed to use general knowledge\n\n"

            "If you violate any rule, the answer is incorrect.\n\n"
            "Weekly summary data:\n"
            f"{weekly_context}\n\n"
            
            "Instructions:\n"
            "- Explain cause-effect using graph relationships\n"
            "- Use words like: increases, reduces, leads to, indicates\n"
            "- Do NOT mention missing graph data\n\n"

            "Graph Relationships:\n"
            f"{graph_context}\n\n"
            "1. A WEEKLY performance summary (5-7 sentences)\n"
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
        "overall_risk_level": structured_data["risk_level"],
        "confidence": weekly_context.get("average_confidence", 0.6),
        "disclaimer": (
            "This weekly insight is informational and non-medical, "
            "based on aggregated patterns rather than individual days."
        )
    }
