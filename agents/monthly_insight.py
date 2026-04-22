from .graph_rag import GraphRAG

graph_rag = GraphRAG(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="Simucare"
)


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

    # Build structured data for GraphRAG
    structured_data = {
        "load_status": monthly_context.get("load_recovery_balance"),
        "risk_level": max(
            monthly_context["risk_distribution"],
            key=monthly_context["risk_distribution"].get
        ),
        "recovery_status": "variable"  # can improve later
    }

    # GraphRAG pipeline
    active_metrics = graph_rag.extract_active_metrics(structured_data)
    graph_results = graph_rag.query_graph(active_metrics)
    graph_context = graph_rag.build_context(graph_results)
    print("ACTIVE METRICS:", active_metrics)
    print("GRAPH RESULTS:", graph_results)
    print("GRAPH CONTEXT:", graph_context)

    print("MONTHLY GRAPH CONTEXT:", graph_context)  # debug

    human_prompt = HumanMessage(
        content=(
            "You MUST follow these strict rules:\n"
            "1. Graph relationships ARE provided below and MUST be used\n"
            "2. You are NOT allowed to say graph data is missing\n"
            "3. You MUST explain cause-effect ONLY using graph relationships\n"
            "4. You are NOT allowed to use general knowledge\n\n"

            "If you violate any rule, the answer is incorrect.\n\n"
            "Monthly summary data:\n"
            f"{monthly_context}\n\n"

            "Instructions:\n"
            "- Explain cause-effect using graph relationships\n"
            "- Use words like: increases, reduces, leads to, indicates\n"
            "- Do NOT mention missing graph data\n\n"

            "Graph Relationships:\n"
            f"{graph_context}\n\n"
            "Generate:\n"
            "1. A MONTHLY performance summary (5–7 sentences)\n"
            "2. Key long-term patterns observed (bullet points)\n"
            "3. One strategic focus area for the upcoming month\n"
            "Monthly summary data:\n"
        )
    )

    response = llm([
        MONTHLY_SYSTEM_PROMPT,
        human_prompt
    ])

    return {
        "monthly_summary_text": response.content,
        "overall_risk_level": structured_data["risk_level"],
        "confidence": monthly_context.get("average_confidence", 0.6),
        "disclaimer": (
            "This monthly insight is informational and non-medical, "
            "based on long-term aggregated patterns rather than daily observations."
        )
    }