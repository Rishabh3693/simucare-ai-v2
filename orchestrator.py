from langgraph import graph
from langgraph.graph import StateGraph, END
from state import AthleteState

from graph_rag import GraphRAG

graph_rag = GraphRAG(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="Simucare"
)

# Import agent callables (pure functions or LLM chains)
from agents.feature_selector import feature_selector_agent
from agents.training_load import training_load_agent
from agents.recovery import recovery_agent
from agents.injury_risk import injury_risk_agent
from agents.knowledge import knowledge_agent
from agents.insight import insight_agent



def feature_selection_node(state: AthleteState):
    output = feature_selector_agent(
        features=state["features"]
    )
    return {
        "selected_features": output
    }


def training_load_node(state: AthleteState):
    output = training_load_agent(
        features=state["selected_features"]["selected_features"]
    )
    return {
        "training_load_analysis": output
    }

# Continue defining nodes for each agent
def recovery_node(state: AthleteState):
    output = recovery_agent(
        features=state["selected_features"]["selected_features"]
    )
    return {
        "recovery_analysis": output
    }

def injury_risk_node(state: AthleteState):
    output = injury_risk_agent(
        load=state["training_load_analysis"],
        recovery=state["recovery_analysis"]
    )
    return {
        "injury_risk_analysis": output
    }

def knowledge_node(state: AthleteState):
    output = knowledge_agent(
        context={
            "load": state["training_load_analysis"],
            "recovery": state["recovery_analysis"],
            "risk": state["injury_risk_analysis"]
        }
    )
    return {
        "knowledge_context": output
    }

def graph_rag_node(state: AthleteState):
    # Combine all structured outputs
    structured_data = {
        **state.get("training_load_analysis", {}),
        **state.get("recovery_analysis", {}),
        **state.get("injury_risk_analysis", {})
    }

    # Step 1: Extract relevant metrics
    active_metrics = graph_rag.extract_active_metrics(structured_data)

    # Step 2: Query Neo4j
    graph_results = graph_rag.query_graph(active_metrics)

    # Step 3: Build context
    graph_context = graph_rag.build_context(graph_results)

    return {
        "graph_context": graph_context
    }

def insight_node(state: AthleteState):
    output = insight_agent(
        load=state["training_load_analysis"],
        recovery=state["recovery_analysis"],
        risk=state["injury_risk_analysis"],
        knowledge=state.get("knowledge_context"),
        graph_context=state.get("graph_context") 
    )
    return {
        "insight_report": output,
        "confidence": output.get("confidence", 0.6)
    }


def build_orchestrator():
    graph = StateGraph(AthleteState)

    # Register nodes
    graph.add_node("feature_select", feature_selection_node)
    graph.add_node("training_load", training_load_node)
    graph.add_node("recovery", recovery_node)
    graph.add_node("injury_risk", injury_risk_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("graph_rag", graph_rag_node)
    graph.add_node("insight", insight_node)

    # Define flow
    graph.set_entry_point("feature_select")

    graph.add_edge("feature_select", "training_load")
    graph.add_edge("feature_select", "recovery")

    # Parallel resolution → injury risk
    graph.add_edge("training_load", "injury_risk")
    graph.add_edge("recovery", "injury_risk")

    # Optional enrichment
    graph.add_edge("injury_risk", "knowledge")

    # Final synthesis
    graph.add_edge("knowledge", "graph_rag")
    graph.add_edge("graph_rag", "insight")
    graph.add_edge("insight", END)

    return graph.compile()
