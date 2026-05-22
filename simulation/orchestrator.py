from __future__ import annotations

from langgraph import graph
from langgraph.graph import StateGraph, END

from simulation.state import SimulationGraphState, SimulationStepResult
from simulation.builders import build_simulation_state
from agents.feature_selector import feature_selector_agent
from agents.training_load import training_load_agent
from agents.recovery import recovery_agent
from agents.injury_risk import injury_risk_agent
from orchestrator import graph_rag_node
from orchestrator import insight_node
from simulation.agents.athlete_system import athlete_dialogue_agent
from simulation.simulation_engine import SimulationEngine

# ----------------------------------------
# Initialize Simulation Engine
# ----------------------------------------
sim_engine = SimulationEngine()


# ----------------------------------------
# Simulation Node (NEW CORE)
# ----------------------------------------
def simulation_node(state: SimulationGraphState):
    base_state = state["simulation_state"]

    # Simulate forward (7 days)
    simulated_states = sim_engine.rollout(base_state, days=7)

    # Use last simulated day
    return {
        "simulation_state": simulated_states[-1]
    }


# ----------------------------------------
# Agent Nodes (UNCHANGED LOGIC)
# ----------------------------------------
def feature_selection_node(state: SimulationGraphState):
    output = feature_selector_agent(
        features=state["simulation_state"]
    )
    print("FEATURE SELECTION OUTPUT:", output)
    return {
        "selected_features": {
            "selected_features": output["selected_features"]  # 🔥 wrap here
        }
    }

def training_load_node(state: SimulationGraphState):
    if "selected_features" not in state:
        raise ValueError("selected_features missing before training_load_node")

    features = state["selected_features"]["selected_features"]

    result = training_load_agent(features)

    print("STATE KEYS:", state.keys())
    print("SELECTED FEATURES:", state.get("selected_features"))

    return {"training_load_analysis": result}

def dialogue_node(state):
    try:
        dialogue = athlete_dialogue_agent(
            state=state["simulation_state"],
            analysis={
                "training_load": state.get("training_load_analysis", {}),
                "recovery": state.get("recovery_analysis", {}),
                "risk": state.get("injury_risk_analysis", {}),
            },
            graph_context=state.get("graph_context", ""),
            history=state.get("history", [])
        )

        if not isinstance(dialogue, dict):
            raise ValueError("Invalid dialogue output")

    except Exception as e:
        print("DIALOGUE NODE ERROR:", e)

        dialogue = {
            "conversation": [],
            "interruption_meta": {}
        }

    return {
        "agent_dialogue": dialogue
    }


def recovery_node(state: SimulationGraphState):
    if "selected_features" not in state:
        raise ValueError("selected_features missing before recovery_node")

    features = state["selected_features"]["selected_features"]

    result = recovery_agent(features)

    return {"recovery_analysis": result}


def risk_node(state: SimulationGraphState):
    result = injury_risk_agent(
        state.get("training_load_analysis"),
        state.get("recovery_analysis")
    )
    return {"injury_risk_analysis": result}


# ----------------------------------------
# Graph Builder
# ----------------------------------------
def build_simulation_graph():
    graph = StateGraph(SimulationGraphState)

    # Nodes
    graph.add_node("simulation", simulation_node)
    graph.add_node("feature_select", feature_selection_node)
    graph.add_node("training_load", training_load_node)
    graph.add_node("recovery", recovery_node)
    graph.add_node("risk", risk_node)
    graph.add_node("graph_rag", graph_rag_node)
    graph.add_node("insight", insight_node)
    graph.add_node("dialogue", dialogue_node)

    graph.set_entry_point("simulation")

    graph.add_edge("simulation", "feature_select")

    # 🔥 FORCE ORDER
    graph.add_edge("feature_select", "training_load")
    graph.add_edge("training_load", "recovery")

    # then continue
    graph.add_edge("recovery", "risk")
    graph.add_edge("risk", "graph_rag")
    graph.add_edge("graph_rag", "insight")
    graph.add_edge("insight", "dialogue")   # ✅ ADD THIS
    graph.add_edge("dialogue", END)
    return graph.compile()

# ----------------------------------------
# Public API
# ----------------------------------------
# def run_simulation_step(user_id: str, day: str) -> SimulationStepResult:
#     simulation_state = build_simulation_state(user_id, day)

#     graph_state: SimulationGraphState = {
#         "simulation_state": simulation_state
#     }

#     graph = build_simulation_graph()
#     result = graph.invoke(graph_state)

#     return {
#         "updated_state": result.get("simulation_state"),
#         "training_load": result.get("training_load_analysis"),
#         "recovery": result.get("recovery_analysis"),
#         "risk": result.get("injury_risk_analysis"),
#         "insight": result.get("insight_report"),
#         "dialogue": result.get("agent_dialogue") or {
#             "athlete": "",
#             "coach": "",
#             "doctor": ""
#         },
#     }

def run_simulation_step(user_id: str, day: str):

    simulation_state = build_simulation_state(user_id, day)

    history = simulation_state.get("history", [])

    graph_state: SimulationGraphState = {
        "simulation_state": simulation_state,
        "history": history[-3:]
    }

    graph = build_simulation_graph()

    result = graph.invoke(graph_state)

    print("GRAPH RESULT:", result)   # 🔥 DEBUG

    if result is None:
        raise ValueError("Graph returned None")

    return {
        "updated_state": result.get("simulation_state"),
        "training_load": result.get("training_load_analysis"),
        "recovery": result.get("recovery_analysis"),
        "risk": result.get("injury_risk_analysis"),
        "insight": result.get("insight_report"),

        # 🔥 IMPORTANT: match your new structure
        "dialogue": result.get("agent_dialogue", {})
    }