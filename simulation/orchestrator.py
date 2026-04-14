from __future__ import annotations

from langgraph.graph import StateGraph, END

from simulation.state import (
    SimulationGraphState,
    SimulationStepResult,
    AgentAction,
    AgentMessage,
)
from simulation.builders import build_simulation_state
from simulation.agents.athlete_system import athlete_agent_system
from simulation.agents.trainer_system import trainer_agent_system
from simulation.agents.doctor_system import doctor_agent_system

MAX_NEGOTIATION_ROUNDS = 3


def _message(round_index: int, from_agent: str, to_agent: str, payload: dict) -> AgentMessage:
    return {
        "round_index": round_index,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "message_type": "proposal",
        "payload": payload,
    }


def _latest_action_by_agent(actions: list[AgentAction], agent_system: str) -> AgentAction | None:
    for action in reversed(actions):
        if action["agent_system"] == agent_system:
            return action
    return None


def athlete_node(state: SimulationGraphState) -> dict:
    action = athlete_agent_system(
    state["simulation_state"],
    doctor_constraints=state["doctor_constraints"]
)
    round_index = state["negotiation_round"]

    message = _message(round_index, "athlete", "trainer", action["proposed_changes"])
    return {
        "agent_actions": state["agent_actions"] + [action],
        "message_bus": state["message_bus"] + [message],
    }


def trainer_node(state: SimulationGraphState) -> dict:
    action = trainer_agent_system(
        state["simulation_state"],
        message_bus=state["message_bus"],
    )
    round_index = state["negotiation_round"]
    message = _message(round_index, "trainer", "doctor", action["proposed_changes"])

    return {
        "agent_actions": state["agent_actions"] + [action],
        "message_bus": state["message_bus"] + [message],
    }


def doctor_node(state: SimulationGraphState) -> dict:
    action = doctor_agent_system(
        state["simulation_state"],
        message_bus=state["message_bus"],
    )
    round_index = state["negotiation_round"]
    message = _message(round_index, "doctor", "all", action["proposed_changes"])

    return {
        "agent_actions": state["agent_actions"] + [action],
        "message_bus": state["message_bus"] + [message],
    }


def round_control_node(state: SimulationGraphState) -> dict:
    return {"negotiation_round": state["negotiation_round"] + 1}


def should_continue_negotiation(state: SimulationGraphState) -> str:
    if state["negotiation_round"] < MAX_NEGOTIATION_ROUNDS:
        return "athlete"
    return "collect_constraints"


def collect_constraints_node(state: SimulationGraphState) -> dict:
    doctor_action = _latest_action_by_agent(state["agent_actions"], "doctor")

    constraints = {
        "max_training_hours": None,
        "mandatory_rest": False,
        "rationale": "No doctor constraints applied.",
    }

    if doctor_action:
        medical_constraints = doctor_action["proposed_changes"].get("medical_constraints", {})
        constraints = {
            "max_training_hours": medical_constraints.get("max_training_hours"),
            "mandatory_rest": bool(medical_constraints.get("mandatory_rest", False)),
            "rationale": doctor_action["reasoning"],
        }

    return {"doctor_constraints": constraints}


def merge_plan_node(state: SimulationGraphState) -> dict:
    athlete_action = _latest_action_by_agent(state["agent_actions"], "athlete")
    trainer_action = _latest_action_by_agent(state["agent_actions"], "trainer")
    doctor_action = _latest_action_by_agent(state["agent_actions"], "doctor")

    merged_training_hours = state["simulation_state"]["training_hours"]

    if athlete_action:
        athlete_hours = athlete_action["proposed_changes"].get("training_hours")
        if isinstance(athlete_hours, (int, float)):
            merged_training_hours = float(athlete_hours)

    if trainer_action:
        trainer_hours = trainer_action["proposed_changes"].get("training_hours")
        if isinstance(trainer_hours, (int, float)):
            merged_training_hours = float(trainer_hours)

    constraints = state["doctor_constraints"]
    if constraints["mandatory_rest"]:
        merged_training_hours = 0.0
    elif isinstance(constraints["max_training_hours"], (int, float)):
        merged_training_hours = min(merged_training_hours, float(constraints["max_training_hours"]))

    selected = doctor_action or trainer_action or athlete_action
    explanation = (
        f"Negotiation completed in {state['negotiation_round']} rounds. "
        "Merged plan with policy order Doctor > Trainer > Athlete and applied doctor constraints."
    )

    return {
        "selected_action": selected,
        "explanation": explanation,
        "simulation_state": {
            **state["simulation_state"],
            "training_hours": round(max(0.0, merged_training_hours), 2),
        },
    }


def update_state_node(state: SimulationGraphState) -> dict:
    simulation_state = state["simulation_state"].copy()
    selected = state.get("selected_action")

    if selected:
        for key, value in selected["proposed_changes"].items():
            if key in simulation_state and value is not None:
                simulation_state[key] = value

    constraints = state["doctor_constraints"]
    if constraints["mandatory_rest"]:
        simulation_state["training_hours"] = 0.0
    elif isinstance(constraints["max_training_hours"], (int, float)):
        simulation_state["training_hours"] = min(
            float(simulation_state["training_hours"]),
            float(constraints["max_training_hours"]),
        )

    return {"simulation_state": simulation_state}


def build_simulation_graph():
    graph = StateGraph(SimulationGraphState)

    graph.add_node("athlete", athlete_node)
    graph.add_node("trainer", trainer_node)
    graph.add_node("doctor", doctor_node)
    graph.add_node("round_control", round_control_node)
    graph.add_node("collect_constraints", collect_constraints_node)
    graph.add_node("merge_plan", merge_plan_node)
    graph.add_node("final_update", update_state_node)

    graph.set_entry_point("athlete")

    graph.add_edge("athlete", "trainer")
    graph.add_edge("trainer", "doctor")
    graph.add_edge("doctor", "round_control")

    graph.add_conditional_edges(
        "round_control",
        should_continue_negotiation,
        {
            "athlete": "athlete",
            "collect_constraints": "collect_constraints",
        },
    )

    graph.add_edge("collect_constraints", "merge_plan")
    graph.add_edge("merge_plan", "final_update")
    graph.add_edge("final_update", END)

    return graph.compile()


def run_simulation_step(user_id: str, day: str) -> SimulationStepResult:
    simulation_state = build_simulation_state(user_id, day)

    graph_state: SimulationGraphState = {
        "simulation_state": simulation_state,
        "agent_actions": [],
        "selected_action": None,
        "explanation": "",
        "message_bus": [],
        "doctor_constraints": {
            "max_training_hours": None,
            "mandatory_rest": False,
            "rationale": "",
        },
        "negotiation_round": 0,
    }

    graph = build_simulation_graph()
    result = graph.invoke(graph_state)

    return {
        "updated_state": result["simulation_state"],
        "agent_actions": result["agent_actions"],
        "explanation": result["explanation"],
    }
