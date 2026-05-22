import numpy as np

class SimulationEngine:

    def __init__(self):
        pass

    def step(self, prev_state: dict) -> dict:
        """
        Simulate next day metrics from previous state
        """

        next_state = prev_state.copy()

        # ---------------------------
        # Training load dynamics
        # ---------------------------
        load = prev_state.get("training_hours", 0)

        # Random variation (controlled)
        load_noise = np.random.normal(0, 0.5)
        next_state["training_hours"] = max(0, load + load_noise)

        # ---------------------------
        # Fatigue / recovery coupling
        # ---------------------------
        hrv = prev_state.get("hrv_deviation", 0)
        rhr = prev_state.get("rhr_deviation", 0)
        sleep = prev_state.get("sleep_debt", 0)

        # simple dynamics
        next_state["hrv_deviation"] = hrv - 0.2 * load + np.random.normal(0, 0.3)
        next_state["rhr_deviation"] = rhr + 0.1 * load + np.random.normal(0, 0.2)
        next_state["sleep_debt"] = sleep + np.random.normal(0, 0.2)

        return next_state

    def rollout(self, initial_state: dict, days: int):
        states = [initial_state]

        current = initial_state
        for _ in range(days):
            current = self.step(current)
            states.append(current)

        return states