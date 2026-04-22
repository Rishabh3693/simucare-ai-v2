import os
from matplotlib import lines
from neo4j import GraphDatabase

class GraphRAG:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    # ---- Map structured features → active metrics
    def extract_active_metrics(self, data):
        active = []

    # ---- LOAD MAPPING ----
        if data.get("load_status") in ["high_load", "overload"]:
            active.append("acr_training_load")
            active.append("acute_training_hours_7d")

        if data.get("load_status") == "recovery_dominant":
            # still include for reasoning
            active.append("chronic_training_hours_28d")

        # ---- RECOVERY MAPPING ----
        if data.get("recovery_status") in ["poor", "variable", "low"]:
            active.append("sleep_debt")
            active.append("hrv_deviation")
            active.append("rhr_deviation")

        # ---- RISK MAPPING ----
        if data.get("risk_level") in ["moderate", "high"]:
            active.append("acr_training_load")

    # ---- BEHAVIOR (OPTIONAL) ----
        if data.get("hard_day"):
            active.append("hard_day")

        if data.get("multi_session_day"):
            active.append("multi_session_day")

        if data.get("late_training_day"):
            active.append("late_training_day")

        return list(set(active))

    # ---- Query graph for relationships (1-hop + 2-hop)
    def query_graph(self, metrics):
        query = """
        MATCH (m:Metric)-[r1]->(c1:Condition)
        WHERE m.name IN $metrics
        OPTIONAL MATCH (c1)-[r2]->(c2:Condition)
        RETURN m.name AS metric,
               type(r1) AS rel1,
               c1.name AS cond1,
               type(r2) AS rel2,
               c2.name AS cond2
        """
        with self.driver.session() as session:
            res = session.run(query, metrics=metrics)
            return [r.data() for r in res]

    # ---- Convert graph results → compact NL context
    def build_context(self, rows):
        if not rows:
            return (
                "sleep_debt reduces recovery; "
                "hrv_deviation indicates poor recovery; "
                "rhr_deviation increases fatigue, which leads to injury risk; "
                "acr_training_load increases injury risk"
            )

        lines = []
        for r in rows:
            if r["cond2"]:
                lines.append(
                    f"{r['metric']} {r['rel1'].lower()} {r['cond1']}, which {r['rel2'].lower()} {r['cond2']}"
                )
            else:
                lines.append(
                    f"{r['metric']} {r['rel1'].lower()} {r['cond1']}"
                )

        return "; ".join(set(lines))