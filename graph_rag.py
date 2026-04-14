import os
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
    def extract_active_metrics(self, data: dict):
        active = []

        # Load
        if data.get("acr_training_load", 0) >= 1.5:
            active.append("acr_training_load")
        if data.get("acute_training_hours_7d", 0) > data.get("chronic_training_hours_28d", 0) * 1.2:
            active.append("acute_training_hours_7d")

        # Recovery
        if data.get("hrv_deviation", 0) < 0:
            active.append("hrv_deviation")
        if data.get("rhr_deviation", 0) > 0:
            active.append("rhr_deviation")

        # Sleep
        if data.get("sleep_debt", 0) > 1:
            active.append("sleep_debt")
        if data.get("total_sleep_hours", 8) < 6.5:
            active.append("total_sleep_hours")

        # Behavior
        if data.get("hard_day", False):
            active.append("hard_day")
        if data.get("multi_session_day", False):
            active.append("multi_session_day")
        if data.get("late_training_day", False):
            active.append("late_training_day")

        # Intensity
        if data.get("total_suffer_score", 0) > 150:
            active.append("total_suffer_score")

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
        # Deduplicate + join
        return "; ".join(sorted(set(lines)))