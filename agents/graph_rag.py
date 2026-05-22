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
        active = set()

        # ---- ALWAYS INCLUDE CORE PHYSIOLOGY ----
        active.update([
            "acr_training_load",
            "hrv_deviation",
            "sleep_debt",
            "rhr_deviation",
            "chronic_training_hours_28d"
        ])

        # ---- LOAD SIGNALS ----
        if data.get("load_status") in ["high_load", "overload"]:
            active.add("acute_training_hours_7d")

        # ---- RECOVERY SIGNALS ----
        if data.get("recovery_status") in ["poor", "variable", "low"]:
            active.update([
                "sleep_debt",
                "hrv_deviation",
                "rhr_deviation"
            ])

        # ---- RISK SIGNALS ----
        if data.get("risk_level") in ["moderate", "high"]:
            active.add("acr_training_load")

        return list(active)

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
        valid_rows = [
            r for r in rows if r.get("rel1") and r.get("cond1")
        ]

        if not valid_rows:
            return (
                "Sleep debt reduces recovery; "
                "HRV deviation indicates poor recovery; "
                "RHR deviation increases fatigue, which leads to injury risk; "
                "ACR training load increases injury risk"
            )

        lines = []
        for r in valid_rows:
            metric = r["metric"].replace("_", " ")
            cond1 = r["cond1"]

            if r["cond2"]:
                cond2 = r["cond2"]
                lines.append(
                    f"{metric} {r['rel1'].lower()} {cond1}, which {r['rel2'].lower()} {cond2}"
                )
            else:
                lines.append(
                    f"{metric} {r['rel1'].lower()} {cond1}"
                )

        return "; ".join(set(lines))