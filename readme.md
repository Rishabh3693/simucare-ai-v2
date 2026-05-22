<img width="728" height="256" alt="Image" src="https://github.com/user-attachments/assets/3186533c-5fd1-4075-bbf5-d55c1f4728e5" />
# SimuCare — Agentic AI for Personalized Performance Optimization

SimuCare is an athlete monitoring and simulation system that combines data preprocessing, state simulation, multi-agent analysis, graph-based reasoning, insight generation, and an interactive frontend. The current build uses a processed athlete-day table as its base dataset, a FastAPI backend for serving simulation results, LangGraph for orchestration, Groq-hosted LLMs for reasoning and dialogue, and Neo4j for graph-grounded causal relationships.

---

## 1) What this project does

SimuCare turns raw athlete data into structured daily signals and then uses those signals to produce:

- **Daily insights**: current training load, recovery, and injury-risk assessment
- **Weekly summaries**: short-term pattern analysis across recent days
- **Monthly summaries**: longer-term trend analysis across a 30-day window
- **Multi-agent dialogue**: athlete, coach, and doctor conversation based on the current state
- **Interactive UI**: a frontend that visualizes the conversation and metrics

The backend flow is driven by a typed athlete state object and a graph-based orchestrator that passes data through feature selection, load analysis, recovery analysis, risk analysis, graph reasoning, and final insight generation. The state schema is defined in `state.py`, and the orchestrator wires the agent flow together in `orchestrator.py`. The current state fields include `user_id`, `day`, `features`, `selected_features`, `training_load_analysis`, `recovery_analysis`, `injury_risk_analysis`, `knowledge_context`, `graph_context`, `insight_report`, `warnings`, and `confidence`. fileciteturn26file6 fileciteturn26file4

---

## 2) Core files you will use

These are the files that matter for running the current build:

- `api/main.py` — FastAPI entrypoint (your `/simulate-day` endpoint lives here)
- `agents/` — feature selection, load, recovery, risk, knowledge, insight, and graph reasoning modules
- `simulation/` — state simulation / day rollout logic
- `data_access.py` — loads the processed athlete-day CSV
- `new_reader.py` — builds the final processed dataset from raw Oura + Strava files
- `reader.py` — exploratory preprocessing / inspection script
- `state.py` — typed state definition shared by the LangGraph pipeline
- `frontend/` — React UI (if you are using the frontend version of the demo)

The orchestrator currently routes the flow from feature selection to training load, recovery, injury risk, knowledge grounding, graph reasoning, and insight generation. fileciteturn26file4

---

## 3) Requirements

### Backend
- Python 3.10+ recommended
- Neo4j server running locally or remotely
- Groq API key for LLM calls
- Python packages used by the backend:
  - `fastapi`
  - `uvicorn`
  - `pandas`
  - `numpy`
  - `python-dotenv`
  - `langchain`
  - `langgraph`
  - `neo4j`
  - `groq`
  - `pydantic`

### Frontend
- Node.js 18+ recommended
- npm
- React/Vite frontend dependencies
- Tailwind CSS for styling

---

## 4) Environment configuration

The current environment file uses these variables:

- `GROQ_API_KEY`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

The `.gitignore` already excludes `.env`, `venv`, `node_modules`, build outputs, caches, and editor folders, so keep the secret file local and do not commit it. fileciteturn26file0 fileciteturn26file1

### Recommended `.env`
Use this format on the machine where you run the project:

```env
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

If you are on Windows, plain `KEY=value` lines are usually the safest. If your existing `.env` includes `export`, Python-dotenv may still parse it, but plain assignments are simpler and more portable.

---

## 5) Important path changes when moving to another system

This project contains absolute Windows paths in the preprocessing scripts. You **must** update them before running on another machine.

### Update these paths in `new_reader.py`
The current preprocessing script reads raw data from hardcoded locations like:

```python
OURA_PATH = "C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Oura\\"
STRAVA_PATH = "C:\\Users\\messt\\Downloads\\MatsLA\\MatsLA\\Strava\\"
OUTPUT_PATH = "data/processed/athlete_day_features.csv"
```

Replace `OURA_PATH` and `STRAVA_PATH` with the local folders that contain your raw Oura and Strava CSV files. Keep `OUTPUT_PATH` pointed at your repo’s `data/processed/` directory unless you intentionally want a different location. The script writes the final unified athlete-day table there. fileciteturn26file3

### Update these paths in `reader.py` if you use it
`reader.py` also contains hardcoded Windows source folders for Oura and Strava data. If you use that exploratory script on another system, replace those paths too.

### Keep `data_access.py` aligned with the output CSV
`data_access.py` expects the processed table at:

```python
DATA_PATH = "data/processed/athlete_day_features.csv"
```

If you change the output path in preprocessing, update this constant as well. fileciteturn26file2

---

## 6) Data pipeline setup order

### Step 1 — Place raw data on the new machine
You need the raw CSV exports for:
- Oura activity
- Oura heart rate
- Oura readiness
- Oura sleep
- Strava activities

Make sure the folder layout in `new_reader.py` matches your local paths after you update them.

### Step 2 — Build the processed dataset
Run the preprocessing script that produces the final athlete-day table:

```bash
python new_reader.py
```

This script:
- parses Oura and Strava timestamps,
- aggregates sleep by wake-up day,
- merges Oura and Strava into one athlete-day table,
- engineers training, sleep, recovery, and behavior features,
- computes rolling features such as acute load, chronic load, ACR, HRV deviation, RHR deviation, and sleep debt,
- and saves the final dataset to `data/processed/athlete_day_features.csv`. fileciteturn26file3

### Step 3 — Verify the processed file exists
Confirm that this file exists:

```text
data/processed/athlete_day_features.csv
```

The backend data loader reads from that exact path. fileciteturn26file2

---

## 7) Neo4j setup

SimuCare uses Neo4j for graph-grounded reasoning. The backend is configured to connect to:

```text
bolt://localhost:7687
```

with the user and password stored in `.env`. fileciteturn26file0

### On the new machine
1. Install Neo4j Desktop or run Neo4j using Docker.
2. Create a database.
3. Ensure Bolt is exposed on port `7687`.
4. Set the Neo4j username/password to match your `.env`.
5. Load the graph relationships used by the project.

### Why this matters
The GraphRAG layer in the orchestrator queries Neo4j for causal relationships and converts those into graph context for the insight generator. If Neo4j is not running or the connection details are wrong, the backend will fail when it reaches graph reasoning. The orchestrator builds the graph context after load/recovery/risk analysis and passes it into the insight step. fileciteturn26file4

### If you do not have the graph loaded yet
You need to import the graph dataset that contains relationships such as:
- `sleep_debt -> poor recovery`
- `hrv_deviation -> poor recovery`
- `rhr_deviation -> fatigue -> injury risk`
- `acr_training_load -> injury risk`

The backend’s GraphRAG layer depends on those causal links to generate explainable insights.

---

## 8) Backend setup and run order

### Step 1 — Create and activate a virtual environment

#### Windows PowerShell
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install Python dependencies
Install the packages used by the backend:

```bash
pip install fastapi uvicorn pandas numpy python-dotenv langchain langgraph neo4j groq pydantic
```

If you already have a `requirements.txt`, you can use that instead.

### Step 3 — Ensure `.env` is present
Place the `.env` file in the project root with your local Groq and Neo4j credentials.

### Step 4 — Make sure the processed data exists
Confirm `data/processed/athlete_day_features.csv` has been created by `new_reader.py`. fileciteturn26file2 fileciteturn26file3

### Step 5 — Start Neo4j
Start the Neo4j database before the API. The backend requires a live Bolt connection. fileciteturn26file0

### Step 6 — Run the FastAPI backend
If your entrypoint is `api/main.py`, run:

```bash
uvicorn api.main:app --reload
```

The earlier script comments also show this backend convention. fileciteturn26file2

### Step 7 — Test the endpoint
The app uses the daily simulation route:

```text
GET /user/{user_id}/simulate-day?day=YYYY-MM-DD
```

Example:

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/user/779b2e98-d061-4748-bcef-78b1c43570ba/simulate-day?day=21-09-2025' \
  -H 'accept: application/json'
```

The response should include:
- `updated_state`
- `training_load`
- `recovery`
- `risk`
- `insight`
- `dialogue`

---

## 9) Frontend setup and run order

If you are using the React frontend:

### Step 1 — Enter the frontend folder
```bash
cd frontend
```

### Step 2 — Install frontend dependencies
```bash
npm install
```

If your frontend uses Tailwind, make sure the Tailwind config and PostCSS config are present and correct.

### Step 3 — Update the API URL if needed
In the React app, point the fetch call to the backend endpoint on the target machine. If the backend host changes, update:

```text
http://127.0.0.1:8000/user/<user_id>/simulate-day?day=21-09-2025
```

### Step 4 — Start the frontend
```bash
npm run dev
```

### Step 5 — Open the browser
Open the Vite address shown in the terminal, usually:

```text
http://localhost:5173
```

---

## 10) Runtime order summary

Use this exact order on a fresh machine:

1. Copy the repo
2. Update hardcoded raw-data folders in `new_reader.py` and `reader.py`
3. Configure `.env`
4. Start Neo4j and load the graph
5. Run `python new_reader.py`
6. Verify `data/processed/athlete_day_features.csv`
7. Install backend dependencies
8. Start the FastAPI backend
9. Install frontend dependencies
10. Start the frontend
11. Open the UI and test `/simulate-day`

---

## 11) What the system outputs

### Daily
A single-day simulation with:
- training load analysis
- recovery analysis
- injury-risk analysis
- graph-grounded explanation
- AI-generated insight
- athlete/coaching/doctor dialogue

### Weekly
A 7-day trend-based summary that highlights short-term load and recovery patterns.

### Monthly
A 30-day aggregated summary that shows longer-term patterns, risk distribution, and strategic direction.

The same processed dataset and reasoning pipeline support all three levels of output.

---

## 12) Troubleshooting

### Neo4j connection error
If the backend says it cannot connect to localhost:7687:
- confirm Neo4j is running
- confirm Bolt is enabled
- check username/password in `.env`
- check the graph database is accessible from the machine

### File not found for the processed dataset
If the backend cannot load data:
- confirm `data/processed/athlete_day_features.csv` exists
- confirm `DATA_PATH` in `data_access.py` points to the right location. fileciteturn26file2

### Raw data path errors
If preprocessing fails:
- replace the Windows absolute paths in `new_reader.py` with local paths on the new machine. fileciteturn26file3

### API rate limits
If Groq returns a rate-limit error:
- reduce the number of LLM calls,
- shorten prompts,
- retry after the wait time in the error message,
- or temporarily test with fewer requests.

### Blank or missing dialogue
If dialogue is missing from the UI:
- confirm the API is returning the `dialogue` object,
- confirm the frontend is using the correct endpoint,
- and confirm the backend is not failing earlier in the Neo4j or LLM steps.

---

## 13) Notes for maintainers

- Keep the `.env` file local and out of version control. fileciteturn26file1
- Keep the processed athlete-day CSV in `data/processed/`.
- Use `new_reader.py` when rebuilding the final dataset; `reader.py` is mainly for exploration and inspection.
- Keep the state schema in `state.py` aligned with the values returned by your LangGraph nodes. fileciteturn26file6
- If you modify the graph reasoning logic, make sure the Neo4j data and the graph context generation stay consistent.

---

## 14) Short project summary

SimuCare is an end-to-end athlete performance intelligence system that combines data preprocessing, simulation, multi-agent reasoning, graph-grounded explainability, and an interactive UI. It is designed to be runnable on another machine as long as the raw CSV paths, `.env` values, Neo4j server, and processed dataset path are configured correctly.
