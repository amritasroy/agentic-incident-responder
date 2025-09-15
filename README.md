# Agentic Incident Responder (Industrial IoT)

> End-to-end, interview-ready agent that plans → gathers evidence (timeseries/logs/RAG) → decides with guardrails → writes a citation-backed ticket. Includes a Streamlit demo and an evaluation harness (metrics + confusion matrix).

**Tech:** Python · LangGraph · Hugging Face · Streamlit · rank-bm25 · scikit-learn

## Quickstart

```bash
# 1) Create & activate venv (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Run a scenario from CLI
python -m app.main --scenario bearing_wear_03 --verbose

# 4) Launch the UI
streamlit run ui\dashboard.py

# 5) Eval all scenarios
python -m eval.harness
```

## What it does

- **Agent loop**: `plan → collect_evidence → decide → write → ticket`.
- **Tools**: timeseries anomaly (robust MAD/z-score), log search (JSONL), **BM25 RAG** over Ops notes.
- **Guardrails**: require ≥2 independent evidence sources **and** a confidence threshold before creating a ticket.
- **Artifacts**: Markdown incident report with citations + optional ticket file under `models/`.
- **Eval**: batch harness writing `eval/results/metrics.csv` + `confusion_matrix.png`.
- **UI**: scenario picker, evidence/RAG/report tabs, and one-click **download ticket**.

## Screenshots

**RAG citations tab**

![](<assets/04_Screenshot_(1464).png>)

**Evidence tab (anomaly + logs)**

![](<assets/05_Screenshot_(1465).png>)

**Packet-loss scenario with ticket download**

![](<assets/07_Screenshot_(1467).png>)

**Evaluation run (metrics table + confusion matrix)**

![](<assets/09_Screenshot_(1469).png>)

## Architecture (high level)

```text
[Planner (HF)] → [Tools: timeseries | logs | RAG] → [Decide (guardrails)] → [Write report] → [Ticket]
```

**Decide** node counts evidence sources (anomaly/logs/RAG) and computes confidence. Tickets are created only if policy passes.

## Repo structure

```text
agentic-inc-respond/
  app/                # graph, tools, memory, llm, prompts
  data/               # timeseries csv, logs jsonl, kb notes, scenarios
  eval/               # harness + results (metrics.csv, confusion_matrix.png)
  models/             # generated tickets (*.md)
  ui/                 # Streamlit dashboard
  requirements.txt    # deps
```

## How to demo (60s)

1. Open the **UI** → pick `bearing_wear_03` → Run Agent.
2. Show **Evidence** (anomaly + logs), **RAG** citations, and **Final Report**.
3. Click **Download ticket (.md)**.
4. Click **Run Evaluation** to regenerate metrics + confusion matrix.

## Notes

- The planner uses a small Hugging Face text-generation model for portability. You can swap models via the UI field.
- RAG uses **rank-bm25** (pure Python) for robust retrieval on short notes; no C++ build tools required.
- A deterministic mode is available (confidence override per scenario) to keep demos stable.

## Author

Amrita Sinha Roy

## License

MIT License
