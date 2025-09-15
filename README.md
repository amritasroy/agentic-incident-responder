# Agentic Incident Responder (Industrial IoT)

End-to-end, **agentic incident responder**: plan → act (tool calls) → check → reflect, with RAG, memory, evaluation, and a demoable CLI.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
# .venv\Scripts\activate.bat
pip install -r requirements.txt

# (Optional) choose a local/remote HF model:
# export HUGGINGFACE_MODEL=HuggingFaceH4/zephyr-7b-beta  # needs GPU/CPU power
# or keep the default distilgpt2 (tiny, demo only)

python app.main --scenario bearing_wear_03 --verbose
```

**Eval:** `python eval.harness` → writes CSV and confusion matrix under `eval/results/`.

# streamlit run ui\dashboard.py

## Architecture

Planner (HF pipeline in `app/llm.py`) ↔ Tools (`app/tools.py`) ↔ Memory (`app/memory.py`) ↔ RAG (`app/rag.py`).  
Guardrails: ticket requires ≥2 independent evidence sources and confidence ≥ 0.6.

## Why it's agentic

- **Autonomy:** planner proposes steps + stop condition
- **Tool use:** time series, logs, anomaly model, KB, ticket
- **Memory:** short-term trace + long-term vector store hooks
- **Reflection-ready:** critic prompt included for future use

## Demo tips

- Record a 60s GIF of CLI trace and final ticket
- Put metrics table + the confusion matrix image in README
- Emphasize planning, guardrails, and citations
