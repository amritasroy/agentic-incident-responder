from __future__ import annotations
import os, json
from pathlib import Path
from transformers import pipeline

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "app" / "prompts"

# You can override with: export HUGGINGFACE_MODEL=HuggingFaceH4/zephyr-7b-beta
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL", "distilgpt2")
_gen = pipeline("text-generation", model=MODEL_NAME, device=-1)

def call_planner(desc: str) -> dict:
    """Use a Hugging Face model to produce a JSON planning object.
    Falls back to a static plan if parsing fails.
    """
    prompt = (PROMPTS_DIR / "planner.md").read_text()
    user = f"Incident description: {desc}\n\nProduce JSON plan with fields: steps[], stop_condition, confidence_hint (0-1)."

    txt = _gen(prompt + "\n" + user, max_length=256, num_return_sequences=1)[0]["generated_text"]
    start = txt.find("{"); end = txt.rfind("}")
    snippet = txt[start:end+1] if (start != -1 and end != -1 and end > start) else ""

    try:
        plan = json.loads(snippet)
        # basic shape check
        if not isinstance(plan.get("steps", []), list):
            raise ValueError("steps not list")
        return plan
    except Exception:
        return {
            "steps":["triage","timeseries","logs","hypothesize","verify","remediate"],
            "stop_condition":"confidence>=0.6 and evidence_sources>=2",
            "confidence_hint":0.3
        }
