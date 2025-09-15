from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import json, time, pathlib
from sklearn.ensemble import IsolationForest
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------- Pydantic Schemas ----------
class TimeSeriesIn(BaseModel):
    sensor_id: str
    window: int = Field(600, description="seconds")

class TimeSeriesOut(BaseModel):
    points: List[float]
    sampling_hz: float = 1.0

class SearchLogsIn(BaseModel):
    query: str
    scenario: str

class SearchLogsOut(BaseModel):
    hits: List[Dict[str, Any]]

class AnomalyScoreIn(BaseModel):
    points: List[float]

class AnomalyScoreOut(BaseModel):
    score: float  # 0-1 anomaly confidence
    details: Dict[str, Any]

class KBQueryIn(BaseModel):
    issue: str
    top_k: int = 3

class KBNote(BaseModel):
    id: str
    snippet: str
    score: float

class KBQueryOut(BaseModel):
    notes: List[KBNote]

class TicketIn(BaseModel):
    payload: Dict[str, Any]

class TicketOut(BaseModel):
    path: str

# ---------- Tools (pure Python stubs; swap for real infra) ----------

def get_timeseries(inp: TimeSeriesIn, scenario: str):
    csv_path = DATA_DIR / "timeseries" / f"{scenario}.csv"
    df = pd.read_csv(csv_path)
    values = df["value"].tail(inp.window).tolist()
    return TimeSeriesOut(points=values, sampling_hz=1.0)

def search_logs(inp: SearchLogsIn):
    path = DATA_DIR / "logs" / f"{inp.scenario}.jsonl"
    hits = []
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if any(tok in json.dumps(j).lower() for tok in inp.query.lower().split("|")):
                hits.append(j)
    return SearchLogsOut(hits=hits[:20])

def _load_or_train_iso() -> IsolationForest:
    model_path = MODELS_DIR / "anomaly_model.pkl"
    if model_path.exists():
        import joblib
        return joblib.load(model_path)
    X = np.random.normal(0, 1, size=(2000, 1))
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X)
    try:
        import joblib
        joblib.dump(iso, model_path)
    except Exception:
        pass
    return iso

# def anomaly_score(inp: AnomalyScoreIn):
#     x = np.array(inp.points).reshape(-1, 1)
#     iso = _load_or_train_iso()
#     raw = iso.decision_function(x)  # higher is normal
#     conf = float(np.clip(-raw.mean(), 0, 1))
#     return AnomalyScoreOut(score=conf, details={"raw_mean": float(raw.mean())})
def anomaly_score(inp: AnomalyScoreIn) -> AnomalyScoreOut:
    import numpy as np
    x = np.array(inp.points, dtype=float)
    n = len(x)
    if n < 5:
        return AnomalyScoreOut(score=0.0, details={"reason": "too_short", "n": n})

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-8  # median absolute deviation (robust)
    peak = float(np.max(np.abs(x - med)))
    peak_to_mad = peak / mad

    # Map robust spike ratio to [0,1]; 3≈mild, 6≈clear, 10+≈very strong
    score = max(0.0, min(1.0, (peak_to_mad - 3.0) / 7.0))
    return AnomalyScoreOut(score=float(score), details={"peak_to_mad": peak_to_mad, "median": med})


def kb_query(inp: KBQueryIn, retriever):
    docs = retriever(inp.issue, top_k=inp.top_k)
    return KBQueryOut(notes=[KBNote(id=d["id"], snippet=d["snippet"], score=float(d["score"])) for d in docs])

# def create_ticket(inp: TicketIn):
#     ts = int(time.time())
#     out_path = MODELS_DIR / f"ticket_{ts}.md"
#     with open(out_path, "w") as f:
#         f.write(inp.payload["markdown"])
#     return TicketOut(path=str(out_path))
def create_ticket(inp: TicketIn | dict) -> TicketOut:
    ts = int(time.time())
    out_path = MODELS_DIR / f"ticket_{ts}.md"
    # support both pydantic model and plain dict
    payload = inp.payload if hasattr(inp, "payload") else inp["payload"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload["markdown"])
    return TicketOut(path=str(out_path))
