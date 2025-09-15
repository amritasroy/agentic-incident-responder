from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from .tools import (
    TimeSeriesIn, SearchLogsIn, AnomalyScoreIn, KBQueryIn,
    get_timeseries, search_logs, anomaly_score, kb_query, create_ticket,
    TicketIn
)
from .rag import SimpleRAG
from .memory import ShortTerm, LongTerm
import json, textwrap, os

# Optional HF planner
USE_HF_PLANNER = True
try:
    from .llm import call_planner
except Exception:
    call_planner = None  # fall back

class AgentState(BaseModel):
    scenario: str
    description: str
    confidence: float = 0.0
    evidence: Dict[str, Any] = {}
    report_md: str = ""

rag = SimpleRAG()

# ---- Nodes ----
def plan_node(state: AgentState, st: ShortTerm):
    plan = None
    if USE_HF_PLANNER and call_planner is not None:
        try:
            plan = call_planner(state.description)
        except Exception as e:
            st.log("plan:error", {"msg": str(e)})
            plan = None

    if plan is None:
        plan = {
            "steps": ["triage", "timeseries", "logs", "hypothesize", "verify", "remediate"],
            "stop_condition": "confidence>=0.6 and evidence_sources>=2",
            "confidence_hint": 0.3,
        }
    st.log("plan", {"plan": plan})
    return state

# def collect_evidence_node(state: AgentState, st: ShortTerm):
#     ts = get_timeseries(TimeSeriesIn(sensor_id="sensor_A", window=300), state.scenario)
#     st.log("tool:get_timeseries", {"n": len(ts.points)})
#     anom = anomaly_score(AnomalyScoreIn(points=ts.points))
#     st.log("tool:anomaly_score", {"score": anom.score})

#     logs = search_logs(SearchLogsIn(query="error|warn|vibration|packet|overheat", scenario=state.scenario))
#     st.log("tool:search_logs", {"hits": len(logs.hits)})

#     kb = kb_query(KBQueryIn(issue=state.description, top_k=3), retriever=rag)
#     st.log("tool:kb_query", {"notes": [n.id for n in kb.notes]})

#     state.evidence = {
#         "anomaly_score": anom.score,
#         "logs": logs.hits[:5],
#         "kb": [n.model_dump() for n in kb.notes],
#     }
#     return state

def collect_evidence_node(state: AgentState, st: ShortTerm):
    ts = get_timeseries(TimeSeriesIn(sensor_id="sensor_A", window=300), state.scenario)
    st.log("tool:get_timeseries", {"n": len(ts.points)})
    anom = anomaly_score(AnomalyScoreIn(points=ts.points))
    st.log("tool:anomaly_score", {"score": anom.score})

    logs = search_logs(SearchLogsIn(query="error|warn|vibration|packet|overheat|gateway|backhaul|loss|cpu", scenario=state.scenario))
    st.log("tool:search_logs", {"hits": len(logs.hits)})

    # 🔑 Build a stronger RAG query from description + log messages
    top_msgs = " ".join([str(h.get("msg", "")) for h in logs.hits[:5]])
    rag_query_text = f"{state.description} {top_msgs}".strip()

    kb = kb_query(KBQueryIn(issue=rag_query_text, top_k=3), retriever=rag)
    st.log("tool:kb_query", {"notes": [n.id for n in kb.notes]})

    state.evidence = {
        "anomaly_score": anom.score,
        "logs": logs.hits[:5],
        "kb": [n.model_dump() for n in kb.notes],
    }
    return state

def decide_node(state: AgentState, st: ShortTerm):
    sources = 0
    if state.evidence.get("anomaly_score", 0) >= 0.6:
        sources += 1
    if len(state.evidence.get("logs", [])) > 0:
        sources += 1
    if len(state.evidence.get("kb", [])) > 0:
        sources += 1
    conf = 0.2 + 0.3 * (1 if state.evidence.get("anomaly_score",0)>=0.6 else 0) + 0.25 * (sources>=2)
    state.confidence = min(1.0, conf)
    st.log("decide", {"sources": sources, "confidence": state.confidence})
    return state

def maybe_gather_more(state: AgentState, st: ShortTerm):
    if state.confidence < 0.6:
        logs = search_logs(SearchLogsIn(query="bearing|gateway|temp|cpu", scenario=state.scenario))
        prev = len(state.evidence.get("logs", []))
        state.evidence["logs"] = (state.evidence.get("logs", []) + logs.hits)[:10]
        st.log("branch:more_evidence", {"added": len(state.evidence["logs"]) - prev})
    return state

def write_node(state: AgentState, st: ShortTerm):
    sources = 0
    if state.evidence.get("anomaly_score", 0) >= 0.6: sources += 1
    if len(state.evidence.get("logs", [])) > 0: sources += 1
    if len(state.evidence.get("kb", [])) > 0: sources += 1

    if sources < 2 or state.confidence < 0.6:
        st.log("guardrails", {"ticket_allowed": False})
        verdict = "Human confirmation required."
    else:
        st.log("guardrails", {"ticket_allowed": True})
        verdict = "Proceed to remediation: schedule bearing inspection and reduce load by 10%."

    ev_logs = state.evidence.get("logs", [])
    kb_cites = "\n".join([f"- {n['id']} (score={n['score']:.3f})" for n in state.evidence.get("kb", [])])

    md = f"""
# Incident Report: {state.scenario}

**Summary**: {state.description}

**Confidence**: {state.confidence:.2f}

## Evidence
- **Anomaly Score**: {state.evidence.get('anomaly_score',0):.2f}
- **Log hits** (trimmed):
{json.dumps(ev_logs[:3], indent=2) if ev_logs else 'None'}
- **RAG citations**:
{kb_cites or 'None'}

## Recommendation
{verdict}

{st.to_markdown()}
"""
    state.report_md = textwrap.dedent(md)
    st.log("write", {"chars": len(state.report_md)})
    return state

def ticket_node(state: AgentState, st: ShortTerm):
    allow = (state.confidence >= 0.6 and (len(state.evidence.get("logs",[]))>0) and len(state.evidence.get("kb",[]))>0)
    if allow:
        path = create_ticket({"payload": {"markdown": state.report_md}})
        st.log("ticket", {"path": path.path})
    else:
        st.log("ticket", {"skipped": True})
    return state

# ---- Graph wiring ----
def build_graph():
    g = StateGraph(AgentState)
    st = ShortTerm()

    g.add_node("plan", lambda s: plan_node(s, st))
    g.add_node("collect_evidence", lambda s: collect_evidence_node(s, st))
    g.add_node("decide", lambda s: decide_node(s, st))
    g.add_node("maybe_more", lambda s: maybe_gather_more(s, st))
    g.add_node("write", lambda s: write_node(s, st))
    g.add_node("ticket", lambda s: ticket_node(s, st))

    g.add_edge(START, "plan")
    g.add_edge("plan", "collect_evidence")
    g.add_edge("collect_evidence", "decide")
    g.add_edge("decide", "maybe_more")
    g.add_edge("maybe_more", "write")
    g.add_edge("write", "ticket")
    g.add_edge("ticket", END)

    return g.compile()
