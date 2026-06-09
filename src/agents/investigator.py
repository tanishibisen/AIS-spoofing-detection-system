import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import pandas as pd
from pathlib import Path
from rag.retriever import retrieve_context, explain_vessel

# ── State definition ─────────────────────────────────────
class InvestigationState(TypedDict):
    mmsi: str
    flags: dict
    spoofing_score: int
    country: str
    speed: float
    rag_context: str
    severity: str
    report: str
    recommendation: str

# ── Node 1: Extract vessel details ───────────────────────
def extract_details(state: InvestigationState) -> InvestigationState:
    print(f"  [Node 1] Extracting details for MMSI {state['mmsi']}...")
    score = state["spoofing_score"]
    flags = state["flags"]

    active_flags = [k for k, v in flags.items() if v]
    print(f"  → Active flags: {active_flags}")
    print(f"  → Spoofing score: {score}")
    return state

# ── Node 2: Query RAG knowledge base ─────────────────────
def query_knowledge_base(state: InvestigationState) -> InvestigationState:
    print(f"  [Node 2] Querying RAG knowledge base...")
    flags = state["flags"]
    active_flags = [k for k, v in flags.items() if v]
    query = f"AIS spoofing {' '.join(active_flags)} MMSI violation maritime regulation"
    docs = retrieve_context(query, k=3)
    context = "\n\n".join([
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])
    print(f"  → Retrieved {len(docs)} regulation chunks")
    state["rag_context"] = context
    return state

# ── Node 3: Assess severity ───────────────────────────────
def assess_severity(state: InvestigationState) -> InvestigationState:
    print(f"  [Node 3] Assessing severity...")
    score = state["spoofing_score"]
    flags = state["flags"]

    if score >= 4 or flags.get("fake_mmsi"):
        severity = "HIGH"
    elif score >= 2 or flags.get("heading_cog_mismatch"):
        severity = "MEDIUM"
    else:
        severity = "LOW"

    print(f"  → Severity: {severity}")
    state["severity"] = severity
    return state

# ── Node 4: Generate investigation report ────────────────
def generate_report(state: InvestigationState) -> InvestigationState:
    print(f"  [Node 4] Generating investigation report...")
    flags = state["flags"]
    active_flags = [k for k, v in flags.items() if v]
    flag_lines = "\n".join(f"  • {f.replace('_', ' ').title()}" for f in active_flags)

    report = f"""
╔══════════════════════════════════════════════════════════╗
║           AIS SPOOFING INVESTIGATION REPORT              ║
╚══════════════════════════════════════════════════════════╝

VESSEL DETAILS
--------------
MMSI          : {state['mmsi']}
Country       : {state['country']}
Speed         : {state['speed']} knots
Spoofing Score: {state['spoofing_score']}/5
Severity      : {state['severity']}

DETECTED VIOLATIONS
-------------------
{flag_lines}

REGULATORY CONTEXT (ITU-R M.1371 / IMO SN/Circ.227)
----------------------------------------------------
{state['rag_context'][:800]}

RECOMMENDATION
--------------
{state['recommendation']}

STATUS: {'🚨 ESCALATED TO AUTHORITIES' if state['severity'] == 'HIGH' else '⚠️ FLAGGED FOR MONITORING'}
"""
    state["report"] = report
    return state

# ── Node 5: Make recommendation ──────────────────────────
def make_recommendation(state: InvestigationState) -> InvestigationState:
    print(f"  [Node 5] Making recommendation...")
    severity = state["severity"]
    flags = state["flags"]

    if severity == "HIGH":
        rec = (
            "IMMEDIATE ACTION REQUIRED:\n"
            "  1. Report to flag state maritime administration\n"
            "  2. Alert nearest coast guard authority\n"
            "  3. Cross-reference with sanctions database\n"
            "  4. Do not allow port entry until identity verified"
        )
    elif severity == "MEDIUM":
        rec = (
            "MONITORING REQUIRED:\n"
            "  1. Continue tracking vessel movements\n"
            "  2. Request voluntary identity verification\n"
            "  3. Log for pattern analysis\n"
            "  4. Escalate if additional flags detected"
        )
    else:
        rec = (
            "STANDARD MONITORING:\n"
            "  1. Log anomaly for record\n"
            "  2. No immediate action required\n"
            "  3. Review if pattern repeats"
        )

    state["recommendation"] = rec
    return state

# ── Conditional edge ──────────────────────────────────────
def route_by_severity(state: InvestigationState) -> str:
    if state["severity"] == "HIGH":
        return "generate_report"
    elif state["severity"] == "MEDIUM":
        return "generate_report"
    else:
        return "generate_report"

# ── Build the graph ───────────────────────────────────────
def build_investigation_graph():
    graph = StateGraph(InvestigationState)

    graph.add_node("extract_details", extract_details)
    graph.add_node("query_knowledge_base", query_knowledge_base)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("make_recommendation", make_recommendation)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("extract_details")
    graph.add_edge("extract_details", "query_knowledge_base")
    graph.add_edge("query_knowledge_base", "assess_severity")
    graph.add_edge("assess_severity", "make_recommendation")
    graph.add_edge("make_recommendation", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()

# ── Run investigation ─────────────────────────────────────
def investigate_vessel(mmsi: str, flags: dict, score: int,
                       country: str, speed: float) -> str:
    print(f"\n{'='*60}")
    print(f"STARTING INVESTIGATION: MMSI {mmsi}")
    print(f"{'='*60}")

    app = build_investigation_graph()
    result = app.invoke({
        "mmsi": mmsi,
        "flags": flags,
        "spoofing_score": score,
        "country": country,
        "speed": speed,
        "rag_context": "",
        "severity": "",
        "report": "",
        "recommendation": ""
    })
    return result["report"]

if __name__ == "__main__":
    report = investigate_vessel(
        mmsi="987654321",
        flags={
            "fake_mmsi": True,
            "aivdo_unknown": True,
            "heading_cog_mismatch": False,
            "anchored_moving": False,
            "position_jump": False
        },
        score=5,
        country="Unknown",
        speed=12.3
    )
    print(report)