import streamlit as st
from typing import TypedDict, Optional, List, Dict, Any
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# -------------------------------
# PAGE
# -------------------------------
st.set_page_config(page_title="Loan Approval - LangGraph (Incremental Loop)", layout="wide")
st.title("🏦 Loan Approval (LangGraph: Incremental Loop + Human-in-the-loop)")

# -------------------------------
# LLM
# -------------------------------
def get_llm(api_key, model, temp):
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temp
    )

# -------------------------------
# STATE
# -------------------------------
class LoanState(TypedDict):
    # input
    name: str
    income: float
    credit_score: int
    loan_amount: float

    # evolving
    risk: Optional[str]
    reason: Optional[str]
    confidence: Optional[float]
    gaps: Optional[List[str]]
    evidence: Optional[Dict[str, Any]]
    history: List[Dict[str, Any]]

    # human
    human_notes: Optional[str]
    income_verified: Optional[str]

    # control
    iteration: int
    max_iters: int
    decision: Optional[str]

    # deps
    llm: object


# -------------------------------
# UTIL
# -------------------------------
def safe_json(text: str, fallback: dict):
    try:
        return json.loads(text)
    except:
        return fallback


# -------------------------------
# NODE: ASSESS
# -------------------------------
def assess_node(state: LoanState):
    llm = state["llm"]

    prompt = f"""
You are a loan risk evaluator.

Applicant:
- Income: {state['income']}
- Credit Score: {state['credit_score']}
- Loan Amount: {state['loan_amount']}

Existing Evidence:
{state.get('evidence')}

Human Inputs:
- Notes: {state.get('human_notes')}
- Income Verified: {state.get('income_verified')}

Return STRICT JSON:
{{
  "risk": "low|medium|high",
  "confidence": 0.0-1.0,
  "reason": "...",
  "gaps": ["missing item 1", "missing item 2"]
}}
"""

    resp = llm.invoke(prompt).content
    parsed = safe_json(resp, {
        "risk": "medium",
        "confidence": 0.4,
        "reason": resp,
        "gaps": ["income verification", "bank statement summary"]
    })

    risk = parsed.get("risk", "medium")
    conf = float(parsed.get("confidence", 0.4))
    reason = parsed.get("reason", "")
    gaps = parsed.get("gaps", [])

    hist = state.get("history", [])
    hist.append({
        "iteration": state["iteration"] + 1,
        "stage": "assess",
        "risk": risk,
        "confidence": conf,
        "gaps": gaps
    })

    return {
        "risk": risk,
        "confidence": conf,
        "reason": reason,
        "gaps": gaps,
        "history": hist,
        "iteration": state["iteration"] + 1
    }


# -------------------------------
# NODE: PLAN
# -------------------------------
def plan_node(state: LoanState):
    gaps = state.get("gaps", [])[:2]
    plan = {"to_fetch": gaps}

    hist = state.get("history", [])
    hist.append({
        "iteration": state["iteration"],
        "stage": "plan",
        "plan": plan
    })

    return {"plan": plan, "history": hist}


# -------------------------------
# NODE: COLLECT EVIDENCE (FIXED NAME)
# -------------------------------
def collect_evidence_node(state: LoanState):
    plan = state.get("plan", {}).get("to_fetch", [])
    ev = state.get("evidence", {}) or {}

    for item in plan:
        if "income" in item.lower():
            ev["income_verified_flag"] = True
            ev["income_stability"] = "stable"

        if "bank" in item.lower():
            ev["bank_statement_summary"] = {
                "avg_balance": 75000,
                "overdrafts": 0
            }

        if "employment" in item.lower():
            ev["employment_verified"] = True

    hist = state.get("history", [])
    hist.append({
        "iteration": state["iteration"],
        "stage": "collect_evidence",
        "added": plan,
        "evidence": ev
    })

    return {"evidence": ev, "history": hist}


# -------------------------------
# ROUTING
# -------------------------------
def route_after_assess(state: LoanState):
    if state["confidence"] >= 0.75:
        return "approve" if state["risk"] == "low" else "reject"

    if state["iteration"] < state["max_iters"]:
        return "plan"

    return "human"


# -------------------------------
# FINAL NODES
# -------------------------------
def approve_node(state):
    return {"decision": "APPROVED"}


def reject_node(state):
    return {"decision": "REJECTED"}


# -------------------------------
# GRAPH
# -------------------------------
def build_graph():
    g = StateGraph(LoanState)

    g.add_node("assess", assess_node)
    g.add_node("plan", plan_node)
    g.add_node("collect_evidence", collect_evidence_node)  # ✅ FIXED
    g.add_node("approve", approve_node)
    g.add_node("reject", reject_node)

    g.set_entry_point("assess")

    g.add_conditional_edges(
        "assess",
        route_after_assess,
        {
            "approve": "approve",
            "reject": "reject",
            "plan": "plan",
            "human": END
        }
    )

    g.add_edge("plan", "collect_evidence")
    g.add_edge("collect_evidence", "assess")

    g.add_edge("approve", END)
    g.add_edge("reject", END)

    return g.compile()


# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
model = st.sidebar.selectbox("Model", ["models/gemini-2.5-flash", "models/gemini-1.5-pro","models/gemini-2.0-flash","models/gemini-2.0-pro"])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
max_iters = st.sidebar.slider("Max Auto Iterations", 1, 5, 2)

# -------------------------------
# INPUT
# -------------------------------
st.subheader("👤 Applicant Details")

name = st.text_input("Name")
income = st.number_input("Monthly Income", value=50000)
credit = st.number_input("Credit Score", value=650)
loan = st.number_input("Loan Amount", value=200000)

run = st.button("🚀 Evaluate Loan")

# -------------------------------
# INITIAL RUN
# -------------------------------
if run:
    if not api_key:
        st.error("Enter API key")
        st.stop()

    llm = get_llm(api_key, model, temp)
    graph = build_graph()

    state = {
        "name": name,
        "income": income,
        "credit_score": credit,
        "loan_amount": loan,
        "iteration": 0,
        "max_iters": max_iters,
        "evidence": {},
        "history": [],
        "llm": llm
    }

    result = graph.invoke(state)
    st.session_state.state = result


# -------------------------------
# DISPLAY
# -------------------------------
if "state" in st.session_state:
    s = st.session_state.state

    st.subheader("🤖 Current Assessment")
    st.write(f"Risk: **{s.get('risk')}** | Confidence: **{round(s.get('confidence',0),2)}**")
    st.write(f"Reason: {s.get('reason')}")
    st.write(f"Iteration: {s.get('iteration')}")

    st.subheader("📜 Iteration History")
    for h in s.get("history", []):
        st.write(h)

    # -------------------------------
    # HUMAN ESCALATION
    # -------------------------------
    if not s.get("decision") and s.get("confidence", 0) < 0.75 and s["iteration"] >= s["max_iters"]:
        st.subheader("🧑‍💻 Human Review (Escalation)")

        notes = st.text_area("Reviewer Notes")
        verified = st.selectbox("Income Verified?", ["Yes", "No"])

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve"):
                st.session_state.state["decision"] = "APPROVED"
                st.rerun()

        with col2:
            if st.button("❌ Reject"):
                st.session_state.state["decision"] = "REJECTED"
                st.rerun()

        with col3:
            if st.button("🔁 Re-evaluate"):
                graph = build_graph()
                new_state = {
                    **s,
                    "human_notes": notes,
                    "income_verified": verified
                }
                result = graph.invoke(new_state)
                st.session_state.state = result
                st.rerun()

    if s.get("decision"):
        st.subheader("🏁 Final Decision")
        st.success(s.get("decision"))

    with st.expander("🔍 Full State"):
        st.json(s)
