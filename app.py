"""
app.py
======
Streamlit UI for the ED Triage & Resource Allocation Environment.

Run with:
    streamlit run app.py

Features
--------
- Task selector (Easy / Medium / Hard) at the top
- LEFT panel  : ED Floor — live colour-coded bed grid
- CENTRE panel: Patient Queue — styled table with LLM hint badges
- RIGHT panel : Action Panel — dropdown + submit button
- Bottom bar  : Running reward ticker, step count, live task score
- End screen  : AgentGrader score breakdown with per-dimension pass/fail
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

# Make sure the package is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── ED Triage imports ────────────────────────────────────────────────────────
from ed_triage import (
    Action,
    AgentGrader,
    EDTriageEnv,
    TASK_EASY,
    TASK_MEDIUM,
    TASK_HARD,
    get_task,
)
from ed_triage.llm_helper import analyze_complaint, hint_badge_html, set_api_key, chat_with_patient_ai, get_autonomous_action
from ed_triage.schemas import BedInventory, EDState, Patient

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="ED Triage Sim",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Global CSS — dark medical theme
# ============================================================
st.markdown(
    """
<style>
/* ─── Base ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #fdfbfb;
    color: #374151;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

/* ─── Top bar ─────────────────────────────────────── */
.top-bar {
    background: linear-gradient(135deg, #fdf2f8 0%, #dcfce7 100%);
    border: 1px solid #fbcfe8;
    border-radius: 12px;
    padding: 14px 22px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.top-bar h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 800;
    color: #ec4899;
    letter-spacing: -0.5px;
}
.top-bar .subtitle {
    font-size: 0.85rem;
    color: #10b981;
    font-weight: 600;
    margin-top: 2px;
}

/* ─── Panel cards ─────────────────────────────────── */
.panel-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    min-height: 200px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.panel-title {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #10b981;
    margin-bottom: 12px;
    border-bottom: 2px solid #dcfce7;
    padding-bottom: 6px;
}

.monitor-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.monitor-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 800;
    color: #4b5563;
    text-transform: uppercase;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}

/* ─── Bed grid ─────────────────────────────────────── */
.bed-section-label {
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 10px 0 6px 0;
}
.bed-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.bed {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.70rem;
    font-weight: 800;
    cursor: default;
    transition: transform 0.15s;
    border: 2px solid rgba(0,0,0,0.02);
}
.bed:hover { transform: scale(1.12); }
.bed-free      { background: #dcfce7; color: #166534; border-color: #bbf7d0;}
.bed-occupied  { background: #fee2e2; color: #991b1b; border-color: #fecaca;}
.bed-critical  { background: #fef08a; color: #854d0e; border-color: #fde047;
                  box-shadow: 0 0 8px rgba(253,224,71,0.6); }

/* ─── Stats metric strip ──────────────────────────── */
.metric-strip {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    gap: 32px;
    align-items: center;
    margin-top: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-item { text-align: center; }
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ec4899;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.metric-positive { color: #10b981 !important; }
.metric-negative { color: #ef4444 !important; }

/* ─── Patient queue table ─────────────────────────── */
.patient-row {
    background: #fdfbfb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 7px;
    transition: background 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.patient-row:hover { background: #f9fafb; }
.patient-row.is-head {
    border-color: #ec4899;
    background: #fdf2f8;
    box-shadow: 0 0 0 2px #fbcfe8;
}
.sev-badge {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 900;
}
.sev-1 { background:#fef2f2; color:#ef4444; border: 2px solid #fecaca;}
.sev-2 { background:#fff7ed; color:#f97316; border: 2px solid #fed7aa;}
.sev-3 { background:#fefce8; color:#eab308; border: 2px solid #fef08a;}
.sev-4 { background:#f0fdf4; color:#22c55e; border: 2px solid #bbf7d0;}
.sev-5 { background:#eff6ff; color:#3b82f6; border: 2px solid #bfdbfe;}

.patient-name { font-weight: 800; font-size: 1.0rem; color: #1f2937; }
.patient-sub  { font-size: 0.80rem; color: #6b7280; font-weight: 500;}
.wait-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 800;
}
.wait-ok   { background:#dcfce7; color:#166534; }
.wait-warn { background:#fef08a; color:#854d0e; }
.wait-crit { background:#fee2e2; color:#991b1b; }

/* ─── Action panel ─────────────────────────────────── */
.action-patient-card {
    background: #fdfbfb;
    border: 2px solid #ec4899;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 4px rgba(236, 72, 153, 0.1);
}
.action-patient-name { font-size: 1.15rem; font-weight: 800; color: #1f2937; }
.action-vitals { font-size: 0.82rem; color: #6b7280; margin-top: 6px; font-weight: 600;}

/* ─── End screen ───────────────────────────────────── */
.end-screen {
    text-align: center;
    padding: 40px 20px;
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
}
.end-score {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
}
.end-pass { color: #10b981; }
.end-fail { color: #ef4444; }
.dim-bar-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.dim-bar-label { width: 140px; text-align: right; font-size: 0.85rem; color: #4b5563; font-weight: 600;}
.dim-bar-track {
    flex: 1;
    height: 16px;
    background: #f3f4f6;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}
.dim-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* ─── Streamlit overrides ──────────────────────────── */
div[data-testid="stSelectbox"] > label { color: #6b7280 !important; font-size:0.85rem !important; font-weight:700 !important;}
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #10b981, #059669);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 800;
    font-size: 1.0rem;
    transition: opacity 0.2s, transform 0.1s;
    box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
}
div[data-testid="stButton"] > button:hover { opacity: 0.9; transform: translateY(-1px); }
div[data-testid="stButton"].danger > button {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2);
}
.stSelectbox [data-baseweb="select"] {
    background: #ffffff !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #1f2937 !important;
    font-weight: 600;
}
.stTextInput input {
    background: #ffffff !important;
    border: 2px solid #e5e7eb !important;
    color: #1f2937 !important;
    border-radius: 10px !important;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Session state initialisation
# ============================================================

TASK_MAP = {
    "Easy — Clear-Cut Triage": ("easy", TASK_EASY),
    "Medium — Hidden Severity": ("medium", TASK_MEDIUM),
    "Hard — Mass Casualty":     ("hard", TASK_HARD),
}


def _init_session() -> None:
    """Bootstrap session state with defaults (idempotent)."""
    if "initialised" not in st.session_state:
        st.session_state.initialised = True
        st.session_state.task_label = "Easy — Clear-Cut Triage"
        st.session_state.env = None
        st.session_state.grader = AgentGrader()
        st.session_state.episode_done = False
        st.session_state.cumulative_reward = 0.0
        st.session_state.step_count = 0
        st.session_state.llm_hints: Dict[str, Dict] = {}
        st.session_state.chat_history: Dict[str, List] = {}
        st.session_state.action_log: List[str] = []
        st.session_state.final_score = None
        st.session_state.final_breakdown = None
        st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")


def _start_episode(task_label: str) -> None:
    """Create a fresh environment for the selected task and reset state."""
    task_name, task_cfg = TASK_MAP[task_label]
    env = EDTriageEnv(**task_cfg.env_kwargs, task=task_name)
    env.reset()

    st.session_state.env = env
    st.session_state.task_label = task_label
    st.session_state.task_name = task_name
    st.session_state.episode_done = False
    st.session_state.cumulative_reward = 0.0
    st.session_state.step_count = 0
    st.session_state.llm_hints = {}
    st.session_state.chat_history = {}
    st.session_state.action_log = []
    st.session_state.final_score = None
    st.session_state.final_breakdown = None

    # Pre-fetch LLM hints for the initial queue
    _fetch_llm_hints(env.state())


def _fetch_llm_hints(state: EDState) -> None:
    """Fetch LLM triage hints for any new waiting patients."""
    api_key = st.session_state.get("api_key", "")
    for p in state.waiting_patients:
        if p.id not in st.session_state.llm_hints:
            hint = analyze_complaint(
                p.chief_complaint,
                api_key=api_key if api_key else None,
                use_llm=bool(api_key),
            )
            st.session_state.llm_hints[p.id] = hint


# ============================================================
# Rendering helpers
# ============================================================

SEV_COLOUR = {1: "sev-1", 2: "sev-2", 3: "sev-3", 4: "sev-4", 5: "sev-5"}


def _sev_int(sev: float) -> int:
    return max(1, min(5, round(sev)))




def _render_bed_grid(total: int, free: int, active_patients: List[Patient], title: str) -> None:
    """Renders a section of beds as a clinical life-support grid."""
    st.markdown(f'<div class="monitor-card">', unsafe_allow_html=True)
    
    used = total - free
    pct = int((used / total) * 100) if total > 0 else 0
    
    st.markdown(
        f'<div class="monitor-label">'
        f'<span>{title}</span>'
        f'<span style="color:{"#ef4444" if pct > 80 else "#10b981"}">{pct}% LOAD</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="bed-grid">', unsafe_allow_html=True)
    for i in range(total):
        occupied = i < used
        p_name = ""
        is_critical = False
        
        # Match patient to bed if occupied (simple index mapping for UI)
        p_list = [p for p in active_patients if p.assigned_tier == title]
        if i < len(p_list):
            p_name = p_list[i].name
            is_critical = p_list[i].severity_score <= 2.0
            
        cls = "bed-occupied" if occupied else "bed-free"
        if occupied and is_critical:
            cls = "bed-critical"
            
        tooltip = p_name if p_name else "Empty"
        st.markdown(
            f'<div class="bed {cls}" title="{tooltip}">{i+1}</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div></div>', unsafe_allow_html=True)

def _render_floor(state: EDState) -> None:
    """Left panel: live bed grid status monitor."""
    with st.container():
        st.markdown('<div class="panel-title">ED Status Monitor</div>', unsafe_allow_html=True)
        
        # ICU Section
        icu_total = state.available_beds.icu + sum(1 for p in state.active_patients if p.assigned_tier == "ICU")
        _render_bed_grid(
            icu_total,
            state.available_beds.icu,
            state.active_patients, 
            "ICU"
        )
        
        # General Ward
        gen_total = state.available_beds.general + sum(1 for p in state.active_patients if p.assigned_tier == "General")
        _render_bed_grid(
            gen_total,
            state.available_beds.general,
            state.active_patients, 
            "General"
        )
        
        # Observation
        obs_total = state.available_beds.observation + sum(1 for p in state.active_patients if p.assigned_tier == "Observation")
        _render_bed_grid(
            obs_total,
            state.available_beds.observation,
            state.active_patients, 
            "Observation"
        )
        
        st.markdown(
            """
            <div style="margin-top:10px;font-size:0.65rem;color:#9ca3af;font-weight:600;">
                <span style="color:#10b981;">● AVAILABLE</span> &nbsp;
                <span style="color:#ef4444;">● OCCUPIED</span> &nbsp;
                <span style="color:#f59e0b;">● CRITICAL</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def _render_patient_queue(state: EDState) -> None:
    """Centre panel: Native Streamlit patient queue table."""
    hints = st.session_state.llm_hints

    st.markdown(
        f'<div class="panel-card" style="padding-bottom: 4px;">'
        f'<div class="panel-title">Active Queue'
        f' <span style="font-weight:400;color:#6b7280;">({len(state.waiting_patients)} waiting)</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if not state.waiting_patients:
        st.info("Queue empty — waiting for new arrivals")
        return

    # Header Row
    c1, c2, c3, c4, c5 = st.columns([0.5, 3, 1, 1, 1])
    c2.markdown("<small style='color:#6b7280;font-weight:700;'>PATIENT & COMPLAINT</small>", unsafe_allow_html=True)
    c3.markdown("<small style='color:#6b7280;font-weight:700;text-align:center;'>WAIT TIME / AGE</small>", unsafe_allow_html=True)
    c4.markdown("<small style='color:#6b7280;font-weight:700;text-align:center;'>IDEAL DEPT</small>", unsafe_allow_html=True)
    c5.markdown("<small style='color:#6b7280;font-weight:700;text-align:right;'>PRIORITY</small>", unsafe_allow_html=True)

    for i, p in enumerate(state.waiting_patients):
        is_head = i == 0
        sev_i = _sev_int(p.severity_score)
        wt = p.wait_time
        
        hint = hints.get(p.id, {})
        badge = hint_badge_html(hint) if hint else ""
        wt_cls = "wait-ok" if wt <= 3 else ("wait-warn" if wt <= 6 else "wait-crit")

        vitals = p.vitals
        vitals_str = (
            f"BP {vitals.bp_systolic}/{vitals.bp_diastolic} | HR {vitals.hr} | "
            f"SpO₂ {vitals.spo2:.0f}% | Temp {vitals.temp}°C"
        )
        
        # Premium Patient Card
        with st.container():
            st.markdown(
                f'''
                <div class="patient-card {"active" if is_head else ""}">
                    <div class="triage-icon t-{sev_i}">{sev_i}</div>
                    <div style="flex-grow: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div class="patient-name">{p.name}</div>
                                <div class="patient-sub">{p.chief_complaint[:50]}</div>
                            </div>
                            <div style="text-align: right;">
                                <div class="wait-badge {wt_cls}">{wt} STEP{"S" if wt != 1 else ""}</div>
                                <div style="font-size: 0.65rem; color: #9ca3af; margin-top: 4px; font-weight: 700;">{p.age}Y • {p.ideal_tier}</div>
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-family: monospace; font-size: 0.75rem; color: #6b7280; background: #f8fafc; padding: 4px 8px; border-radius: 4px;">
                            {vitals_str}
                        </div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            if hint and "suggested_tier" in hint:
                st.markdown(badge, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)



def _render_action_panel(state: EDState) -> None:
    """Right panel: action selector and AI Chat"""
    patient = state.current_patient

    if patient is None:
        st.info("No patient to triage — waiting for arrivals")
        return

    sev_i = _sev_int(patient.severity_score)
    sev_cls = SEV_COLOUR[sev_i]
    hint = st.session_state.llm_hints.get(patient.id, {})

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🩺 Take Action", "🤖 AI Assistant", "🚀 Auto-Pilot"])

    with tab1:
        waiting_ids = [
            f"{p.name} (Severity {p.severity_score})" 
            for p in st.session_state.env.state().waiting_patients
        ]
        selected_label = st.selectbox("Select patient to triage", waiting_ids)
        selected_index = waiting_ids.index(selected_label) if waiting_ids else 0
        selected_patient = st.session_state.env.state().waiting_patients[selected_index] if waiting_ids else patient

        sev_i = _sev_int(selected_patient.severity_score)
        sev_cls = SEV_COLOUR[sev_i]

        st.markdown(
            f"""
            <div class="action-patient-card">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div class="sev-badge {sev_cls}">{sev_i}</div>
                    <div>
                        <div class="action-patient-name">{selected_patient.name}</div>
                        <div class="patient-sub">Age {selected_patient.age} | Wait: {selected_patient.wait_time} step(s)</div>
                    </div>
                </div>
                <div style="margin-top:10px;font-size:0.85rem;color:#374151;">
                    <strong>Complaint:</strong> {selected_patient.chief_complaint}
                </div>
                <div class="action-vitals">
                    BP {selected_patient.vitals.bp_systolic}/{selected_patient.vitals.bp_diastolic} | HR {selected_patient.vitals.hr} | SpO₂ {selected_patient.vitals.spo2:.0f}% | Temp {selected_patient.vitals.temp}°C
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action_labels = {
            "Assign ICU": Action.ASSIGN_ICU,
            "Assign General Ward": Action.ASSIGN_GENERAL,
            "Assign Observation Bay": Action.ASSIGN_OBSERVATION,
            "Hold in Queue": Action.HOLD_QUEUE,
            "Discharge Patient": Action.DISCHARGE,
        }

        chosen_label = st.selectbox("Select Action", list(action_labels.keys()), key="action_select")

        target_id = selected_patient.id if selected_patient else None

        if chosen_label == "Discharge Patient":
            active_ids = [
                f"{p.name} (Bed: {p.assigned_tier})"
                for p in st.session_state.env.state().active_patients
            ]
            discharge_label = st.selectbox("Select patient to discharge", active_ids)
            discharge_index = active_ids.index(discharge_label) if active_ids else 0
            if active_ids:
                target_id = st.session_state.env.state().active_patients[discharge_index].id

        if st.button("Submit Action", key="submit_action"):
            action = action_labels[chosen_label]
            env: EDTriageEnv = st.session_state.env
            next_state, reward, done, info = env.step(action, patient_id=target_id)

            breakdown = info.get("reward_breakdown", {})
            
            if breakdown.get("critical_miss", 0) != 0:
                st.toast("CRITICAL MISS — severity 5 patient misassigned! (-25)", icon="🚨")
            
            if breakdown.get("deterioration", 0) != 0:
                penalty = breakdown["deterioration"]
                st.toast(f"Patient deteriorated in queue ({penalty:+.0f})", icon="⚠️")
            
            if breakdown.get("bed_overflow", 0) != 0:
                st.toast(f"Bed overflow penalty! ({breakdown['bed_overflow']:+.0f})", icon="🛑")
            
            if breakdown.get("tier_reward", 0) > 0:
                st.toast(f"Correct assignment! (+{breakdown['tier_reward']:.0f})", icon="✅")

            st.session_state.cumulative_reward += reward
            st.session_state.step_count += 1
            st.session_state.episode_done = done

            _fetch_llm_hints(next_state)

            bd = info["reward_breakdown"]
            st.session_state.action_log.append({
                "step": st.session_state.step_count,
                "auto": False,
                "patient": patient.name,
                "action": chosen_label,
                "reward": reward
            })

            if done:
                grader: AgentGrader = st.session_state.grader
                score, breakdown = grader.grade(env.episode_log, task=st.session_state.task_name)
                st.session_state.final_score = score
                st.session_state.final_breakdown = breakdown

            st.rerun()

    with tab2:
        st.markdown(f"**Chat about {patient.name}**")
        if patient.id not in st.session_state.chat_history:
            st.session_state.chat_history[patient.id] = []
            
        chat_log = st.session_state.chat_history[patient.id]
        
        # Display existing chat
        _container = st.container(height=280)
        with _container:
            for message in chat_log:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat Input
        if user_msg := st.chat_input("Ask Gemini about this patient..."):
            # Display user msg
            with _container:
                st.chat_message("user").write(user_msg)
            
            # Prepare contextual data
            ctx = f"Name: {patient.name}, Age: {patient.age}, Complaint: {patient.chief_complaint}, Vitals: {patient.vitals}, Wait Time: {patient.wait_time}s"
            
            # Request response
            with _container:
                with st.spinner("Gemini is typing..."):
                    reply = chat_with_patient_ai(
                        patient_context=ctx,
                        user_message=user_msg,
                        chat_history=chat_log,
                        api_key=st.session_state.api_key
                    )
                st.chat_message("model").write(reply)
                
            # Log exchange
            chat_log.append({"role": "user", "content": user_msg})
            chat_log.append({"role": "model", "content": reply})

    with tab3:
        st.markdown("**Run Remaining Queue Autonomously**")
        st.caption("The AI Agent will take over and automatically allocate beds until the episode is done.")
        
        if st.button("Start Auto-Pilot", key="btn_auto_pilot", type="primary"):
            env: EDTriageEnv = st.session_state.env
            api_key = st.session_state.get("api_key", "")
            
            with st.spinner("Autonomous Agent taking control..."):
                done = st.session_state.episode_done
                while not done:
                    current_p = env.state().current_patient
                    p_name = current_p.name if current_p else "—"
                    
                    action = get_autonomous_action(env.state(), api_key)
                    next_state, reward, done_status, info = env.step(action)
                    
                    st.session_state.cumulative_reward += reward
                    st.session_state.step_count += 1
                    st.session_state.action_log.append({
                        "step": st.session_state.step_count,
                        "auto": True,
                        "patient": p_name,
                        "action": action.name.replace("ASSIGN_", ""),
                        "reward": reward
                    })
                    
                    done = done_status
                    
                st.session_state.episode_done = True
                grader: AgentGrader = st.session_state.grader
                score, breakdown = grader.grade(env.episode_log, task=st.session_state.task_name)
                st.session_state.final_score = score
                st.session_state.final_breakdown = breakdown
                
            st.rerun()


def _render_active_strip(state: EDState) -> None:
    """Small strip showing patients currently in beds."""
    if not state.active_patients:
        return
    items = []
    for p in state.active_patients:
        sev_i = _sev_int(p.severity_score)
        sev_cls = SEV_COLOUR[sev_i]
        items.append(
            f'<div style="background:#fdfbfb;border:1px solid #e5e7eb;'
            f'border-radius:8px;padding:6px 10px;display:inline-flex;'
            f'align-items:center;gap:6px;margin:3px;box-shadow: 0 1px 2px rgba(0,0,0,0.02);">'
            f'<div class="sev-badge {sev_cls}" style="width:20px;height:20px;'
            f'font-size:0.6rem;">{sev_i}</div>'
            f'<div><div style="font-size:0.78rem;font-weight:600;">{p.name}</div>'
            f'<div style="font-size:0.64rem;color:#6b7280;">{p.assigned_tier}</div></div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="panel-card" style="padding:10px 14px;">'
        f'<div class="panel-title">Active Patients ({len(state.active_patients)})</div>'
        f'<div style="display:flex;flex-wrap:wrap;">{"".join(items)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_metric_bar(state: EDState) -> None:
    """Bottom metrics bar: reward, steps, beds, queue length."""
    reward = st.session_state.cumulative_reward
    r_cls = "metric-positive" if reward >= 0 else "metric-negative"
    step = st.session_state.step_count
    queue = state.queue_length
    active = state.occupancy
    time_min = state.time_elapsed

    st.markdown(
        f'''
        <div class="metric-strip">
            <div class="metric-item">
                <div class="metric-value {r_cls}" style="font-size: 1.1rem;">{reward:+.0f}</div>
                <div class="metric-label" style="font-size: 0.6rem;">REWARD</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="font-size: 1.1rem; color: var(--text-main);">{step}</div>
                <div class="metric-label" style="font-size: 0.6rem;">STEPS</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="font-size: 1.1rem; color: var(--text-main);">{time_min}M</div>
                <div class="metric-label" style="font-size: 0.6rem;">SIM TIME</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="font-size: 1.1rem; color: var(--text-main);">{queue}</div>
                <div class="metric-label" style="font-size: 0.6rem;">QUEUE</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="font-size: 1.1rem; color: var(--text-main);">{active}</div>
                <div class="metric-label" style="font-size: 0.6rem;">IN BEDS</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Live normalized score
    total_possible = st.session_state.get("max_possible_reward", 100)
    live_score = max(0.0, min(1.0, 
        st.session_state.cumulative_reward / total_possible
    ))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Task score", f"{live_score:.2f}")
    
    # Fatigue indicator
    if st.session_state.step_count >= 20:
        with col2:
            st.error("Fatigue active — rewards scaled to 85%", icon="😓")


def _render_end_screen(score: float, bd: Any, task_label: str, state: 'EDState') -> None:
    """End screen: Polished Clinical Report."""
    pct = int(score * 100)
    passed = bd.passed
    status_cls = "end-pass" if passed else "end-fail"
    status_txt = "PASS" if passed else "FAIL"
    
    # Dimension bars
    dim_bars = []
    for dim, raw in bd.dimensions.items():
        w = bd.weights.get(dim, 0.0)
        contrib = bd.weighted.get(dim, 0.0)
        pct_dim = int(raw * 100)
        bar_colour = "#10b981" if raw >= 0.7 else "#f59e0b" if raw >= 0.4 else "#ef4444"
        crit_ok = bd.criteria_results.get(dim)
        crit_label = (f' <span style="color:{bar_colour};">✓</span>' if crit_ok else f' <span style="color:#ef4444;">✗</span>') if crit_ok is not None else ""
        
        dim_bars.append(f"""
        <div style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size: 0.75rem; font-weight: 700; color: #6b7280; text-transform: uppercase; margin-bottom: 4px;">
                <span>{dim.replace("_"," ")}{crit_label}</span>
                <span>{pct_dim}%</span>
            </div>
            <div style="height: 6px; background: #f1f5f9; border-radius: 3px; overflow: hidden;">
                <div style="width: {pct_dim}%; height: 100%; background: {bar_colour}; border-radius: 3px;"></div>
            </div>
        </div>
        """)

    # Criteria
    crit_rows = []
    for crit, ok in bd.criteria_results.items():
        icon = '<span style="color:#10b981; font-weight:700;">✓ PASSED</span>' if ok else '<span style="color:#ef4444; font-weight:700;">✗ FAILED</span>'
        crit_rows.append(f'<div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.8rem; color: #475569;"><span>{crit.replace("_"," ")}</span>{icon}</div>')

    # Roster
    roster_rows = []
    for p in state.active_patients:
        color = "#ec4899" if p.assigned_tier == "ICU" else "#10b981" if p.assigned_tier == "General" else "#f59e0b"
        roster_rows.append(f'<div style="display:flex; justify-content:space-between; padding: 4px 0; font-size: 0.8rem; border-bottom: 1px solid #f8fafc;"><span style="font-weight:600;">{p.name}</span><span style="color:{color}; font-weight:700;">{p.assigned_tier}</span></div>')

    st.markdown(f"""
    <div class="report-screen">
        <div class="report-header">
            <div style="font-size: 0.7rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.1em;">Final Assessment Report</div>
            <div class="report-score {status_cls}">{pct}%</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #1f2937;">Scenario {status_txt}</div>
            <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px;">{task_label} • Completed in {state.step_count} Steps</div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 40px; margin-top: 30px;">
            <div>
                <div style="font-size: 0.65rem; font-weight: 800; color: #9ca3af; text-transform: uppercase; margin-bottom: 15px; letter-spacing: 0.05em;">Performance Breakdown</div>
                {"".join(dim_bars)}
            </div>
            <div>
                <div style="font-size: 0.65rem; font-weight: 800; color: #9ca3af; text-transform: uppercase; margin-bottom: 15px; letter-spacing: 0.05em;">Clinical Outcomes</div>
                {"".join(crit_rows)}
                <div style="margin-top: 20px; padding: 12px; background: #f8fafc; border-radius: 8px;">
                    <div style="font-size: 0.6rem; font-weight: 800; color: #9ca3af; text-transform: uppercase; margin-bottom: 8px;">Patient Roster</div>
                    {"".join(roster_rows)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_action_log() -> None:
    """Collapsible action history with beautiful styling."""
    log = st.session_state.action_log
    if not log:
        return
    with st.expander(f"Action History ({len(log)} steps)", expanded=False):
        html_rows = []
        for entry in reversed(log[-30:]):
            step = entry.get("step", 0)
            auto = "🤖 AUTO" if entry.get("auto") else "👤 HUMAN"
            pat = entry.get("patient", "")
            act = entry.get("action", "")
            rew = entry.get("reward", 0.0)
            
            # Formatting the reward badge
            rew_color = "#10b981" if rew > 0 else "#ef4444" if rew < 0 else "#9ca3af"
            rew_bg = "#dcfce7" if rew > 0 else "#fee2e2" if rew < 0 else "#f3f4f6"
            
            # Handling empty queue skips cleanly
            if pat == "—":
                html_rows.append(f'''
                <div style="padding: 6px 12px; margin-bottom: 4px; border-radius: 6px; background-color: #f9fafb; border: 1px dashed #d1d5db; display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; color: #9ca3af;">
                    <div><span style="font-weight:600;">Step {step}</span> &nbsp;|&nbsp; ⏱️ Waiting for next arrivals (Time Advanced)</div>
                    <div style="font-size:0.75rem; font-weight:700;">{auto}</div>
                </div>
                ''')
            else:
                act_color = "#3b82f6"
                if "ICU" in act: act_color = "#ec4899"
                elif "GENERAL" in act: act_color = "#10b981"
                elif "OBSERVATION" in act: act_color = "#f59e0b"
                
                html_rows.append(f'''
                <div style="padding: 8px 12px; margin-bottom: 6px; border-radius: 8px; background-color: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.02); display: flex; align-items: center; justify-content: space-between; font-size: 0.85rem; color: #374151;">
                    <div style="display:flex; align-items:center; gap: 10px;">
                        <div style="color: #6b7280; font-weight:600; width: 50px;">Step {step}</div>
                        <div style="font-weight: 600; min-width: 120px;">{pat}</div>
                        <div style="background-color:{act_color}1a; color:{act_color}; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{act}</div>
                    </div>
                    <div style="display:flex; align-items:center; gap: 12px;">
                        <span style="font-size:0.75rem; font-weight:700; color:#9ca3af;">{auto}</span>
                        <div style="background-color:{rew_bg}; color:{rew_color}; padding:2px 8px; border-radius:6px; font-weight:700; min-width: 45px; text-align:center;">{rew:+.1f}</div>
                    </div>
                </div>
                ''')
                
        st.markdown("".join(html_rows), unsafe_allow_html=True)


# ============================================================
# Main app
# ============================================================

def main() -> None:
    _init_session()

    # ── Header Navigation ────────────────────────────────────
    st.markdown(
        '''
        <div class="nav-header">
            <div class="nav-brand">
                <span style="font-size: 1.8rem;">🏥</span>
                <h1>ED Triage Console</h1>
            </div>
            <div style="font-size: 0.75rem; color: #9ca3af; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">
                System Health: <span style="color: #10b981;">Operational</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ── Top Controls ──────────────────────────────────────────
    with st.container():
        col1, col2, col3 = st.columns([1.5, 1, 2])
        with col1:
            task_label = st.selectbox(
                "Select Scenario",
                list(TASK_MAP.keys()),
                index=list(TASK_MAP.keys()).index(st.session_state.task_label),
                label_visibility="collapsed"
            )
        with col2:
            if st.button("Initialize Episode", use_container_width=True, type="primary"):
                _start_episode(task_label)
                st.rerun()
        with col3:
            api_key_input = st.text_input(
                "Gemini API Secret",
                value=st.session_state.get("api_key", ""),
                type="password",
                placeholder="Enter Gemini API Key...",
                label_visibility="collapsed"
            )
            if api_key_input != st.session_state.get("api_key", ""):
                st.session_state.api_key = api_key_input
                if api_key_input:
                    set_api_key(api_key_input)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # ── Guard: no env yet ────────────────────────────────────
    if st.session_state.env is None:
        st.markdown(
            """
            <div style="text-align:center;padding:80px 20px;color:#6b7280;">
                <div style="font-size:3rem;">🏥</div>
                <h2 style="color:#ec4899;">Welcome to ED Triage Sim</h2>
                <p>Select a task difficulty above and press <strong>Start / Reset</strong>
                to begin triage decisions.</p>
                <p style="font-size:0.80rem;">
                Optionally enter a Google Gemini API key to get LLM-powered triage hints and interactive AI chats.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    env: EDTriageEnv = st.session_state.env
    state: EDState = env.state()

    # ── END SCREEN ────────────────────────────────────────────
    if st.session_state.episode_done and st.session_state.final_score is not None:
        _render_end_screen(
            st.session_state.final_score,
            st.session_state.final_breakdown,
            st.session_state.task_label,
            state
        )
        _render_action_log()
        if st.button("Play Again", key="btn_replay"):
            _start_episode(st.session_state.task_label)
            st.rerun()
        return

    # ── THREE-COLUMN LAYOUT ───────────────────────────────────
    left, centre, right = st.columns([1.6, 2.8, 1.8], gap="medium")

    with left:
        _render_floor(state)
        _render_active_strip(state)

    with centre:
        _render_patient_queue(state)

    with right:
        _render_action_panel(state)

    # ── Metric bar ────────────────────────────────────────────
    _render_metric_bar(state)

    # ── Action log (collapsible) ──────────────────────────────
    _render_action_log()

    # ── Task description ──────────────────────────────────────
    task_name, task_cfg = TASK_MAP[st.session_state.task_label]
    with st.expander("Task Details", expanded=False):
        # Humanise the data for the UI
        criteria_strs = []
        for k, v in task_cfg.success_criteria.items():
            name = k.replace('_', ' ').title()
            criteria_strs.append(f"{name} must be ≥ {v*100:.0f}%")
        
        weights_strs = []
        for k, v in task_cfg.score_weights.items():
            name = k.replace('_', ' ').title()
            weights_strs.append(f"{v*100:.0f}% {name}")
            
        fatigue_step = task_cfg.fatigue_from_step
        fatigue_str = "Disabled for this scenario" if fatigue_step > 1000 else f"Begins at step {fatigue_step}"
        
        st.markdown(
            f"**{task_cfg.name.title()}** — {task_cfg.description}  \n\n"
            f"**✅ Minimum to Pass:** {', '.join(criteria_strs)}  \n"
            f"**⚖️ Scoring Emphasis:** {', '.join(weights_strs)}  \n"
            f"**⏳ Time Limit:** {task_cfg.env_kwargs.get('max_steps')} sim-steps  \n"
            f"**🥱 Staff Fatigue:** {fatigue_str}"
        )


if __name__ == "__main__":
    main()
