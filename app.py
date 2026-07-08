"""
ICU Twin — Early Deterioration Dashboard
========================================
Run:  streamlit run app.py

Expects a `dashboard_bundle/` folder (created by the export cell in the notebook)
containing: model.cbm, config.json, all_bins.parquet, scored_patients.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Config & loading
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ICU Twin", layout="wide", page_icon="🫀")

BUNDLE = Path("dashboard_bundle")

# clinical palette — calm, high-contrast risk states
C_CRIT   = "#C1121F"   # deep red   — alert
C_WATCH  = "#E8A33D"   # amber      — watch
C_STABLE = "#2A9D8F"   # teal       — stable
C_INK    = "#1D2733"   # near-black text
C_MUTE   = "#8896A6"   # muted grey
C_PANEL  = "#F5F7FA"   # panel bg

def hex_to_rgba(hex_color, alpha=0.08):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.markdown(f"""
<style>
  .stApp {{ background: #FFFFFF; }}
  h1, h2, h3 {{ color: {C_INK}; letter-spacing: -0.01em; }}
  .risk-pill {{ padding: 2px 10px; border-radius: 999px; color: white;
               font-weight: 600; font-size: 0.85rem; }}
  .reason {{ padding: 6px 12px; margin: 4px 0; border-left: 3px solid {C_MUTE};
            background: {C_PANEL}; border-radius: 4px; font-size: 0.9rem; }}
  .reason-up   {{ border-left-color: {C_CRIT}; }}
  .reason-down {{ border-left-color: {C_STABLE}; }}
  .metric-big  {{ font-size: 2.4rem; font-weight: 700; line-height: 1; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model(str(BUNDLE / "model.cbm"))
    return m

@st.cache_data
def load_data():
    cfg = json.load(open(BUNDLE / "config.json"))
    allb = pd.read_parquet(BUNDLE / "all_bins.parquet")
    scored = pd.read_parquet(BUNDLE / "scored_patients.parquet")
    return cfg, allb, scored

@st.cache_resource
def load_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


try:
    model = load_model()
    cfg, all_bins, scored = load_data()
    explainer = load_explainer(model)
except Exception as e:
    st.error(f"Could not load the dashboard bundle from `{BUNDLE}/`. "
             f"Run the export cell in the notebook first.\n\n{e}")
    st.stop()

FEATURES  = cfg["features"]
THRESH    = cfg["threshold"]
VITALS    = [v for v in cfg["vitals"] if v in all_bins.columns]

# deterioration thresholds (for the what-if guard + display)
THRESH_RULES = {"spo2": ("<", 90), "map": ("<", 65),
                "hr": (">", 130), "rr": (">", 30), "temp": (">", 39)}
INTERVENTIONS = [   # (condition, vital, delta, up_is_better, label)
    (lambda r: r["spo2"] < 94,  "spo2", +5, True,  "O₂ therapy"),
    (lambda r: r["map"]  < 70,  "map",  +8, True,  "Vasopressors"),
    (lambda r: r["hr"]   > 110, "hr",  -15, False, "Rate control"),
    (lambda r: r["rr"]   > 24,  "rr",   -6, False, "Resp support"),
]
CLIP = {"spo2": (50,100), "map": (20,200), "hr": (20,250), "rr": (3,60), "temp": (30,43)}

PRETTY = {"hr":"Heart rate","spo2":"SpO₂","rr":"Resp rate","map":"MAP","temp":"Temp",
          "gcs":"GCS","news_score":"NEWS score","n_breaches":"Threshold breaches",
          "shock_index":"Shock index","breach_cum":"Cumulative abnormal time",
          "pulse_pressure_proxy":"Pulse-pressure proxy","age":"Age"}


def risk_level(p):
    if p >= THRESH:            return "CRITICAL", C_CRIT
    if p >= THRESH * 0.6:      return "WATCH",    C_WATCH
    return "STABLE", C_STABLE

def score_row(row):
    X = pd.DataFrame([row[FEATURES]])
    return model.predict_proba(X)[:, 1][0]

def apply_interventions(row):
    r = row.copy(); applied = []
    for cond, v, dv, up, label in INTERVENTIONS:
        if v in row and pd.notna(row[v]) and cond(row):
            r[v] = np.clip(row[v] + dv, *CLIP[v])
            applied.append(label)
            for suf, val in [("_d1", dv), ("_rollmean", row.get(f"{v}_rollmean", row[v]) + dv/3)]:
                if f"{v}{suf}" in r: r[f"{v}{suf}"] = val
            if f"{v}_slope" in r:
                r[f"{v}_slope"] = (abs(row.get(f"{v}_slope",0))+1) if up else -(abs(row.get(f"{v}_slope",0))+1)
    return r, applied

def humanize(fname, value, contrib):
    base = fname.split("_")[0]
    pretty = PRETTY.get(fname, PRETTY.get(base, fname))
    up = contrib > 0
    if fname.endswith("_slope"):
        txt = f"{pretty} {'rising' if (value or 0)>0 else 'falling'}"
    elif fname.endswith("_missing"):
        txt = f"{pretty} not recently measured"
    elif fname.endswith("_rollstd"):
        txt = f"{pretty} unstable"
    elif pd.isna(value):
        txt = f"{pretty} missing"
    else:
        txt = f"{pretty} = {value:.0f}"
    return txt, up


# ──────────────────────────────────────────────────────────────────────
# Sidebar — patient list (the ICU overview / alert queue)
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ICU — patient queue")
sb = scored.copy().sort_values("risk", ascending=False)
n_crit = (sb["risk"] >= THRESH).sum()
st.sidebar.markdown(f"**{len(sb)}** patients · "
                    f"<span class='risk-pill' style='background:{C_CRIT}'>{n_crit} critical</span>",
                    unsafe_allow_html=True)

# build readable labels for the queue
def queue_label(r):
    lvl, _ = risk_level(r["risk"])
    dot = {"CRITICAL":"🔴","WATCH":"🟡","STABLE":"🟢"}[lvl]
    return f"{dot}  Stay {int(r['stay_id'])} · {r['risk']:.0%}"

sb["label"] = sb.apply(queue_label, axis=1)
choice = st.sidebar.radio("Select patient", sb["label"].tolist(), label_visibility="collapsed")
sel_stay = int(choice.split("Stay ")[1].split(" ·")[0])
patient = sb[sb.stay_id == sel_stay].iloc[0]


# ──────────────────────────────────────────────────────────────────────
# Main — patient twin
# ──────────────────────────────────────────────────────────────────────
lvl, col = risk_level(patient["risk"])
left, right = st.columns([1, 1.1])

with left:
    st.markdown(f"## Stay {sel_stay}")
    meta = f"{int(patient['age'])}y · {patient.get('gender','?')}"
    if "first_careunit" in patient: meta += f" · {patient['first_careunit']}"
    st.caption(meta)
    st.markdown(
        f"<div class='metric-big' style='color:{col}'>{patient['risk']:.0%}</div>"
        f"<span class='risk-pill' style='background:{col}'>{lvl}</span> "
        f"<span style='color:{C_MUTE}'>deterioration risk · next 3h</span>",
        unsafe_allow_html=True)
    st.caption(f"Alert threshold {THRESH:.0%} · model CV PR-AUC "
               f"{cfg.get('cv_pr_auc',0):.2f} ± {cfg.get('cv_pr_std',0):.2f}")

    # current vitals with breach flags
    st.markdown("#### Current vitals")
    vc = st.columns(len(VITALS))
    for i, v in enumerate(VITALS):
        val = patient.get(v, np.nan)
        breached = False
        if v in THRESH_RULES and pd.notna(val):
            op, t = THRESH_RULES[v]
            breached = (val > t) if op == ">" else (val < t)
        vc[i].metric(PRETTY.get(v, v).replace("SpO₂","SpO2"),
                     "—" if pd.isna(val) else f"{val:.0f}",
                     delta="breach" if breached else None, delta_color="inverse")

with right:
    # risk trend over the stay
    hist = all_bins[all_bins.stay_id == sel_stay].sort_values("bin")
    if len(hist) > 1:
        st.markdown("#### Risk trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["bin"], y=hist["risk"], mode="lines",
                      line=dict(color=col, width=3), fill="tozeroy",
                      fillcolor=hex_to_rgba(col, 0.08),
                      name="risk"))
        fig.add_hline(y=THRESH, line_dash="dash", line_color=C_MUTE,
                      annotation_text="alert", annotation_position="right")
        fig.update_layout(height=230, margin=dict(l=0,r=0,t=6,b=0),
                          yaxis=dict(range=[0,1], tickformat=".0%"),
                          plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────
# Why + What-if, side by side
# ──────────────────────────────────────────────────────────────────────
why_col, whatif_col = st.columns(2)

with why_col:
    st.markdown("### Why this risk")
    sv = explainer.shap_values(pd.DataFrame([patient[FEATURES]]))[0]
    order = np.argsort(-np.abs(sv))[:6]
    for i in order:
        txt, up = humanize(FEATURES[i], patient[FEATURES[i]], sv[i])
        cls = "reason-up" if up else "reason-down"
        arrow = "▲" if up else "▼"
        st.markdown(f"<div class='reason {cls}'>{arrow} {txt}</div>", unsafe_allow_html=True)
    st.caption("SHAP feature contributions · shows the model's reasoning, "
               "which is correlational, not a causal clinical prediction.")

with whatif_col:
    st.markdown("### What-if intervention")
    r_after, applied = apply_interventions(patient)
    if not applied:
        st.info("No standard intervention is clinically indicated for this "
                "patient's current vitals — nothing abnormal in a treatable direction.")
    else:
        risk_after = score_row(r_after)
        lvl_a, col_a = risk_level(risk_after)
        d1, d2, d3 = st.columns([1, 0.3, 1])
        d1.markdown(f"**Before**<br><span class='metric-big' style='color:{col}'>"
                    f"{patient['risk']:.0%}</span>", unsafe_allow_html=True)
        d2.markdown("<div style='font-size:2rem;color:#8896A6;text-align:center'>→</div>",
                    unsafe_allow_html=True)
        d3.markdown(f"**After**<br><span class='metric-big' style='color:{col_a}'>"
                    f"{risk_after:.0%}</span>", unsafe_allow_html=True)
        st.markdown(f"**Applied:** {', '.join(applied)}")
        if patient["risk"] >= THRESH > risk_after:
            st.success("Intervention clears the alert threshold.")
        # which drivers the intervention resolved
        sa = explainer.shap_values(pd.DataFrame([r_after[FEATURES]]))[0]
        delta = sv - sa
        seen = set(); shown = 0
        for i in np.argsort(-delta):
            base = FEATURES[i].split("_")[0]
            if delta[i] > 0.02 and sv[i] > 0 and base not in seen:
                seen.add(base); shown += 1
                txt, _ = humanize(FEATURES[i], patient[FEATURES[i]], sv[i])
                st.markdown(f"<div class='reason reason-down'>✓ resolved: {txt}</div>",
                            unsafe_allow_html=True)
            if shown >= 3: break
        st.caption("Simulates the model's response to vital changes — "
                   "an estimate of the model's belief, not a validated outcome.")

st.divider()
st.caption("ICU Twin · research/education demo on MIMIC-IV demo data · not a medical device.")
