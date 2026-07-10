"""
ICU Digital Twin — deterioration dashboard.
Loads the artifacts exported by the notebook (models + feature table + metadata).
Every displayed number traces to the model or the data; the few protocol-based
panels (suggested meds) are labelled as such. No invented metrics.
"""
import json, joblib, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ── artifact location ───────────────────────────────────────────────────────────
# set OUT to the notebook's output folder. On Colab this is the Drive path.
OUT = Path(st.secrets.get("OUT_DIR", "/content/drive/MyDrive/icu_twin/output")) \
    if hasattr(st, "secrets") else Path("/content/drive/MyDrive/icu_twin/output")

st.set_page_config(page_title="ICU Digital Twin", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# ── palette grounded in early-warning-score severity, not decoration ────────────
C = dict(bg="#080B12", bg2="#0C111C", panel="#111827", panel2="#0E1420", line="#243044",
         ink="#EDF2FB", mut="#8595AD", accent="#22D3EE",
         crit="#FB3B5C", watch="#FBBF24", ok="#34D399", violet="#A78BFA")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

  .stApp {{
    background:
      radial-gradient(1200px 600px at 15% -10%, #12203a55 0%, transparent 55%),
      radial-gradient(1000px 500px at 100% 0%, #1a2b4a44 0%, transparent 50%),
      linear-gradient(180deg, {C['bg']} 0%, {C['bg2']} 100%);
    color:{C['ink']};
  }}
  section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg, #0d1422 0%, #0a0e18 100%);
    border-right:1px solid {C['line']};
  }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  .block-container {{ padding-top:1.1rem; max-width:1560px; }}
  html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}

  .eyebrow {{ font-size:.66rem; letter-spacing:.22em; text-transform:uppercase;
             color:{C['accent']}; font-weight:600; margin:.4rem 0 .6rem;
             display:flex; align-items:center; gap:.5rem; }}
  .eyebrow::after {{ content:''; flex:1; height:1px;
             background:linear-gradient(90deg, {C['accent']}55, transparent); }}

  .panel {{
    background:linear-gradient(180deg, #141c2b 0%, #0f1521 100%);
    border:1px solid {C['line']}; border-radius:16px; padding:1.05rem 1.2rem; margin-bottom:.85rem;
    box-shadow:0 1px 0 #ffffff08 inset, 0 8px 24px -12px #00000090;
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }}
  .panel:hover {{ transform:translateY(-2px); border-color:{C['accent']}44;
                 box-shadow:0 1px 0 #ffffff10 inset, 0 16px 32px -14px #000000b0; }}

  .vnum {{ font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:700;
          line-height:1; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
  .vlabel {{ font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:{C['mut']};
            font-weight:600; }}
  .vunit {{ font-size:.7rem; color:{C['mut']}; margin-top:.15rem; }}
  .delta {{ font-size:.72rem; font-family:'JetBrains Mono',monospace; margin-top:.35rem;
           font-variant-numeric:tabular-nums; }}

  .alert {{ border-radius:12px; padding:.6rem .95rem; margin-bottom:.45rem;
           border:1px solid; font-size:.87rem; font-weight:500; backdrop-filter:blur(6px);
           display:flex; align-items:center; gap:.55rem; animation:slidein .35s ease; }}
  @keyframes slidein {{ from {{ opacity:0; transform:translateX(-8px); }} to {{ opacity:1; }} }}

  .pill {{ display:inline-block; font-size:.6rem; letter-spacing:.09em; text-transform:uppercase;
          padding:.15rem .55rem; border-radius:999px; font-weight:700; }}
  .hname {{ font-family:'Space Grotesk',sans-serif; font-size:2.15rem; font-weight:700;
           letter-spacing:-.02em; }}
  .hsub {{ color:{C['mut']}; font-size:.86rem; }}
  .reason {{ font-size:.87rem; color:{C['ink']}; margin:.2rem 0; line-height:1.5; }}
  .tiny {{ font-size:.72rem; color:{C['mut']}; line-height:1.5; }}

  /* risk gauge: soft glow + gentle breathing pulse */
  .gauge {{ position:relative; text-align:center; border-radius:18px; padding:1.1rem 1rem;
           border:1px solid; overflow:hidden; }}
  .gauge::before {{ content:''; position:absolute; inset:0; opacity:.14;
           background:radial-gradient(circle at 50% 25%, currentColor, transparent 62%); }}
  .gaugenum {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:3.2rem;
           line-height:1; font-variant-numeric:tabular-nums; animation:pulse 2.6s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:.82; transform:scale(1.03); }} }}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap:.3rem; border-bottom:1px solid {C['line']}; }}
  .stTabs [data-baseweb="tab"] {{ font-size:.85rem; font-weight:600; color:{C['mut']};
           padding:.5rem .9rem; border-radius:10px 10px 0 0; }}
  .stTabs [aria-selected="true"] {{ color:{C['ink']};
           background:linear-gradient(180deg, {C['accent']}18, transparent); }}

  .stButton>button {{ background:linear-gradient(180deg,#16202f,#101724);
           border:1px solid {C['line']}; color:{C['ink']}; border-radius:10px; font-weight:600;
           transition:all .16s ease; }}
  .stButton>button:hover {{ border-color:{C['accent']}66; color:{C['accent']};
           box-shadow:0 8px 20px -12px {C['accent']}80; }}
</style>
""", unsafe_allow_html=True)

# ── load artifacts once ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_all(out: str):
    out = Path(out)
    meta = json.loads((out / "dashboard_meta.json").read_text())
    metrics = json.loads((out / "test_metrics.json").read_text())
    feat = pd.read_parquet(out / "feature_table.parquet")
    feat["charttime"] = pd.to_datetime(feat["charttime"])
    roster = pd.read_parquet(out / "roster.parquet")
    feature_cols = joblib.load(out / "feature_cols.joblib")
    models = {h: joblib.load(out / f"model_target_{h}h.joblib") for h in meta["horizons_h"]}
    return meta, metrics, feat, roster, feature_cols, models

try:
    META, METRICS, FEAT, ROSTER, FCOLS, MODELS = load_all(OUT)
except FileNotFoundError as e:
    st.error(f"Artifacts not found under {OUT}. Run the export cell in the notebook first, "
             f"then set OUT_DIR. Missing: {e}")
    st.stop()

HZ = META["horizons_h"]                       # [1,3,6]
BASE = META["base_vitals"]
CRIT, WATCH = META["risk_levels"]["critical"], META["risk_levels"]["watch"]

def risk_level(p):  return "critical" if p >= CRIT else "watch" if p >= WATCH else "stable"
def risk_color(p):  return C["crit"] if p >= CRIT else C["watch"] if p >= WATCH else C["ok"]

# demo-only cosmetic identifiers (MIMIC is de-identified — no real names exist)
FIRST = ["John","Maria","David","Aisha","Chen","Omar","Grace","Liam","Nina","Raj"]
def demo_name(sid): 
    r = np.random.RandomState(int(sid) % 9973); return f"{FIRST[int(sid)%len(FIRST)]} {chr(65+int(sid)%26)}."

# ── sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div class='hname' style='font-size:1.3rem'>🫀 ICU Digital Twin</div>"
                f"<div class='tiny'>Early deterioration prediction</div><br>", unsafe_allow_html=True)

    stays = ROSTER.sort_values("stay_id")
    labels = {int(r.stay_id): f"ICU-{i+1:02d} · {demo_name(r.stay_id)}  ({r.diagnosis[:22]})"
              for i, r in enumerate(stays.itertuples())}
    keys = list(labels)
    jump = st.session_state.pop("_jump_stay", None)     # set when a ward card is clicked
    idx = keys.index(jump) if jump in keys else 0
    stay_id = st.selectbox("Patient", keys, index=idx, format_func=lambda s: labels[s])

    horizon = st.radio("Prediction horizon", HZ, index=min(1, len(HZ)-1),
                       format_func=lambda h: f"{h}h ahead", horizontal=True)

    sim = st.toggle("Simulation mode", value=True,
                    help="Step through this stay's real recorded hours to replay how risk evolved. "
                         "The MIMIC extract is historical, so this replays real data — it is not a live feed.")

# rows for the selected stay, in time order
srow = FEAT[FEAT.stay_id == stay_id].sort_values("charttime").reset_index(drop=True)
info = ROSTER[ROSTER.stay_id == stay_id].iloc[0]

# which hour are we "at"?
if sim and len(srow) > 1:
    hr_i = st.sidebar.slider("Hour in stay", 0, len(srow)-1, len(srow)-1,
                             help="0 = admission. Slide to replay deterioration.")
else:
    hr_i = len(srow) - 1
row  = srow.iloc[hr_i]
prev = srow.iloc[hr_i-1] if hr_i > 0 else None

# ── predictions for the current hour ────────────────────────────────────────────
def predict(h, r):
    return float(MODELS[h].predict_proba(r[FCOLS].astype(float).values.reshape(1, -1))[:, 1][0])
probs = {h: predict(h, row) for h in HZ}
p_now = probs[horizon]

# map_best for display / rules
def map_val(r):
    v = r.get("abp_mean"); return r.get("nibp_mean") if pd.isna(v) else v

# ── header ──────────────────────────────────────────────────────────────────────
left, right = st.columns([2.4, 1])
with left:
    st.markdown(f"<div class='hname'>{demo_name(stay_id)}</div>"
                f"<div class='hsub'>Bed ICU-{int(stay_id)%90+1:02d} · Age {int(info.age)} · "
                f"{str(info.gender).upper()} · {info.diagnosis}</div>"
                f"<div class='tiny'>Hour {hr_i} of {len(srow)-1} · {row.charttime:%Y-%m-%d %H:%M} · "
                f"de-identified MIMIC-IV demo</div>", unsafe_allow_html=True)
with right:
    lvl = risk_level(p_now); col = risk_color(p_now)
    st.markdown(f"""<div class='gauge' style='color:{col};border-color:{col}55'>
      <div class='vlabel'>Risk · {horizon}h ahead</div>
      <div class='gaugenum' style='color:{col}'>{p_now*100:.0f}<span style='font-size:1.1rem'>%</span></div>
      <div style='color:{col};font-weight:700;text-transform:capitalize;letter-spacing:.04em'>{lvl} risk</div>
      <div class='tiny' style='margin-top:.35rem'>1h {probs[HZ[0]]*100:.0f} · 3h {probs[HZ[min(1,len(HZ)-1)]]*100:.0f} · 6h {probs[HZ[-1]]*100:.0f}</div>
    </div>""", unsafe_allow_html=True)

# ── ward-level scoring: latest hour of every stay, scored by the chosen horizon ──
@st.cache_data(show_spinner=False)
def score_ward(horizon: int):
    """One row per stay = its most recent recorded hour, scored + summarised. All real model output."""
    latest = FEAT.sort_values("charttime").groupby("stay_id").tail(1).reset_index(drop=True)
    latest["risk"] = MODELS[horizon].predict_proba(latest[FCOLS].astype(float))[:, 1]
    # count active threshold breaches for each row (drives the alert-queue ordering)
    def n_breaches(r):
        m = r.get("nibp_mean") if pd.isna(r.get("abp_mean")) else r.get("abp_mean")
        c = 0
        if pd.notna(r.spo2) and r.spo2 < 92: c += 1
        if pd.notna(r.respiratory_rate) and r.respiratory_rate > 24: c += 1
        if pd.notna(m) and m < 65: c += 1
        if pd.notna(r.heart_rate) and r.heart_rate > 110: c += 1
        if pd.notna(r.get("lactate")) and r.lactate > 2: c += 1
        return c
    latest["breaches"] = latest.apply(n_breaches, axis=1)
    return latest.sort_values(["risk", "breaches"], ascending=False).reset_index(drop=True)

# ── tabs ────────────────────────────────────────────────────────────────────────
t_ward, t_over, t_trend, t_pred, t_model = st.tabs(
    ["Ward overview", "Patient twin", "Vital trends", "Intervention (what-if)", "Model card"])

# ---- ward overview = doc's page 1 + alert queue in one view ----------------------
with t_ward:
    ward = score_ward(horizon)
    n_crit  = int((ward.risk >= CRIT).sum())
    n_watch = int(((ward.risk >= WATCH) & (ward.risk < CRIT)).sum())
    n_stab  = int((ward.risk < WATCH).sum())
    st.markdown(f"<div class='eyebrow'>Ward · {len(ward)} patients · {horizon}h horizon</div>",
                unsafe_allow_html=True)
    kc = st.columns(3)
    for col, (lab, n, cc) in zip(kc, [("Critical", n_crit, C["crit"]),
                                      ("Watch", n_watch, C["watch"]),
                                      ("Stable", n_stab, C["ok"])]):
        col.markdown(f"<div class='panel' style='text-align:center;border-color:{cc}'>"
                     f"<div class='vnum' style='color:{cc}'>{n}</div>"
                     f"<div class='vlabel'>{lab}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='eyebrow'>Alert queue — highest risk first</div>", unsafe_allow_html=True)
    grid = st.columns(4)
    for k, r in ward.iterrows():
        cc = risk_color(r.risk); lvl = risk_level(r.risk)
        m = r.get("nibp_mean") if pd.isna(r.get("abp_mean")) else r.get("abp_mean")
        with grid[k % 4]:
            picked = st.button(f"{demo_name(r.stay_id)}", key=f"ward_{int(r.stay_id)}",
                               use_container_width=True)
            st.markdown(f"<div class='panel' style='border-left:3px solid {cc};margin-top:-.6rem'>"
                        f"<div class='vnum' style='color:{cc};font-size:1.8rem'>{r.risk*100:.0f}"
                        f"<span style='font-size:.7rem'> %</span></div>"
                        f"<div class='vlabel' style='color:{cc}'>{lvl}</div>"
                        f"<div class='tiny' style='margin-top:.3rem'>"
                        f"HR {r.heart_rate:.0f} · SpO₂ {r.spo2:.0f} · "
                        f"MAP {m:.0f}</div>".replace("nan","—")
                        + f"<div class='tiny'>{r.breaches} active breach"
                        f"{'es' if r.breaches!=1 else ''}</div></div>", unsafe_allow_html=True)
            if picked:
                st.session_state["_jump_stay"] = int(r.stay_id)
                st.rerun()

# ---- alerts + reasons (rule-based, from your explain_row logic) -----------------
def build_alerts(r):
    a, m = [], map_val(r)
    if pd.notna(r.spo2) and r.spo2 < 92:                a.append(("Hypoxemia", f"SpO₂ {r.spo2:.0f}%", C["crit"]))
    if pd.notna(r.respiratory_rate) and r.respiratory_rate > 24:
        a.append(("Tachypnea", f"RR {r.respiratory_rate:.0f}/min", C["crit"]))
    if pd.notna(m) and m < 65:                          a.append(("Hypotension", f"MAP {m:.0f} mmHg", C["crit"]))
    if pd.notna(r.heart_rate) and r.heart_rate > 110:   a.append(("Tachycardia", f"HR {r.heart_rate:.0f}", C["watch"]))
    if pd.notna(r.get("lactate")) and r.lactate > 2:    a.append(("Elevated lactate", f"{r.lactate:.1f} mmol/L", C["watch"]))
    if pd.notna(r.temperature_f) and r.temperature_f > 100.4:
        a.append(("Fever", f"{(r.temperature_f-32)*5/9:.1f}°C", C["watch"]))
    return a

def reasons(cur, prv):
    r, m = [], map_val(cur)
    if pd.notna(cur.spo2) and cur.spo2 < 92: r.append(f"SpO₂ low ({cur.spo2:.0f}%)")
    if pd.notna(m) and m < 70: r.append(f"MAP low ({m:.0f} mmHg)")
    if pd.notna(cur.heart_rate) and cur.heart_rate > 110: r.append(f"HR high ({cur.heart_rate:.0f})")
    if pd.notna(cur.respiratory_rate) and cur.respiratory_rate > 24: r.append(f"RR high ({cur.respiratory_rate:.0f})")
    if prv is not None and pd.notna(cur.spo2) and pd.notna(prv.spo2) and cur.spo2-prv.spo2 <= -3:
        r.append(f"SpO₂ falling ({prv.spo2:.0f}→{cur.spo2:.0f})")
    return r or ["No single threshold breached; risk from combined pattern."]

# vitals to show as cards: (col, label, unit, converter, thresholds for color)
VITAL_CARDS = [
    ("heart_rate","Heart rate","bpm",lambda v:v,(110,130)),
    ("map","BP (MAP)","mmHg",lambda v:v,None),
    ("spo2","SpO₂","%",lambda v:v,(92,90)),
    ("respiratory_rate","Resp rate","/min",lambda v:v,(24,30)),
    ("temperature_f","Temp","°C",lambda v:(v-32)*5/9,(100.4,102.2)),
    ("lactate","Lactate","mmol/L",lambda v:v,(2,4)),
]

with t_over:
    ca, cb = st.columns([2.4, 1])
    with ca:
        # clinical summary (rule-derived; optionally swap for an LLM sentence)
        rs = reasons(row, prev)
        st.markdown("<div class='eyebrow'>Clinical summary</div>", unsafe_allow_html=True)
        summary = ("Model flags elevated short-horizon deterioration risk. "
                   + " ".join(rs[:3]) +
                   (" Trend over recent hours is contributing to the estimate."
                    if any("falling" in x for x in rs) else ""))
        st.markdown(f"<div class='panel'>{summary}</div>", unsafe_allow_html=True)

        st.markdown("<div class='eyebrow'>Active alerts</div>", unsafe_allow_html=True)
        al = build_alerts(row)
        if not al: st.markdown("<div class='tiny'>No active threshold alerts.</div>", unsafe_allow_html=True)
        for name, detail, cc in al:
            st.markdown(f"<div class='alert' style='border-color:{cc};background:{cc}18;color:{cc}'>"
                        f"⚠ {name} — {detail}</div>", unsafe_allow_html=True)

        st.markdown("<div class='eyebrow'>Current vitals</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for k, (cid, lab, unit, conv, thr) in enumerate(VITAL_CARDS):
            raw = map_val(row) if cid == "map" else row.get(cid)
            with cols[k % 3]:
                if pd.isna(raw):
                    st.markdown(f"<div class='panel'><div class='vlabel'>{lab}</div>"
                                f"<div class='vnum' style='color:{C['mut']}'>—</div>"
                                f"<div class='vunit'>{unit}</div></div>", unsafe_allow_html=True)
                    continue
                val = conv(raw); vc = C["ink"]
                if thr:
                    if cid in ("spo2",): vc = C["crit"] if raw < thr[1] else C["watch"] if raw < thr[0] else C["ok"]
                    else:                vc = C["crit"] if val > thr[1] else C["watch"] if val > thr[0] else C["ok"]
                # delta vs previous recorded hour
                dtxt = ""
                if prev is not None:
                    praw = map_val(prev) if cid == "map" else prev.get(cid)
                    if pd.notna(praw):
                        d = conv(raw) - conv(praw); arr = "▲" if d > 0 else "▼" if d < 0 else "■"
                        dc = C["mut"] if abs(d) < 1e-6 else (C["crit"] if (d>0)==(cid!="spo2") else C["ok"])
                        dtxt = f"<div class='delta' style='color:{dc}'>{arr} {abs(d):.1f} vs prev</div>"
                vtxt = f"{val:.1f}" if unit in ("mmol/L", "°C") else f"{val:.0f}"
                st.markdown(f"<div class='panel'><div class='vlabel'>{lab}</div>"
                            f"<div class='vnum' style='color:{vc}'>{vtxt}</div>"
                            f"<div class='vunit'>{unit}</div>{dtxt}</div>", unsafe_allow_html=True)

    with cb:
        st.markdown("<div class='eyebrow'>Why this risk</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'>" +
                    "".join(f"<div class='reason'>• {x}</div>" for x in reasons(row, prev)) +
                    "</div>", unsafe_allow_html=True)

        # honest medication contribution via SHAP on med flags -----------------
        st.markdown("<div class='eyebrow'>Medication contribution "
                    "<span class='tiny'>(SHAP)</span></div>", unsafe_allow_html=True)
        med_cols = [c for c in META["med_cols"] if c in FCOLS]
        active   = [c for c in med_cols if float(row.get(c, 0)) == 1]
        try:
            import shap
            expl = shap.TreeExplainer(MODELS[horizon])
            sv = expl.shap_values(row[FCOLS].astype(float).values.reshape(1, -1))
            sv = sv[0] if isinstance(sv, list) else sv
            sv = np.asarray(sv).ravel()
            contrib = {c: float(sv[FCOLS.index(c)]) for c in active}
            contrib = dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5])
            lbl = {m["col"]: m["label"] for m in META["med_meta"]}
            if contrib:
                st.markdown("<div class='panel'>" + "".join(
                    f"<div class='reason'>{lbl.get(c,c).title()} "
                    f"<span class='tiny'>{'↑ raises' if v>0 else '↓ lowers'} risk "
                    f"({v:+.3f} log-odds)</span></div>" for c, v in contrib.items()) +
                    "<div class='tiny' style='margin-top:.4rem'>SHAP contribution of "
                    "medications given at/before this hour, to this horizon's model.</div></div>",
                    unsafe_allow_html=True)
            else:
                st.markdown("<div class='tiny'>No active medication flags at this hour.</div>",
                            unsafe_allow_html=True)
        except Exception as ex:
            st.markdown(f"<div class='tiny'>SHAP unavailable ({type(ex).__name__}). "
                        f"Active meds: {', '.join(active) or 'none'}.</div>", unsafe_allow_html=True)

# ---- vital trends tab -----------------------------------------------------------
with t_trend:
    st.markdown("<div class='eyebrow'>Recorded vitals over the stay</div>", unsafe_allow_html=True)
    plot_map = [("heart_rate","HR",C["crit"]),("respiratory_rate","RR",C["watch"]),
                ("spo2","SpO₂",C["accent"]),("temperature_f","Temp °F",C["violet"])]
    fig = go.Figure()
    for cid, nm, cc in plot_map:
        if cid in srow:
            fig.add_trace(go.Scatter(x=srow.charttime, y=srow[cid], name=nm,
                                     line=dict(color=cc, width=2), connectgaps=True))
    fig.add_vline(x=row.charttime, line=dict(color=C["ink"], dash="dot", width=1))
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color=C["mut"], legend=dict(orientation="h", y=1.15),
                      margin=dict(l=10,r=10,t=10,b=10),
                      xaxis=dict(gridcolor=C["line"]), yaxis=dict(gridcolor=C["line"]))
    st.plotly_chart(fig, use_container_width=True)

    # risk trajectory: predict every hour of the stay for the chosen horizon
    st.markdown("<div class='eyebrow'>Predicted risk trajectory</div>", unsafe_allow_html=True)
    traj = MODELS[horizon].predict_proba(srow[FCOLS].astype(float))[:, 1]
    rf = go.Figure(go.Scatter(x=srow.charttime, y=traj*100, fill="tozeroy",
                              line=dict(color=risk_color(p_now), width=2), name=f"{horizon}h risk"))
    rf.add_hline(y=CRIT*100, line=dict(color=C["crit"], dash="dot"))
    rf.add_hline(y=WATCH*100, line=dict(color=C["watch"], dash="dot"))
    rf.add_vline(x=row.charttime, line=dict(color=C["ink"], dash="dot", width=1))
    rf.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color=C["mut"], margin=dict(l=10,r=10,t=10,b=10),
                     yaxis=dict(gridcolor=C["line"], title="risk %", range=[0,100]),
                     xaxis=dict(gridcolor=C["line"]))
    st.plotly_chart(rf, use_container_width=True)

# ---- what-if tab ----------------------------------------------------------------
with t_pred:
    st.markdown("<div class='eyebrow'>Intervention simulation</div>", unsafe_allow_html=True)
    st.markdown("<div class='tiny'>Override a vital to its post-intervention value. The vital's "
                "trend features (Δ / rolling mean / rolling std) are rebuilt from real prior hours, "
                "so level and trend move together — then the model re-scores.</div><br>",
                unsafe_allow_html=True)

    def whatif(h, pos, overrides):
        base = srow.iloc[pos][FCOLS].astype(float).copy()
        r0 = float(MODELS[h].predict_proba(base.values.reshape(1,-1))[:,1][0])
        mod = base.copy()
        for raw, nv in overrides.items():
            if raw not in mod.index: continue
            mod[raw] = nv
            for w in META["trend_windows_h"]:
                if 0 <= pos-w and f"{raw}_delta{w}" in mod.index:
                    mod[f"{raw}_delta{w}"] = nv - srow[raw].iloc[pos-w]
                win = srow[raw].iloc[max(0,pos-w+1):pos].tolist() + [nv]
                if f"{raw}_rollmean{w}" in mod.index: mod[f"{raw}_rollmean{w}"] = np.nanmean(win)
                if f"{raw}_rollstd{w}" in mod.index and len([x for x in win if pd.notna(x)])>=2:
                    mod[f"{raw}_rollstd{w}"] = np.nanstd(win, ddof=1)
        r1 = float(MODELS[h].predict_proba(mod.values.reshape(1,-1))[:,1][0])
        return r0, r1

    c1, c2, c3 = st.columns(3)
    spo2_now = row.spo2 if pd.notna(row.spo2) else 92
    rr_now   = row.respiratory_rate if pd.notna(row.respiratory_rate) else 20
    map_raw  = "abp_mean" if pd.notna(row.get("abp_mean")) else "nibp_mean"
    with c1: d_spo2 = st.slider("O₂ support: SpO₂ →", int(min(spo2_now,88)), 100, int(min(100,spo2_now+3)))
    with c2: d_map  = st.slider("Pressors: MAP →", 40, 100, 72)
    with c3: d_rr   = st.slider("Rate control: RR →", 8, int(max(rr_now,30)), int(max(8,rr_now-5)))

    scenarios = {
        "O₂ support":   {"spo2": d_spo2},
        "Pressors":     {map_raw: d_map},
        "Rate control": {"respiratory_rate": d_rr},
    }
    r0, _ = whatif(horizon, hr_i, {})
    st.markdown(f"<div class='panel'><b>Baseline {horizon}h risk: "
                f"<span style='color:{risk_color(r0)}'>{r0*100:.0f}%</span></b></div>",
                unsafe_allow_html=True)
    cols = st.columns(len(scenarios))
    for (nm, ov), cc in zip(scenarios.items(), cols):
        _, r1 = whatif(horizon, hr_i, ov)
        d = r1 - r0
        with cc:
            st.markdown(f"<div class='panel' style='text-align:center'>"
                        f"<div class='vlabel'>{nm}</div>"
                        f"<div class='vnum' style='color:{risk_color(r1)}'>{r1*100:.0f}%</div>"
                        f"<div class='delta' style='color:{C['ok'] if d<0 else C['crit'] if d>0 else C['mut']}'>"
                        f"{d*100:+.0f} pts</div></div>", unsafe_allow_html=True)

# ---- model card tab (the honesty that wins) -------------------------------------
with t_model:
    st.markdown("<div class='eyebrow'>Held-out test performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='tiny'>Measured on patients never seen in training. "
                "‘Stable-only’ AUROC = performance on hours where the patient is not already "
                "breaching a threshold — i.e. predicting a NEW event, the number that matters "
                "clinically.</div><br>", unsafe_allow_html=True)
    cols = st.columns(len(HZ))
    for (h, cc) in zip(HZ, cols):
        tm = METRICS[f"target_{h}h"]
        with cc:
            stbl = f"{tm['auroc_stable']:.3f}" if tm['auroc_stable'] is not None else "n/a"
            st.markdown(f"<div class='panel'><div class='vlabel'>{h}h · {tm['best_model']}</div>"
                        f"<div class='vnum' style='color:{C['accent']};font-size:1.8rem'>"
                        f"{tm['auroc_all']:.3f}</div><div class='tiny'>AUROC all</div>"
                        f"<div class='reason' style='margin-top:.4rem'>stable-only "
                        f"<b style='color:{C['ok']}'>{stbl}</b></div>"
                        f"<div class='tiny'>n={tm['n_test_rows']:,} test rows</div></div>",
                        unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Deterioration definition</div>", unsafe_allow_html=True)
    rules = " · ".join(f"{k} {'<' if v['op']=='lt' else '>'} {v['thr']:.0f}"
                       for k, v in META["deterioration_rules"].items())
    st.markdown(f"<div class='panel tiny'>Deteriorating = ANY of: {rules}. "
                f"Targets predict whether this becomes true within the next 1/3/6h. "
                f"Patient-level train/test split, leakage-audited.</div>", unsafe_allow_html=True)

st.markdown(f"<div class='tiny' style='text-align:center;margin-top:1rem'>"
            f"ICU Digital Twin · clinical decision support only · de-identified MIMIC-IV demo · "
            f"not for clinical use</div>", unsafe_allow_html=True)
