# ICU Twin — Early Deterioration Prediction

A live patient-state model for the ICU that flags deterioration risk in the next few hours and explains *why*, built on real MIMIC-IV data — with a SHAP-powered explanation panel and a Streamlit dashboard.

---

## The task

Patients in intensive care are monitored continuously, but deterioration is often noticed late. The model answers, for each patient at each hour:

> **Will this patient deteriorate within the next 3 hours?**

and does so in a way a clinician can act on — showing which vital signs are driving the risk, not just a black-box score.

### What counts as "deterioration"

We deliberately do **not** predict death (harder, ethically loaded, rare in a short window). Deterioration is any of these clinical thresholds being breached:

| Vital | Condition |
|-------|-----------|
| SpO₂  | < 90 |
| MAP   | < 65 mmHg |
| Heart rate | > 130 bpm |
| Respiratory rate | > 30 /min |
| Temperature | > 39 °C |

`deterioration_next_3h` = 1 if **any** threshold is breached at **any** point in the next 3 hours of that ICU stay.

---

## Data

**Source:** [MIMIC-IV Clinical Database Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/) — openly licensed (ODbL) 100-patient subset. No credentialing needed. Full MIMIC-IV (~65k ICU stays) needs PhysioNet credentialing + CITI training + a signed DUA; the pipeline scales to it unchanged.

**Tables used (5):** `icustays` (stay windows — the spine), `chartevents` (all vitals, long format), `d_items` (itemid → label dictionary), `patients` (age/gender), `admissions` (hospital context).

**Cohort:** 100 patients, 140 ICU stays, 668,862 chartevents rows.

---

## Data engineering & preparation

Raw `chartevents` is long format — one row per measurement, keyed by `itemid`. Turning that into a model-ready table is where most of the real work is.

**1. Vital extraction.** Keep only numeric readings (`valuenum`) for the relevant itemids:

| Vital | itemid(s) | Handling |
|-------|-----------|----------|
| Heart rate | 220045 | — |
| SpO₂ | 220277 | pulse oximetry |
| Respiratory rate | 220210 | — |
| MAP | 220052, 220181 | **coalesced** — arterial preferred, non-invasive fallback |
| Temperature | 223762, 223761 | **Fahrenheit → Celsius** conversion |
| GCS | 3 components summed | eye + verbal + motor per (stay, bin) |

Two of these mattered enormously on the demo: **90% of temperature is charted in Fahrenheit** (skip the conversion and the feature is nonsense), and **60% of MAP is non-invasive** (keep only arterial and you lose most of the signal).

**2. Resample to a regular grid.** Vitals are charted irregularly, so each stay is binned to a fixed **60-minute** grid; multiple readings per bin collapse by median; short gaps forward-filled up to 2 bins.

**3. Plausibility clipping.** Physiological ranges (HR 20–250, SpO₂ 50–100, RR 3–60, MAP 20–200, temp 30–43 °C) — values outside become NaN. On the demo this removed 69 real MIMIC artifacts (e.g. a 99 °C temp, an HR of 0). Correctness insurance, not a metric boost.

**4. Missingness as signal.** A `{vital}_missing` flag per vital — "we don't have a trustworthy reading here" is itself predictive.

**5. Label construction.** `deterioration_next_3h` computed by looking forward 3 bins per stay (leakage-safe, per-stay).

**Result:** 12,233 stay-hours, **37% positive rate** — well balanced, no resampling needed.

### The one rule that governs everything: leakage-safe splitting

The train/test split is **grouped by `stay_id`** (`GroupShuffleSplit`), never by row. Bins from one patient must never straddle the split, or the model memorizes patients instead of learning deterioration. Every engineered feature is **causal and per-stay** (trailing windows only, no peeking at future bins), and all feature engineering runs through a single function so a leak can't sneak in.

---

## Feature engineering

Starting from 32 base features (5 vitals + 20 trend features + 5 missingness flags + age + gender), features were added in **blocks and kept only if cumulative test PR-AUC rose** — a disciplined ablation, not a kitchen sink.

**Trend features** (per vital): `_d1` (1-bin change), `_d2` (2-bin change), `_rollmean` (3-bin mean), `_slope` (3-bin least-squares slope), `_rollstd` (3-bin volatility).

**Engineered blocks and what each earned (cumulative test PR-AUC):**

| Block added | Features | ROC-AUC | PR-AUC | Δ PR-AUC |
|-------------|----------|---------|--------|----------|
| Base (32) | 32 | 0.749 | 0.689 | — |
| + interactions (shock index, pulse-pressure proxy) | 36 | 0.756 | 0.695 | +0.006 |
| + threshold-distance features | 42 | 0.754 | 0.693 | +0.004 |
| + burden (cumulative breaches, abnormal time) | 48 | 0.768 | 0.697 | +0.008 |
| + NEWS score | 49 | 0.768 | 0.700 | +0.011 |
| + GCS | 54 | 0.770 | 0.706 | +0.017 |
| + careunit one-hot | 63 | 0.766 | 0.694 | **dropped** |

**Care-unit one-hots were dropped** — they hurt PR-AUC (overfitting on 100 patients). Redundant `_d2` twins were also pruned. **Final locked feature set: 49 features**, test PR-AUC 0.702.

Honest note: on 100 patients some gains are within noise. The blocks with the largest, most consistent lift — burden, NEWS, GCS — are also the most clinically sensible, which is the right kind of agreement.

---

## Models & ensemble

Every model uses NaN-native trees (no imputation — missingness is signal) and `scale_pos_weight` for the 37/63 imbalance.

**Individual models (test set):**

| Model | ROC-AUC | PR-AUC | Recall (deteriorates) |
|-------|---------|--------|----------------------|
| Persistence baseline | 0.648 | 0.534 | 0.43 |
| RandomForest | 0.754 | 0.688 | 0.62 |
| XGBoost | 0.761 | 0.694 | 0.64 |
| LightGBM | 0.763 | 0.696 | 0.64 |
| CatBoost (tuned) | 0.764 | 0.690 | 0.66 |
| Ridge / Lasso (linear) | 0.765 | 0.708 | — |

The single models cluster around 0.69–0.71 PR-AUC — model choice hit diminishing returns, which is exactly why feature engineering mattered more.

**Ensemble.** Rank-mean blending (more robust than probability averaging when models sit on different scales):

| Blend | ROC-AUC | PR-AUC |
|-------|---------|--------|
| 3 boosters | 0.772 | 0.707 |
| 3 boosters + Ridge + Lasso | **0.776** | **0.713** |

The linear models add genuine diversity — they capture a smooth signal the trees miss, so blending them in helps.

### The honest headline number: cross-validated

Single-split numbers on 140 stays are noisy, so the real result is **5-fold, grouped by stay**:

- **5-model blend: PR-AUC 0.717 ± 0.050** (ROC-AUC 0.797 ± 0.028)
- CatBoost alone: 0.709 ± 0.049
- **The blend beats CatBoost in 5 of 5 folds**, and both crush the ~0.53 persistence baseline consistently.

That ±0.05 is the truth about small-data variance — reported, not hidden.

### Operating threshold (not the default 0.5)

Accuracy is the wrong metric — missing a deterioration is far costlier than a false alarm. The threshold was tuned on the precision-recall curve and set to **0.341, targeting ~85% recall**:

- Catches **85%** of deteriorations (1190 of 1400), misses 210
- False-alarm rate 53% on stable bins; ~66 alerts per 100 patient-hours

This is the operating point baked into the dashboard, chosen deliberately for an early-warning system where recall is king.

---

## Explainability (SHAP)

The model has to justify itself to a clinician, so every prediction is explained with **SHAP `TreeExplainer`** on the CatBoost model (exact and fast for tree ensembles).

**How it's wired:**
1. `TreeExplainer(model)` is built once and cached.
2. For a patient's current state, `shap_values` gives each feature's signed contribution to *that* prediction.
3. The top 6 by absolute contribution are surfaced, with sign → direction (red = pushing risk up, teal = pulling it down) and translated into plain language ("MAP falling", "SpO₂ = 88", "temperature not recently measured").
4. For the what-if panel, SHAP is recomputed after a simulated intervention; the **difference** in contributions shows which risk drivers the intervention resolved.

**What the model relies on globally (mean |SHAP|):** `map_rollmean` (0.47) dominates, then `age` (0.29), `n_breaches` (0.29), `breach_cum` (0.24), `rr_rollmean` (0.17). Reassuringly clinical — the model leans on blood-pressure level, cumulative instability, and respiratory rate, not spurious artifacts.

Framing matters and is stated in-app: SHAP shows the model's **correlational reasoning**, not a causal clinical claim.

---

## Streamlit dashboard

`app.py` serves a clinical dashboard from a `dashboard_bundle/` (produced by the notebook's export cell): `model.cbm`, `config.json`, `all_bins.parquet`, `scored_patients.parquet`.

**Panels:**
- **Patient queue** (sidebar) — all stays ranked by risk, color-coded critical/watch/stable; the alert queue.
- **Patient twin** — current risk %, risk level, demographics, current vitals with breach flags.
- **Risk trajectory** — the patient's risk across their whole stay with the alert-threshold line.
- **Why this risk** — the SHAP explanation as ranked contribution bars.
- **What-if intervention** — apply a guarded clinical intervention (O₂, vasopressors, rate/resp control) and see risk before → after, plus which drivers it resolved.

The deployment model is retrained on **all** patients (the dashboard serves, it doesn't evaluate — CV already gave the honest number), while every bin is pre-scored so the UI is instant.

### Running the dashboard on Colab

Colab's `localhost` isn't reachable from your browser, so it needs a public tunnel. The reliable recipe — persistent background launch + a **cloudflared** tunnel (localtunnel drops Streamlit's JS chunks and breaks the page):

```bash
cd /content
pkill -9 -f streamlit; pkill -9 -f cloudflared; sleep 2

# theme: force dark text on white (avoids washed-out rendering)
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
base = "light"
primaryColor = "#C1121F"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F7FA"
textColor = "#1D2733"
EOF

# make sure the bundle is local (app reads ./dashboard_bundle)
[ -d dashboard_bundle ] || cp -r "/content/drive/MyDrive/ICU Twin Synthetic/dashboard_bundle" .

streamlit run app.py --server.port 8501 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  > streamlit.log 2>&1 &
sleep 8; tail -6 streamlit.log        # must say "You can now view your Streamlit app"

wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8501   # prints a https://…trycloudflare.com URL
```

Open the `trycloudflare.com` URL. Leave the terminal open (cloudflared holds the tunnel; Ctrl-C kills it, and the URL changes on every launch). **For a live demo, run `streamlit run app.py` on a local laptop instead** — no tunnel, nothing to drop mid-pitch.

---

## Scope & limitations

- **Demo scale.** 100 patients / 140 stays validates the pipeline and trains a first model but isn't enough for clinical claims; the ±0.05 CV spread reflects that. Code runs unchanged on full MIMIC-IV.
- **Threshold-based label**, not clinician-adjudicated — a reproducible simplification.
- **Not a medical device.** Research/education only.

## Repository layout

```
prepare_dataset.py      raw MIMIC tables -> icu_features.csv
hackathon_icu.ipynb     full pipeline: clean -> engineer -> model -> ensemble -> SHAP -> export
app.py                  Streamlit dashboard
dashboard_bundle/       model.cbm, config.json, all_bins.parquet, scored_patients.parquet
data/mimic-demo/        the 5 demo CSVs (hosp/ + icu/)
```

---

## ⚠️ Notebook paths are Colab-specific — fix before running elsewhere

The notebook is written for **Google Colab + Google Drive**, so paths are hardcoded to the Drive mount. To run it anywhere else (local laptop, another cloud), change these:

1. **Drive mount (Cell 1)** — remove/skip when not on Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Project path (Cell 2)** — repoint to your local folder:
   ```python
   # Colab:
   PROJECT = Path('/content/drive/MyDrive/ICU Twin Synthetic')
   # Local — wherever you cloned the repo:
   PROJECT = Path('.')            # or an absolute path like Path('/home/you/icu-twin')
   ```

3. **Bundle path for the dashboard.** The export cell writes to `PROJECT/dashboard_bundle`, but `app.py` reads `./dashboard_bundle`. On Colab you copy it into `/content`; locally, either run `app.py` from inside `PROJECT`, or edit `app.py`:
   ```python
   BUNDLE = Path("dashboard_bundle")                     # default: run app.py from repo root
   # BUNDLE = Path("/home/you/icu-twin/dashboard_bundle")  # or an absolute path
   ```

4. **cloudflared tunnel is Colab-only.** Running locally, skip it — open `http://localhost:8501` directly, since `localhost` is now your own machine.

Everything else (data prep, feature engineering, models, SHAP) is path-independent and runs unchanged once `PROJECT` points at the right folder.
