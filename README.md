# ICU Digital Twin — Early Deterioration Prediction

**What it is, in one sentence:** a system that watches an ICU patient's vital signs hour by hour, predicts whether they are likely to get worse in the next 1, 3, or 6 hours, explains *why*, and lets a clinician test "what if we intervene?" — all shown on a live dashboard.

This README has two layers. The first part is written for **anyone** — a reviewer, a judge, or a colleague with no machine-learning background. The technical detail comes afterward, clearly marked, for people who want to check how it works under the hood.

---

## Part 1 — For everyone (no technical background needed)

### The problem we're solving

In an intensive care unit, patients can deteriorate quickly. Nurses and doctors watch monitors, but a single ward may have many patients, and the early warning signs — oxygen creeping down, heart rate creeping up, blood pressure sliding — are easy to miss when they happen slowly across several patients at once.

We built a tool that does this watching automatically. For every patient, every hour, it answers three questions:

1. **How is this patient right now?** (Their current vital signs, shown clearly.)
2. **Are they likely to get worse soon?** (A risk score for the next 1, 3, and 6 hours.)
3. **Why?** (A plain list of what's driving the concern — e.g. "oxygen is low and falling, breathing rate is high.")

It also lets a clinician ask **"what if?"** — for example, *if we give oxygen support and raise SpO₂ by 3 points, does the predicted risk drop?* The tool re-calculates and shows the before-and-after.

### What "deterioration" means here

We deliberately did **not** try to predict death. That is harder, ethically heavier, and less useful for a bedside tool. Instead we predict something concrete and clinically recognisable: whether the patient will cross any of these well-known danger thresholds soon.

| Vital sign | Danger threshold |
|---|---|
| Oxygen saturation (SpO₂) | below 90% |
| Mean blood pressure (MAP) | below 65 mmHg |
| Heart rate | above 130 beats/min |
| Breathing rate | above 30 breaths/min |
| Temperature | above 39 °C |

If a patient is heading toward *any* of these within the next few hours, that's what we flag. A nurse looking at the dashboard immediately understands what the alert means, because these are the same numbers they already watch.

### Where the data comes from

We used **MIMIC-IV**, a large, publicly available, and fully **anonymised** database of real ICU patients from a US hospital. We used its official demo subset (about 100 patients). No real names or identities are in it — any names you see on the dashboard (e.g. "Chen Y.") are cosmetic labels we generate for display only, because the real data has no names by design.

### What the dashboard shows

- **Ward view** — every patient in the unit, sorted with the highest-risk first. This is the "who do I need to worry about right now" screen. It doubles as an alert queue.
- **Patient view** — one patient in detail: current vitals, active alerts, the plain-language reasons behind their risk, and a clinical summary.
- **Trends** — how each vital has moved over the stay, plus how the predicted risk has risen or fallen over time.
- **What-if** — the intervention simulator described above.
- **Model card** — an honest report of how accurate the system is (more on this below).

### The most important thing about this project: honesty

It is easy to build a demo that *looks* impressive but quietly cheats — for example, by testing the system on the same patients it learned from, which makes accuracy look far better than it really is. We went out of our way **not** to do that. Three examples a reviewer should know:

1. **We never test on patients the system has already seen.** Patients are split into a training group and a separate testing group, and the two never mix. The accuracy we report comes only from patients the system had never encountered.

2. **We report the "harder" accuracy number, not the flattering one.** A patient who is *already* in danger is easy to flag — that's not the useful case. The useful case is spotting a patient who looks *stable now* but is about to worsen. We measure and report accuracy specifically on those currently-stable patients, which is a tougher and more meaningful test.

3. **Everything on the dashboard traces back to real data or the model.** The alerts come from the patient's actual vital signs against the thresholds above. The "why" explanations come from the same rules. Even the medication panel uses a genuine model-attribution method rather than made-up numbers. Nothing is invented for show.

This honesty is not just ethics — it's what makes the results trustworthy to a clinician, and it's the single strongest point in the project's favour.

### What this is and isn't

It **is** a working prototype that demonstrates the full idea end to end on real (anonymised) ICU data. It **is not** a medical device and must not be used for actual patient care. Every screen carries a "decision support only" note. Think of it as a convincing proof of concept, not a product.

---

## Part 2 — For technical reviewers

### Pipeline overview

The project is one Colab notebook that runs top to bottom, plus a Streamlit dashboard. The notebook is organised as a linear pipeline where each stage reads from a single shared configuration, so the deterioration rule and feature list are defined once and reused everywhere (no silent inconsistencies).

```
MIMIC-IV demo  →  hourly vital grid  →  deterioration label + 1/3/6h targets
      →  feature engineering (raw + trends + meds)  →  patient-level split + leakage audit
      →  5 models × 3 horizons (CV)  →  Optuna tuning (train only)
      →  honest evaluation (overall + stable-only)  →  SHAP + what-if + risk card  →  artifacts
```

### Stage-by-stage (matches the notebook cells)

- **Data ingestion (cells 1–3).** Downloads the MIMIC-IV demo tables to Drive (idempotent — skips files already present), loads them, and runs a cohort sanity check (no orphan ICU stays). ~140 ICU stays across ~100 patients.

- **Config as single source of truth (cell 2).** The deterioration rule (SpO₂<90, MAP<65, HR>130, RR>30, Temp>39 °C) and the prediction horizons (1/3/6 h) are defined once. Temperature is charted in Fahrenheit in MIMIC, so the 39 °C threshold is converted to 102.2 °F here rather than hard-coded downstream.

- **Feature registry + coverage check (cells 4–5).** Vital-sign `itemid`s are selected by verifying their actual coverage in *this* cohort (evidence-based, not guessed from labels). Blood pressure has both invasive (arterial) and non-invasive sources; GCS is summed from its three components.

- **Vitals → hourly grid (cells 7–8).** Raw events are pivoted wide and resampled to a per-stay hourly grid. Missing values are forward-filled with a **strict 3-hour cap and never using future data**, so no information leaks backward in time.

- **Labelling (cell 9).** `deteriorating_now` = any threshold breached at that hour. The prediction targets ask: does deterioration occur *strictly within* the next 1/3/6 hours? End-of-stay rows where the future window runs off the end become NaN rather than being guessed — a common and important leakage guard.

- **Feature engineering (cells 10–13).**
  - *Trends:* per-vital delta, rolling mean, and rolling std over 3h and 6h windows, all backward-looking.
  - *Medications:* binary "given at/before this hour" flags derived from `inputevents`, restricted to drugs appearing in ≥5 stays (avoids single-patient overfitting on a small cohort). Flags are leakage-safe (they only turn on from the first recorded dose onward).

- **Split + leakage audit (cells 14–16).** A `GroupShuffleSplit` **by `subject_id`** ensures a patient's rows are never in both train and test. An explicit assert-based audit then verifies zero subject overlap, zero stay overlap, row conservation, and no duplicate (stay, time) keys — and halts the notebook if any check fails.

- **Modelling (cells 16–19).** Five models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) across all three horizons, evaluated with 5-fold **patient-grouped** cross-validation. Linear/RF models get median imputation; the gradient boosters receive raw NaNs so that missingness itself can act as signal.

- **Tuning (cells 20–21).** Optuna (25 trials) tunes the three boosters using 3-fold grouped CV **on the training patients only**; the held-out test set is untouched until the very end. Out-of-fold predictions are generated within the training patients for the best booster per horizon.

- **Honest evaluation (cells 22–23).** Two numbers are reported per horizon: overall AUROC, and **stable-only AUROC** (rows where the patient is not already breaching a threshold — i.e. predicting a *new* event rather than the persistence of an ongoing one). The stable-only figure is the clinically meaningful one. A final clean pass refits on the 80 train patients and scores the 20 held-out test patients.

- **Explainability + twin behaviour (cells 24–27).** SHAP on the held-out test shows which vitals drive predictions per horizon. A rule-based per-patient reason list mirrors the clinical thresholds. The what-if simulator overrides a vital and — importantly — **rebuilds that vital's trend features from real prior hours**, so level and trend move together and the intervention is physiologically coherent. A digital-twin risk card emits the goal-document patient JSON (vitals, 1/3/6h risk, level, reasons).

- **Artifacts (cell 28+).** Trained models, the feature table, the patient roster, metadata, and held-out test metrics are saved so the dashboard is *production-like*: it loads saved objects and does not re-run the pipeline.

### Known limitations (stated plainly, because reviewers will ask)

- **Small cohort.** The MIMIC-IV *demo* is ~100 patients; absolute metrics will shift on the full dataset. The pipeline is built to scale to full MIMIC/eICU without code changes.
- **Mild tuning optimism.** Hyperparameters were tuned on the training patients with grouped CV; the held-out test is the honest headline, but the tuned-OOF numbers carry the usual small optimism and are labelled as such in the notebook.
- **Medication SHAP is associational, not causal.** "Drug X raises risk" reflects *which patients receive that drug* (confounding by indication), not a claim that the drug is harmful. The dashboard wording reflects this.
- **Simulation, not live monitoring.** The MIMIC extract is historical; "simulation mode" replays real recorded hours rather than streaming a live feed.
- **Not a medical device.** Decision support only; not validated for clinical use.

### Tech stack

Python · pandas / NumPy · scikit-learn · XGBoost / LightGBM / CatBoost · Optuna · SHAP · Streamlit · Plotly · joblib. Data: MIMIC-IV demo (PhysioNet).

### How to run

1. Open the notebook in Google Colab and run all cells (mounts Drive, downloads the demo data on first run, trains, and saves artifacts to your project's `output/` folder).
2. Place `app.py` in your `icu_twin` project folder.
3. Run the dashboard cell (installs Streamlit + starts an ngrok tunnel) and open the printed URL.

### Repository contents

- `icu_twin_clean.ipynb` — the full pipeline notebook.
- `app.py` — the Streamlit dashboard.
- `output/` — saved models, feature table, roster, metadata, metrics (created when the notebook runs).

---

*ICU Digital Twin · built on de-identified MIMIC-IV demo data · clinical decision support demonstration only · not for clinical use.*
