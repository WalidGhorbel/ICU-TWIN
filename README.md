# ICU Twin — Early Deterioration Prediction

A live patient-state model for the ICU that flags deterioration risk in the next few hours and explains *why*, built on real MIMIC-IV data.

---

## The task

Patients in intensive care are monitored continuously, but deterioration is often noticed late. The goal here is a model that, for each patient at each point in time, answers:

> **Will this patient deteriorate within the next 3 hours?**

and does so in a way a clinician can act on — showing which vital signs are trending the wrong way, not just a black-box score.

### What counts as "deterioration"

We deliberately do **not** predict death (harder, ethically loaded, and rare in a short window). Instead we define deterioration as any of the following clinical thresholds being breached:

| Vital | Condition |
|-------|-----------|
| SpO₂  | < 90 |
| MAP   | < 65 mmHg |
| Heart rate | > 130 bpm |
| Respiratory rate | > 30 /min |
| Temperature | > 39 °C |

The prediction target `deterioration_next_3h` is 1 if **any** of these is breached at **any** point in the next 3 hours for that ICU stay.

---

## Data

**Source:** [MIMIC-IV Clinical Database Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/) — an openly licensed (Open Data Commons ODbL) 100-patient subset of MIMIC-IV. No credentialing required. The full MIMIC-IV (~65k ICU stays) needs PhysioNet credentialing + a CITI training course + a signed Data Use Agreement; this pipeline is written so it scales straight to the full set once access is granted.

**Tables used (5 of them):**

| File | Role |
|------|------|
| `icustays.csv`    | Defines each ICU stay and its time window (`intime`→`outtime`). The spine. |
| `chartevents.csv` | Bedside measurements in long format — the source of all vitals. |
| `d_items.csv`     | Dictionary mapping `itemid` codes to human labels. |
| `patients.csv`    | Age and gender. |
| `admissions.csv`  | Hospital-level context (optional). |

---

## Data preparation

`prepare_dataset.py` turns the raw long-format tables into a model-ready wide table. The steps:

1. **Extract vitals.** `chartevents` is long format — one row per measurement, keyed by `itemid`. We keep only numeric readings (`valuenum`) for the itemids we care about:

   | Vital | itemid(s) | Note |
   |-------|-----------|------|
   | Heart rate | 220045 | — |
   | SpO₂ | 220277 | pulse oximetry |
   | Respiratory rate | 220210 | — |
   | MAP | 220052, 220181 | **arterial (220052) preferred, non-invasive (220181) fallback** |
   | Temperature | 223762, 223761 | **Fahrenheit (223761) converted to Celsius** |

2. **Coalesce MAP.** In the demo, most MAP readings are non-invasive (8,342) vs arterial (5,560), so both are merged, arterial winning when both exist in the same bin.

3. **Unify temperature.** Most temps are charted in Fahrenheit (3,379) vs Celsius (391); all are converted to a single Celsius scale.

4. **Resample to a regular grid.** Vitals are charted irregularly, so each stay is binned to a fixed interval (default **60 min**). Multiple readings in a bin are collapsed by median. Short gaps are forward-filled up to 2 bins.

5. **Build trend features.** For each vital: 1-step delta, 2-step delta, 3-bin rolling mean, and 3-bin least-squares slope.

6. **Build the label.** `deterioration_next_3h` is computed by looking forward 3 bins per stay.

7. **Merge demographics.** Age and gender joined from `patients`.

**Result on the demo:** 12,233 rows (stay-hours), 140 ICU stays, 100 patients, **37% positive label rate** (well balanced for training).

---

## Feature dictionary

Every column in `icu_features.csv`:

### Identifiers & context

| Column | Meaning |
|--------|---------|
| `subject_id` | Patient ID. |
| `hadm_id` | Hospital admission ID. |
| `stay_id` | **ICU stay ID — the grouping key. Never let one stay's rows span a train/test split.** |
| `bin` | Timestamp of the hour-bin (the time axis). |
| `first_careunit` | Which ICU the stay started in. |

### Demographics

| Column | Meaning |
|--------|---------|
| `age` | Patient age in years (from `anchor_age`). |
| `gender` | M / F. |

### Raw vitals (current value in the bin)

| Column | Meaning | Unit |
|--------|---------|------|
| `hr`   | Heart rate | bpm |
| `spo2` | Oxygen saturation | % |
| `rr`   | Respiratory rate | /min |
| `map`  | Mean arterial pressure | mmHg |
| `temp` | Temperature | °C |

### Trend features

For **each** vital `X` in {`hr`, `spo2`, `rr`, `map`, `temp`}, four features capture how it's *moving*, which is often more predictive than the raw value:

| Column | Meaning |
|--------|---------|
| `X_d1` | Change vs 1 bin ago (short-term delta). |
| `X_d2` | Change vs 2 bins ago (longer delta). |
| `X_rollmean` | Mean over the last 3 bins (smoothed level). |
| `X_slope` | Least-squares slope over the last 3 bins — units per bin. Positive = rising, negative = falling. |

So the full trend set is: `hr_d1, hr_d2, hr_rollmean, hr_slope, spo2_d1, … , temp_slope` (20 columns).

### Labels

| Column | Meaning |
|--------|---------|
| `deteriorated_now` | 1 if a threshold is breached in the **current** bin. Used to build the target; not a model input. |
| `deterioration_next_3h` | **Prediction target.** 1 if any threshold is breached within the next 3 bins. |

---

## Known data-quality caveats

These are real artifacts in MIMIC, not pipeline bugs — worth handling before training:

- **Implausible values.** Raw ranges include `temp` up to 99 °C (a Fahrenheit value entered into the Celsius field) and `hr` of 0 (asystole/monitor artifact). A physiological clip (e.g. HR 20–250, SpO₂ 50–100, temp 30–43, MAP 20–200 → else NaN) is recommended.
- **Temperature coverage is ~75%** vs ~99% for other vitals, because it's charted less often. Either impute, or use a model that handles missing values natively (tree ensembles do).

---

## How to run

```bash
# 1. Put the 5 demo CSVs under data/mimic-demo/  (flat, or in hosp/ + icu/ subdirs)
# 2. Prepare the dataset
python prepare_dataset.py
# -> writes icu_features.csv and icu_dataset_summary.txt
```

Config lives at the top of `prepare_dataset.py`: `RESAMPLE_FREQ` (set `"30min"` for the 30/60-min deltas), `ROLL_WINDOW`, `HORIZON_BINS`, and the vital itemid map.

---

## Scope & limitations

- **Demo scale.** 100 patients / 140 stays is enough to validate the pipeline and train a first model, but not to make clinical claims. The code runs unchanged on the full credentialed MIMIC-IV.
- **Not a medical device.** Research/education only.
- **Label is threshold-based**, not a clinician-adjudicated deterioration event — a simplification chosen for a reproducible, demo-friendly target.

## Next steps

- Physiological plausibility clipping (toggle in prep).
- Training script with **stay-level** train/test split + RandomForest/XGBoost.
- SHAP explainability for the "why" panel.
- Streamlit dashboard: patient twin, risk trend, explanation, alert queue.
