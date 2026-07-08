"""
prepare_dataset.py
==================
Turn the raw MIMIC-IV demo tables into a model-ready dataset for the
ICU deterioration project.

Input  (5 CSVs from the MIMIC-IV demo v2.2):
    icustays.csv, chartevents.csv, d_items.csv, patients.csv, admissions.csv

Output:
    icu_features.csv   -> one row per (stay_id, time-bin) with vitals,
                          trend features, and a next-3h deterioration label
    icu_dataset_summary.txt -> quick sanity report

Pipeline:
    1. Pull the 5 core vitals out of the long-format chartevents table
    2. Coalesce the two MAP sources; unify temperature to Celsius
    3. Resample each ICU stay onto a regular time grid (hourly by default)
    4. Build trend features (deltas, rolling mean, rolling slope)
    5. Build the deterioration-in-next-3h label from clinical thresholds
    6. Merge in age / gender

Run:  python prepare_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG  -- edit these two if your layout differs
# --------------------------------------------------------------------------
# The script looks for each file in DATA_DIR, then DATA_DIR/hosp, DATA_DIR/icu.
# So both a flat folder and the real hosp/ + icu/ download layout just work.
DATA_DIR = "data/mimic-demo"
OUT_DIR = "."

RESAMPLE_FREQ = "60min"  # time-bin size. "60min" matches MIMIC charting.
                         # set "30min" if you want the 30/60-min deltas from the plan.
ROLL_WINDOW = 3        # rolling window length, in bins (3 = last 3 hours at 60min)
HORIZON_BINS = 3       # predict deterioration within this many bins ahead (3h at 60min)
FFILL_LIMIT = 2        # forward-fill a vital across at most this many empty bins

# The 5 core vitals. Format: output_name -> list of itemids that feed it.
VITAL_ITEMIDS = {
    "hr":    [220045],           # Heart Rate (bpm)
    "spo2":  [220277],           # O2 saturation pulseoxymetry (%)
    "rr":    [220210],           # Respiratory Rate (insp/min)
    "map":   [220052, 220181],   # Arterial (preferred) then Non-Invasive mean BP
}
TEMP_C_ITEMID = 223762           # Temperature Celsius
TEMP_F_ITEMID = 223761           # Temperature Fahrenheit (converted to C)

# Clinical thresholds defining "deterioration" (from the project plan)
THRESHOLDS = {
    "spo2": ("lt", 90),   # SpO2 < 90
    "map":  ("lt", 65),   # MAP  < 65
    "hr":   ("gt", 130),  # HR   > 130
    "rr":   ("gt", 30),   # RR   > 30
    "temp": ("gt", 39),   # Temp > 39 C
}


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def find_file(name):
    """Locate a table whether the layout is flat or hosp/ + icu/."""
    for cand in (name, f"hosp/{name}", f"icu/{name}"):
        p = os.path.join(DATA_DIR, cand)
        if os.path.exists(p):
            return p
    sys.exit(f"ERROR: could not find {name} under {DATA_DIR} (or its hosp/ icu/ subdirs)")


def rolling_slope(series, window):
    """Least-squares slope over a trailing window, per time step.
    Slope is 'units per bin'. NaNs inside the window are ignored."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    x_full = np.arange(window, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        y = vals[lo:i + 1]
        x = x_full[:len(y)]
        mask = ~np.isnan(y)
        if mask.sum() >= 2:                       # need 2 points to fit a line
            out[i] = np.polyfit(x[mask], y[mask], 1)[0]
    return out


# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
print("Loading tables ...")
icustays = pd.read_csv(find_file("icustays.csv"),
                       parse_dates=["intime", "outtime"])
patients = pd.read_csv(find_file("patients.csv"))
chart = pd.read_csv(
    find_file("chartevents.csv"),
    usecols=["stay_id", "charttime", "itemid", "valuenum"],
    parse_dates=["charttime"],
)

# keep only numeric readings for the itemids we care about
wanted = set(sum(VITAL_ITEMIDS.values(), [])) | {TEMP_C_ITEMID, TEMP_F_ITEMID}
chart = chart[chart["itemid"].isin(wanted) & chart["valuenum"].notna()].copy()
print(f"  kept {len(chart):,} vital readings across {chart['stay_id'].nunique()} stays")


# --------------------------------------------------------------------------
# 2. MAP EACH READING TO A VITAL NAME  (+ temperature unification)
# --------------------------------------------------------------------------
def label_vital(row):
    iid = row["itemid"]
    for name, ids in VITAL_ITEMIDS.items():
        if iid in ids:
            return name
    if iid in (TEMP_C_ITEMID, TEMP_F_ITEMID):
        return "temp"
    return None

chart["vital"] = chart["itemid"].map(
    {iid: name for name, ids in VITAL_ITEMIDS.items() for iid in ids}
    | {TEMP_C_ITEMID: "temp", TEMP_F_ITEMID: "temp"}
)

# convert Fahrenheit rows to Celsius so 'temp' is one clean scale
f_mask = chart["itemid"] == TEMP_F_ITEMID
chart.loc[f_mask, "valuenum"] = (chart.loc[f_mask, "valuenum"] - 32.0) * 5.0 / 9.0

# For MAP, arterial (220052) is preferred over non-invasive (220181).
# Give arterial higher priority so it wins when both exist in the same bin.
chart["priority"] = np.where(chart["itemid"] == 220052, 1, 0)


# --------------------------------------------------------------------------
# 3. RESAMPLE ONTO A REGULAR GRID PER STAY
# --------------------------------------------------------------------------
print(f"Resampling to {RESAMPLE_FREQ} bins per stay ...")
chart["bin"] = chart["charttime"].dt.floor(RESAMPLE_FREQ)

# within each (stay, bin, vital): for MAP take the highest-priority source,
# otherwise take the median of readings in the bin.
def collapse(group):
    if group["priority"].max() > 0:
        group = group[group["priority"] == group["priority"].max()]
    return group["valuenum"].median()

agg = (chart.groupby(["stay_id", "bin", "vital"])
            .apply(collapse)
            .reset_index(name="val"))

# pivot to wide: one column per vital
wide = agg.pivot_table(index=["stay_id", "bin"], columns="vital", values="val")
wide = wide.reset_index().sort_values(["stay_id", "bin"])

# make sure all 5 vital columns exist even if one is absent
for v in ["hr", "spo2", "rr", "map", "temp"]:
    if v not in wide.columns:
        wide[v] = np.nan

# reindex each stay onto a continuous grid so gaps are explicit, then ffill a little
vcols = ["hr", "spo2", "rr", "map", "temp"]
regridded = []
for sid, g in wide.groupby("stay_id"):
    full = pd.date_range(g["bin"].min(), g["bin"].max(), freq=RESAMPLE_FREQ)
    g = g.set_index("bin").reindex(full)
    g[vcols] = g[vcols].ffill(limit=FFILL_LIMIT)
    g["stay_id"] = sid                     # re-stamp the key on every bin
    g.index.name = "bin"
    regridded.append(g.reset_index())
wide = pd.concat(regridded, ignore_index=True)


# --------------------------------------------------------------------------
# 4. TREND FEATURES
# --------------------------------------------------------------------------
print("Building trend features ...")
feat_frames = []
for v in ["hr", "spo2", "rr", "map", "temp"]:
    g = wide.groupby("stay_id")[v]
    wide[f"{v}_d1"] = g.diff(1)                                   # change over 1 bin
    wide[f"{v}_d2"] = g.diff(2)                                   # change over 2 bins
    wide[f"{v}_rollmean"] = g.transform(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    wide[f"{v}_slope"] = (wide.groupby("stay_id")[v]
                              .transform(lambda s: rolling_slope(s, ROLL_WINDOW)))


# --------------------------------------------------------------------------
# 5. DETERIORATION LABEL (next HORIZON_BINS)
# --------------------------------------------------------------------------
print("Building next-3h deterioration label ...")
breach = pd.Series(False, index=wide.index)
for v, (op, thr) in THRESHOLDS.items():
    col = wide[v]
    breach |= (col < thr) if op == "lt" else (col > thr)
wide["deteriorated_now"] = breach.astype(int)

# label = will a breach occur in ANY of the next HORIZON_BINS bins (per stay)?
def future_label(s):
    # forward-looking rolling max over the next HORIZON_BINS (excluding current)
    rev = s[::-1]
    fut = rev.shift(1).rolling(HORIZON_BINS, min_periods=1).max()[::-1]
    return fut

wide["deterioration_next_3h"] = (
    wide.groupby("stay_id")["deteriorated_now"]
        .transform(future_label)
        .fillna(0).astype(int)
)


# --------------------------------------------------------------------------
# 6. MERGE DEMOGRAPHICS
# --------------------------------------------------------------------------
stay_meta = icustays[["stay_id", "subject_id", "hadm_id", "first_careunit"]]
wide = wide.merge(stay_meta, on="stay_id", how="left")
wide = wide.merge(patients[["subject_id", "gender", "anchor_age"]],
                  on="subject_id", how="left")
wide = wide.rename(columns={"anchor_age": "age"})


# --------------------------------------------------------------------------
# 7. CLEAN + WRITE
# --------------------------------------------------------------------------
# drop bins where every vital is missing (nothing to learn from)
vital_cols = ["hr", "spo2", "rr", "map", "temp"]
wide = wide.dropna(subset=vital_cols, how="all").reset_index(drop=True)

# tidy column order
lead = ["subject_id", "hadm_id", "stay_id", "bin", "first_careunit", "age", "gender"]
label = ["deteriorated_now", "deterioration_next_3h"]
feats = [c for c in wide.columns if c not in lead + label]
wide = wide[lead + feats + label]

out_path = os.path.join(OUT_DIR, "icu_features.csv")
wide.to_csv(out_path, index=False)

# --- summary report ---
n_pos = int(wide["deterioration_next_3h"].sum())
n_tot = len(wide)
lines = [
    "ICU deterioration dataset — prep summary",
    "=" * 42,
    f"rows (stay-bins)      : {n_tot:,}",
    f"unique ICU stays      : {wide['stay_id'].nunique()}",
    f"unique patients       : {wide['subject_id'].nunique()}",
    f"time-bin size         : {RESAMPLE_FREQ}",
    f"positive label rate   : {n_pos:,} / {n_tot:,}  ({100*n_pos/n_tot:.1f}%)",
    f"feature columns       : {len(feats)}",
    "",
    "vital coverage (non-null %):",
]
for v in vital_cols:
    lines.append(f"   {v:5s}: {100*wide[v].notna().mean():5.1f}%")
report = "\n".join(lines)
with open(os.path.join(OUT_DIR, "icu_dataset_summary.txt"), "w") as fh:
    fh.write(report + "\n")

print("\n" + report)
print(f"\nwrote {out_path}")
