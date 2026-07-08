import urllib.request, gzip, shutil, os

BASE = "https://physionet.org/files/mimic-iv-demo/2.2/"
# Only the tables the deterioration model needs — small and fast
FILES = [
    "icu/chartevents.csv.gz",   # vitals live here
    "icu/d_items.csv.gz",       # itemid -> label lookup
    "icu/icustays.csv.gz",      # ICU stay windows
    "hosp/patients.csv.gz",     # age, gender
    "hosp/admissions.csv.gz",   # admit/discharge, diagnosis context
]

for rel in FILES:
    out_gz = os.path.join("data", "mimic-demo", rel)
    out_csv = out_gz[:-3]  # strip .gz
    os.makedirs(os.path.dirname(out_gz), exist_ok=True)
    print(f"downloading {rel} ...")
    urllib.request.urlretrieve(BASE + rel, out_gz)
    with gzip.open(out_gz, "rb") as f_in, open(out_csv, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)   # extract
    os.remove(out_gz)                     # drop the .gz
    print(f"  -> {out_csv}")

print("done")