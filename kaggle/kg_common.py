"""SIH26162 shared config. Works on Kaggle (/kaggle/input/...) and locally."""
from pathlib import Path
import re, os
import numpy as np, pandas as pd

def _find_root():
    env = os.environ.get("SIH_DATA")
    if env: return Path(env)
    for base in [Path("/kaggle/input"), Path("."), Path("..")]:
        if not base.exists(): continue
        for p in base.rglob("firms/modis"):
            return p.parents[1]          # -> the dir containing firms/, eog/, facilities/
    raise FileNotFoundError("could not locate data/ root")

DATA = _find_root()
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
WORK.mkdir(exist_ok=True, parents=True)
CACHE = WORK / "cache"; OUT = WORK / "outputs"
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# Country roles are FIXED by the data explainer, Sec 1.1. India is never trained on.
TRAIN_POS = ["Iraq", "Algeria", "Nigeria", "Libya"]      # flare-dense -> EOG positives
TRAIN_BKG = ["Angola", "Indonesia"]                       # vegetation-fire background
TRAIN_COUNTRIES = TRAIN_POS + TRAIN_BKG
HOLDOUT = "India"

# NOAA-20 is absent for Angola and Indonesia -- never build raw 3-sensor ratios.
SENSORS_BY_COUNTRY = {c: {"MODIS", "VIIRS_SNPP", "VIIRS_N20"} for c in
                      ["India", "Iraq", "Algeria", "Nigeria", "Libya"]}
SENSORS_BY_COUNTRY.update({c: {"MODIS", "VIIRS_SNPP"} for c in ["Angola", "Indonesia"]})

INSTRUMENT_DIRS = {"modis": "MODIS", "viirs_snpp": "VIIRS_SNPP", "viirs_noaa20": "VIIRS_N20"}
FNAME_RE = re.compile(r"(?P<prefix>[a-z0-9-]+)_(?P<year>\d{4})_(?P<country>[A-Za-z]+)\.csv$")

# acq_time MUST stay string ("0136" -> 01:36); reading as int corrupts it.
RAW_DTYPES = {
    "latitude": "float64", "longitude": "float64",
    "scan": "float32", "track": "float32", "frp": "float32",
    "brightness": "float32", "bright_t31": "float32",
    "bright_ti4": "float32", "bright_ti5": "float32",
    "acq_time": "string", "satellite": "string", "instrument": "string",
    "confidence": "string", "version": "string", "daynight": "string", "type": "float32",
}

def firms_files(instrument=None):
    dirs = [instrument] if instrument else list(INSTRUMENT_DIRS)
    out = []
    for d in dirs:
        # NB: NOAA-20 files are named viirs-jpss1_*, NOT viirs-noaa20_*
        for p in sorted((DATA / "firms" / d).glob("*.csv")):
            m = FNAME_RE.search(p.name)
            if not m: raise ValueError(f"unparseable FIRMS filename: {p}")
            out.append(dict(path=p, instdir=d, sensor=INSTRUMENT_DIRS[d],
                            year=int(m.group("year")), country=m.group("country")))
    return out

def read_firms_csv(path):
    """Read one FIRMS CSV, unifying the MODIS and VIIRS schemas."""
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    if "bright_t31" in df.columns:                      # MODIS: 4um / 11um
        df = df.rename(columns={"brightness": "t_mir", "bright_t31": "t_lwir"})
    else:                                               # VIIRS: I4 / I5
        df = df.rename(columns={"bright_ti4": "t_mir", "bright_ti5": "t_lwir"})
    return df

def load_firms(countries=None, sensors=None, columns=None):
    """Load FIRMS into one tidy frame. ~19.3M rows total; fine in Kaggle's 30 GB."""
    fs = firms_files()
    if countries: fs = [f for f in fs if f["country"] in countries]
    if sensors:   fs = [f for f in fs if f["sensor"] in sensors]
    out = []
    for f in fs:
        d = read_firms_csv(f["path"])
        d["acq_dt"] = pd.to_datetime(d.acq_date, format="%Y-%m-%d")
        d["acq_time"] = d.acq_time.str.zfill(4).astype("int16")
        d["sensor"] = f["sensor"]; d["country"] = f["country"]
        d["year"] = np.int16(f["year"]); d["type"] = d["type"].astype("int8")
        d = d.drop(columns=["instrument", "acq_date"])
        if columns: d = d[columns]
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    for c in ["sensor", "country", "satellite", "confidence", "daynight"]:
        if c in df: df[c] = df[c].astype("category")
    return df

def eog_sites(active_years=range(2019, 2025)):
    """Verified positive class: EOG/World Bank flare sites active in the FIRMS window."""
    p = DATA / "eog" / "flare_inventory" / \
        "Flare-Volume-Estimates-by-individual-Flare-Location-2012-2025.xlsx"
    w = pd.read_excel(p)
    yrs = [y for y in active_years if y in w.columns]
    w["vol"] = w[yrs].sum(axis=1, min_count=1)
    w["active"] = (w[yrs].fillna(0) > 0).any(axis=1)
    w = w.rename(columns={"Flare id": "flare_id", "Latitude": "lat", "Longitude": "lon",
                          "Country": "country", "Location": "location",
                          "Field Type": "field_type"})
    return w.loc[w.active, ["flare_id", "country", "lat", "lon", "location",
                            "field_type", "vol"]].reset_index(drop=True)
