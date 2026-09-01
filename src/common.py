"""Shared paths, loaders and dtype policy for SIH26162."""
from pathlib import Path
import re, json
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
FIRMS  = DATA / "firms"
CACHE  = ROOT / "cache"
OUT    = ROOT / "outputs"
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# Country roles, fixed by the data explainer (Sec 1.1). India is NEVER trained on.
TRAIN_POS_COUNTRIES = ["Iraq", "Algeria", "Nigeria", "Libya"]
TRAIN_BKG_COUNTRIES = ["Angola", "Indonesia"]
TRAIN_COUNTRIES     = TRAIN_POS_COUNTRIES + TRAIN_BKG_COUNTRIES
HOLDOUT_COUNTRY     = "India"

INSTRUMENT_DIRS = {"modis": "MODIS", "viirs_snpp": "VIIRS_SNPP", "viirs_noaa20": "VIIRS_N20"}

FNAME_RE = re.compile(r"(?P<prefix>[a-z0-9-]+)_(?P<year>\d{4})_(?P<country>[A-Za-z]+)\.csv$")

RAW_DTYPES = {
    "latitude": "float64", "longitude": "float64",   # keep f64 until after clustering
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
        for p in sorted((FIRMS / d).glob("*.csv")):
            m = FNAME_RE.search(p.name)
            if not m:
                raise ValueError(f"unparseable FIRMS filename: {p}")
            out.append({"path": p, "instdir": d, "sensor": INSTRUMENT_DIRS[d],
                        "year": int(m.group("year")), "country": m.group("country")})
    return out

def read_firms_csv(path):
    """Read one FIRMS CSV and normalise the two schemas to a common one."""
    df = pd.read_csv(path, dtype={k: v for k, v in RAW_DTYPES.items()})
    if "bright_t31" in df.columns:                       # MODIS
        df = df.rename(columns={"brightness": "t_mir", "bright_t31": "t_lwir"})
    else:                                                # VIIRS
        df = df.rename(columns={"bright_ti4": "t_mir", "bright_ti5": "t_lwir"})
    return df
