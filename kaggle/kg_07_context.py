"""NB6: bounded, resumable imagery preparation; no model fitting or India data.

The deliberately enriched sample is for development, not population precision.
Feature columns are pixel-derived only. Labels and acquisition QA stay separate.
"""
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kg_imagery_io import (
    BANDS, COLLECTION, DATES, MAX_SCENES, MIN_CLEAR, PIXEL_M, SIZE, WC_CLASSES,
    acquire, load_worldcover_keys, make_session, validate_chip,
)

COUNTRIES = ["Algeria", "Angola", "Indonesia", "Iraq", "Libya", "Nigeria"]
PROTOCOL = "nb6-context-v1"
REGIONS = {"full": (slice(None), slice(None)), "center": (slice(75,125), slice(75,125))}
INDEX_BANDS = {"ndvi": (3,2), "ndbi": (4,3), "mndwi": (1,4)}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def select_sources(frame, country, n=100, positive_quota=32, seed=42):
    if not 0 <= positive_quota <= n or n < 1:
        raise ValueError("Need 0 <= positive_quota <= n and n >= 1")
    if country not in COUNTRIES or not frame.country.eq(country).all():
        raise ValueError(f"Unexpected country; India is forbidden: {country}")
    if frame.source_id.isna().any() or not frame.source_id.is_unique or frame.block_id.isna().any():
        raise ValueError(f"Invalid source IDs or blocks: {country}")
    if not frame.is_eog_flare.isin([0,1]).all():
        raise ValueError(f"Invalid EOG match labels: {country}")
    if not (np.isfinite(frame[["lat", "lon"]]).all().all()
            and frame.lat.between(-80,84).all() and frame.lon.between(-180,180,inclusive="left").all()):
        raise ValueError(f"Invalid coordinates: {country}")
    frame = frame.sample(frac=1, random_state=seed).copy()
    positive = frame.loc[frame.is_eog_flare.eq(1)]
    if positive.eog_flare_id.isna().any():
        raise ValueError(f"Positive source missing site ID: {country}")
    positive = positive.drop_duplicates("eog_flare_id").drop_duplicates("block_id").head(positive_quota)
    unlabelled = frame.loc[frame.is_eog_flare.eq(0) & ~frame.block_id.isin(positive.block_id)]
    unlabelled = unlabelled.drop_duplicates("block_id").head(n-len(positive))
    chosen = pd.concat([positive, unlabelled], ignore_index=True)
    if len(chosen) != n:
        raise ValueError(f"Insufficient distinct blocks for {n} sources in {country}")
    chosen["chip_id"] = [hashlib.sha256(f"{country}:{s}".encode()).hexdigest()[:20] for s in chosen.source_id]
    # Mix labels within each country before interleaving countries across batches.
    return chosen.sample(frac=1, random_state=seed+100).reset_index(drop=True)


def prepare(input_root, root, n_per_country=100, positive_per_country=32, seed=42):
    input_root, root = Path(input_root), Path(root)
    columns = ["source_id", "country", "block_id", "lat", "lon", "is_eog_flare", "eog_flare_id"]
    parts, hashes = [], {}
    for index, country in enumerate(COUNTRIES):
        name = f"features_{country}_2022_2024.parquet"
        paths = sorted(input_root.rglob(name))
        if len(paths) != 1:
            raise ValueError(f"Attach exactly one NB2 output. {name}: {paths}")
        with paths[0].open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        frame = pd.read_parquet(paths[0], columns=columns)
        chosen = select_sources(frame, country, n_per_country, positive_per_country, seed+index)
        chosen["batch_order"] = np.arange(len(chosen))
        parts.append(chosen)
    sample = pd.concat(parts, ignore_index=True).sort_values(["batch_order","country"]).reset_index(drop=True)
    if not sample.source_id.is_unique or not sample.chip_id.is_unique:
        raise ValueError("Source/chip identifiers must be globally unique")
    config = {
        "protocol": PROTOCOL, "seed": seed, "countries": COUNTRIES,
        "sentinel_collection": COLLECTION, "dates": DATES, "bands": BANDS,
        "shape": [6,SIZE,SIZE], "pixel_m": PIXEL_M, "minimum_clear_fraction": MIN_CLEAR,
        "max_scenes": MAX_SCENES, "clear_scl_classes": [4,5,6], "holdout_loaded": False,
        "n_per_country": n_per_country, "positive_per_country": positive_per_country,
        "input_sha256": hashes,
        "sample_sha256": hashlib.sha256(sample.to_csv(index=False).encode()).hexdigest(),
        "purpose": "enriched image-context development sample, no training or population precision estimate",
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing NB6 root has a different sample/configuration; use a new output root")
    write_json(path, config)
    sample.to_csv(root / "pilot_sources.csv", index=False)
    return sample


def cache_roots(input_root):
    roots = []
    for path in sorted(Path(input_root).rglob("run_config.json")):
        config = json.loads(path.read_text())
        if (config.get("sentinel_collection") == COLLECTION and config.get("dates") == DATES
                and config.get("bands") == BANDS and config.get("pixel_m") == PIXEL_M):
            roots.append(path.parent)
    return roots


def existing_record(row, root, prior_roots):
    """Reuse only successful chips whose stored identity/grid match this source."""
    cached_failure = None
    for directory in [root] + [p for p in prior_roots if p.resolve() != root.resolve()]:
        sidecar = directory / f"{row['chip_id']}.json"
        chip = directory / f"{row['chip_id']}.npz"
        if not sidecar.exists():
            continue
        record = json.loads(sidecar.read_text())
        if any(str(record.get(key)) != str(row[key]) for key in ["source_id","country","chip_id"]):
            raise ValueError(f"Cached record identity mismatch: {sidecar}")
        if (record.get("status") == "failed" and record.get("requested_lon") == float(row["lon"])
                and record.get("requested_lat") == float(row["lat"])):
            cached_failure = record
        if record.get("status") != "ok":
            continue
        if not chip.exists():
            raise FileNotFoundError(f"Successful sidecar has no chip: {chip}")
        validate_chip(chip, row, record)
        if directory.resolve() != root.resolve():
            shutil.copy2(chip, root / chip.name)
            record["reused_from"] = str(directory)
            write_json(root / sidecar.name, record)
        return record
    if cached_failure is not None:
        write_json(root / f"{row['chip_id']}.json", cached_failure)
    return cached_failure


def save_manifest(root, sample, records):
    rows = []
    for row in sample.to_dict("records"):
        record = records.get(row["chip_id"], {})
        summary = {k: v for k,v in record.items() if k not in {"stac_item","rejected_scenes","worldcover_urls"}}
        rows.append({"source_id": row["source_id"], "country": row["country"],
                     "chip_id": row["chip_id"], "status": "pending", **summary})
    manifest = pd.DataFrame(rows)
    temporary = root / "download_manifest.tmp.csv"
    manifest.to_csv(temporary, index=False)
    temporary.replace(root / "download_manifest.csv")
    return manifest


def run_batch(root, input_root="/kaggle/input", max_new=100, max_minutes=60, retry_failed=False, offline=False):
    root = Path(root)
    if max_new < 0 or max_minutes <= 0:
        raise ValueError("max_new must be nonnegative and max_minutes positive")
    config = json.loads((root / "run_config.json").read_text())
    if config.get("protocol") != PROTOCOL or config.get("holdout_loaded") is not False:
        raise ValueError("Wrong acquisition protocol")
    if hashlib.sha256((root / "pilot_sources.csv").read_bytes()).hexdigest() != config["sample_sha256"]:
        raise ValueError("Source manifest changed after preparation")
    sample = pd.read_csv(root / "pilot_sources.csv")
    if not set(sample.country).issubset(COUNTRIES) or not sample.chip_id.is_unique:
        raise ValueError("Invalid/duplicate source manifest or India encountered")
    for row in sample.itertuples():
        expected = hashlib.sha256(f"{row.country}:{row.source_id}".encode()).hexdigest()[:20]
        if row.chip_id != expected:
            raise ValueError("Chip ID does not match source identity")
    prior_roots = cache_roots(input_root)
    records = {}
    started = time.monotonic()
    for row in sample.to_dict("records"):
        record = existing_record(row, root, prior_roots)
        if record is not None:
            records[row["chip_id"]] = record
        else:
            sidecar = root / f"{row['chip_id']}.json"
            if sidecar.exists():
                old = json.loads(sidecar.read_text())
                if old.get("status") == "failed":
                    records[row["chip_id"]] = old
    manifest = save_manifest(root, sample, records)
    attempts, consecutive_failures, session, available = 0, 0, None, None
    stop_reason = "all eligible sources processed"
    try:
        for row in sample.to_dict("records"):
            old = records.get(row["chip_id"], {})
            if old.get("status") == "ok" or (old.get("status") == "failed" and not retry_failed):
                continue
            if offline or attempts >= max_new or (time.monotonic()-started)/60 >= max_minutes:
                stop_reason = "offline or per-run attempt/time limit"
                break
            if shutil.disk_usage(root).free < 1024**3:
                stop_reason = "less than 1 GiB free disk space"
                break
            if session is None:
                session = make_session()
                available = load_worldcover_keys(session, root / "worldcover_keys.json")
            record = acquire(row, root, session, available)
            record.update(requested_lon=float(row["lon"]), requested_lat=float(row["lat"]))
            write_json(root / f"{row['chip_id']}.json", record)
            records[row["chip_id"]] = record
            attempts += 1
            manifest = save_manifest(root, sample, records)
            print(attempts, row["country"], record["status"], record.get("error", "")[:180], flush=True)
            consecutive_failures = consecutive_failures+1 if record["status"] == "failed" else 0
            if consecutive_failures >= 6:
                stop_reason = "six consecutive acquisition failures; inspect errors before continuing"
                break
    finally:
        if session is not None:
            session.close()
    versions = {p: importlib.metadata.version(p) for p in ["numpy","pandas","requests","pyproj"]}
    if not offline and attempts:
        versions["rasterio"] = importlib.metadata.version("rasterio")
    state = {"protocol": PROTOCOL, "counts": manifest.status.value_counts().to_dict(),
             "new_attempts": attempts, "elapsed_minutes": round((time.monotonic()-started)/60,2),
             "stop_reason": stop_reason, "holdout_loaded": False,
             "versions": versions}
    write_json(root / "run_state.json", state)
    return manifest


def normalized_difference(a, b):
    """Exclude negative reflectances and near-zero denominators from indices."""
    denominator = a+b
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0) & (denominator > 1e-6)
    output = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a-b, denominator, out=output, where=valid)
    return output


def summarize(values):
    finite = values[np.isfinite(values)]
    return dict(zip(["p10","median","p90"], map(float,np.percentile(finite,[10,50,90])))) if len(finite) else {
        key: np.nan for key in ["p10","median","p90"]}


def extract_features(path):
    features, quality = {}, {}
    with np.load(path, allow_pickle=False) as data:
        image, valid, wc = data["reflectance"], data["valid_mask"], data["worldcover"]
        for region, index in REGIONS.items():
            x = image[(slice(None),)+index]
            mask, land = valid[index], wc[index]
            quality[f"{region}_clear_fraction"] = float(mask.mean())
            quality[f"{region}_worldcover_fraction"] = float((land != 0).mean())
            quality[f"{region}_negative_fraction"] = float((x[:,mask] < 0).mean())
            quality[f"{region}_below_minus005_fraction"] = float((x[:,mask] < -.05).mean())
            for bi, name in enumerate(BANDS):
                features.update({f"img_{region}_{name}_{stat}": value for stat,value in summarize(x[bi][mask]).items()})
            for name,(a,b) in INDEX_BANDS.items():
                index_values = normalized_difference(x[a],x[b])[mask]
                quality[f"{region}_{name}_valid_fraction"] = float(np.isfinite(index_values).mean())
                features.update({f"img_{region}_{name}_{stat}": value for stat,value in summarize(index_values).items()})
            known = land != 0
            for label in WC_CLASSES:
                features[f"img_{region}_wc_{label}_fraction"] = float((land[known]==label).mean()) if known.any() else np.nan
        builtup = np.argwhere(wc == 50)
        features["img_nearest_builtup_in_chip_m"] = float(np.linalg.norm(
            (builtup + .5 - SIZE/2)*PIXEL_M, axis=1).min()) if len(builtup) else np.nan
    quality["review_reflectance_tail"] = quality["full_below_minus005_fraction"] > .005
    return features, quality


def export_features(root):
    root = Path(root)
    sample = pd.read_csv(root / "pilot_sources.csv")
    manifest = pd.read_csv(root / "download_manifest.csv")
    if not manifest.chip_id.is_unique or set(manifest.chip_id) != set(sample.chip_id):
        raise ValueError("Acquisition manifest does not match the fixed source sample")
    by_chip = manifest.set_index("chip_id")
    records, qa = [], []
    for row in sample.to_dict("records"):
        status = by_chip.loc[row["chip_id"], "status"]
        features, quality = {}, {}
        if status == "ok":
            record = json.loads((root / f"{row['chip_id']}.json").read_text())
            path = root / f"{row['chip_id']}.npz"
            validate_chip(path, row, record)
            features, quality = extract_features(path)
        records.append({"source_id": row["source_id"], **features})
        qa.append({"source_id": row["source_id"], "country": row["country"], "status": status, **quality})
    frame, qa_frame = pd.DataFrame(records), pd.DataFrame(qa)
    feature_cols = [c for c in frame if c.startswith("img_")]
    if not feature_cols:
        raise ValueError("No successful chips yet; acquisition failures are saved in download_manifest.csv")
    if not frame.source_id.is_unique or np.isinf(frame[feature_cols].to_numpy()).any():
        raise ValueError("Duplicate sources or infinite image features")
    frame.to_parquet(root / "image_features.parquet", index=False)
    qa_frame.to_csv(root / "image_quality.csv", index=False)
    coverage = sample[["source_id","country","is_eog_flare"]].merge(
        qa_frame[["source_id","status"]], on="source_id", validate="one_to_one")
    coverage.groupby(["country","is_eog_flare","status"], observed=True).size().rename("n").reset_index().to_csv(
        root / "coverage_by_country_label.csv", index=False)
    write_json(root / "feature_manifest.json", {
        "protocol": PROTOCOL, "features": feature_cols, "n_rows": len(frame),
        "n_successful": int(qa_frame.status.eq("ok").sum()), "holdout_loaded": False,
        "regions": "full 2 km square and central 500 m square",
        "indices": {"ndvi": "(nir-red)/(nir+red)", "ndbi": "(swir16-nir)/(swir16+nir)", "mndwi": "(green-swir16)/(green+swir16)"},
        "index_masks": "nonnegative finite operands and sum > 1e-6; invalid is NaN",
        "landcover_denominator": "known WorldCover pixels only; missing coverage is not class zero",
        "builtup_distance": "metres to closest built-up pixel center inside chip; NaN if absent, not a global nearest distance",
        "excluded": "labels, country, coordinates, IDs except join key, dates, acquisition status and all quality columns",
        "limitations": "single scene; SWIR resampled from 20 m; WorldCover year 2021; enriched sample; no trained model or accuracy estimate",
    })
    return frame, qa_frame


def make_preview(root):
    import matplotlib.pyplot as plt
    root = Path(root)
    sample = pd.read_csv(root / "pilot_sources.csv")
    manifest = pd.read_csv(root / "download_manifest.csv")
    good = sample.merge(manifest[["chip_id","status"]], on="chip_id", validate="one_to_one")
    good = good.loc[good.status.eq("ok")].groupby("country",sort=True).head(1)
    if good.empty:
        raise ValueError("No successful chips to preview")
    fig, axes = plt.subplots(len(good), 3, figsize=(10,3*len(good)), squeeze=False)
    for ax, row in zip(axes, good.itertuples()):
        with np.load(root / f"{row.chip_id}.npz", allow_pickle=False) as data:
            rgb = data["reflectance"][[2,1,0]].transpose(1,2,0)
            ax[0].imshow(np.nan_to_num(np.clip(rgb/.3,0,1),nan=0))
            ax[1].imshow(data["valid_mask"],vmin=0,vmax=1,cmap="gray")
            ax[2].imshow(data["worldcover"],vmin=0,vmax=100,cmap="tab20")
        ax[0].set_title(f"{row.country}: RGB")
        ax[1].set_title("Valid mask")
        ax[2].set_title("Land-cover class IDs")
        for a in ax: a.axis("off")
    fig.tight_layout()
    fig.savefig(root / "country_preview.png",dpi=120)
    plt.close(fig)
    return root / "country_preview.png"


def bundle(root):
    import zipfile
    root = Path(root)
    output = root.parent / "nb6_context_results.zip"
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.iterdir()):
            if path.is_file() and not (".partial" in path.name or ".tmp" in path.name):
                archive.write(path,arcname=f"{root.name}/{path.name}")
        for name in ["kg_07_context.py","kg_imagery_io.py"]:
            path = Path(__file__).parent / name
            archive.write(path,arcname=f"code/{name}")
    return output
