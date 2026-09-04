"""Public COG acquisition, extracted from the tested NB4 single-scene pilot.

Rasterio is imported only when downloading. Local NPZ QA/features need no GDAL.
"""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pyproj import Transformer

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
SIZE, PIXEL_M, MIN_CLEAR, MAX_SCENES = 200, 10, .8, 8
DATES = "2022-01-01T00:00:00Z/2024-12-31T23:59:59Z"
COLLECTION = "sentinel-2-c1-l2a"
STAC = "https://earth-search.aws.element84.com/v1/search"
WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def make_session():
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )))
    return session


def load_worldcover_keys(session, cache):
    if cache.exists():
        return set(json.loads(cache.read_text()))
    keys = set()
    params = {"list-type": "2", "prefix": "v200/2021/map/", "max-keys": 1000}
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        response = session.get(WC_BASE, params=params, timeout=(15, 60))
        response.raise_for_status()
        xml = ET.fromstring(response.content)
        keys.update(node.text for node in xml.findall(".//s:Contents/s:Key", ns))
        if xml.findtext("s:IsTruncated", namespaces=ns) != "true":
            break
        token = xml.findtext("s:NextContinuationToken", namespaces=ns)
        if not token:
            raise ValueError("WorldCover listing omitted pagination token")
        params["continuation-token"] = token
    if not keys:
        raise ValueError("Empty WorldCover listing")
    cache.write_text(json.dumps(sorted(keys)), encoding="utf-8")
    return keys


def crop_grid(lon, lat):
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    zone = min(60, int((lon + 180) // 6) + 1)
    crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    half = SIZE * PIXEL_M / 2
    affine = from_origin(x - half, y + half, PIXEL_M, PIXEL_M)
    bounds = transform_bounds(crs, "EPSG:4326", x-half, y-half, x+half, y+half)
    return crs, affine, bounds


def read_crop(url, crs, affine, bilinear=False):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    if not url.startswith("https://"):
        raise ValueError(f"Expected public HTTPS asset: {url}")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                      GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_HTTP_TIMEOUT="60"):
        with rasterio.open(url) as source:
            with WarpedVRT(source, crs=crs, transform=affine, width=SIZE, height=SIZE,
                           src_nodata=source.nodata if source.nodata is not None else 0,
                           nodata=np.nan, dtype="float32",
                           resampling=Resampling.bilinear if bilinear else Resampling.nearest) as vrt:
                return vrt.read(1)


def worldcover_crop(bounds, crs, affine, available):
    west, south, east, north = bounds
    if east - west > 1 or east < west:
        raise ValueError("Unsupported antimeridian footprint")
    output = np.zeros((SIZE, SIZE), dtype="uint8")
    used = []
    for lat0 in range(3*math.floor(south/3), 3*math.floor(north/3)+1, 3):
        for lon0 in range(3*math.floor(west/3), 3*math.floor(east/3)+1, 3):
            tile = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
            key = f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            if key not in available:
                continue
            url = WC_BASE + key
            raster = read_crop(url, crs, affine)
            valid = np.isfinite(raster) & (raster != 0)
            if not np.isin(raster[valid], WC_CLASSES).all():
                raise ValueError("Unexpected WorldCover classes")
            output[valid] = raster[valid].astype("uint8")
            used.append(url)
    return output, used


def sentinel_crop(session, lon, lat, crs, affine):
    from rasterio.errors import RasterioError
    response = session.post(STAC, json={
        "collections": [COLLECTION],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": DATES, "limit": MAX_SCENES,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }, timeout=(15, 60))
    response.raise_for_status()
    attempts = []
    for item in response.json()["features"][:MAX_SCENES]:
        try:
            assets = item["assets"]
            scl = np.nan_to_num(read_crop(assets["scl"]["href"], crs, affine), nan=0).astype("uint8")
            valid = np.isin(scl, [4, 5, 6])
            if min(valid.mean(), valid[75:125, 75:125].mean()) < MIN_CLEAR:
                attempts.append({"item": item["id"], "reason": "cloud/coverage", "clear_fraction": float(valid.mean())})
                continue
            bands = []
            for band in BANDS:
                asset = assets[band]
                metadata = asset["raster:bands"][0]
                scale, offset = float(metadata["scale"]), float(metadata["offset"])
                if not (np.isfinite(scale) and scale > 0 and np.isfinite(offset)):
                    raise ValueError(f"Invalid scale/offset for {band}")
                bands.append(read_crop(asset["href"], crs, affine, bilinear=True) * scale + offset)
            image = np.stack(bands).astype("float32")
            valid &= np.isfinite(image).all(axis=0)
            if min(valid.mean(), valid[75:125, 75:125].mean()) < MIN_CLEAR:
                attempts.append({"item": item["id"], "reason": "missing band pixels"})
                continue
            if (image[:, valid] < -.05).mean() > .05:
                raise ValueError("Suspect reflectance offset: too many values below -0.05")
            image[:, ~valid] = np.nan
            return image, scl, valid, item, attempts
        except (RasterioError, KeyError, ValueError) as error:
            attempts.append({"item": item.get("id"), "reason": str(error)[:700]})
    raise RuntimeError("No acceptable scene: " + json.dumps(attempts))


def validate_chip(path, row, record):
    if record.get("status") != "ok" or any(
        str(record.get(key)) != str(row[key]) for key in ["source_id", "country", "chip_id"]
    ):
        raise ValueError(f"Chip identity/status mismatch: {path}")
    item = record["stac_item"]
    if item["collection"] != COLLECTION or item["id"] != record["scene_id"]:
        raise ValueError(f"Wrong collection or scene: {path}")
    date = item["properties"]["datetime"]
    if not ("2022-01-01" <= date[:10] <= "2024-12-31") or record["scene_datetime"] != date:
        raise ValueError(f"Scene outside common window or metadata mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        image, valid, wc = data["reflectance"], data["valid_mask"], data["worldcover"]
        if image.shape != (6, SIZE, SIZE) or image.dtype != np.float32:
            raise ValueError(f"Unexpected image shape/dtype: {path}")
        if valid.shape != (SIZE, SIZE) or valid.dtype != bool:
            raise ValueError(f"Invalid mask: {path}")
        if min(valid.mean(), valid[75:125, 75:125].mean()) < MIN_CLEAR:
            raise ValueError(f"Clear coverage below acceptance threshold: {path}")
        if not (np.isfinite(image[:, valid]).all() and np.isnan(image[:, ~valid]).all()):
            raise ValueError(f"Reflectance/mask inconsistency: {path}")
        if (image[:, valid] < -.05).mean() > .05:
            raise ValueError(f"Suspect reflectance offset: {path}")
        if wc.shape != valid.shape or not np.isin(wc, [0] + WC_CLASSES).all():
            raise ValueError(f"Invalid land cover: {path}")
        if not np.array_equal(data["worldcover_valid"], wc != 0):
            raise ValueError(f"WorldCover mask mismatch: {path}")
        if data["scl"].shape != valid.shape or not np.isin(data["scl"][valid], [4, 5, 6]).all():
            raise ValueError(f"Invalid SCL values under clear mask: {path}")
        if data["bands"].tolist() != BANDS:
            raise ValueError(f"Band order mismatch: {path}")
        a, b, c, d, e, f = data["transform"]
        if not np.allclose([a, b, d, e], [PIXEL_M, 0, 0, -PIXEL_M]):
            raise ValueError(f"Unexpected pixel grid: {path}")
        crs = str(data["crs"])
        if crs != record["crs"]:
            raise ValueError(f"CRS metadata mismatch: {path}")
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(c+100*a, f+100*e)
        if not np.allclose([lon, lat], [float(row["lon"]), float(row["lat"])], rtol=0, atol=1e-7):
            raise ValueError(f"Chip center differs from source: {path}")


def acquire(row, root, session, available):
    from rasterio.errors import RasterioError
    chip = root / f"{row['chip_id']}.npz"
    record = {k: row[k] for k in ["source_id", "country", "chip_id"]}
    record.update(status="failed", chip_file=chip.name)
    try:
        crs, affine, bounds = crop_grid(float(row["lon"]), float(row["lat"]))
        image, scl, valid, item, attempts = sentinel_crop(session, float(row["lon"]), float(row["lat"]), crs, affine)
        wc, urls = worldcover_crop(bounds, crs, affine, available)
        record.update(status="ok", crs=crs, clear_fraction=float(valid.mean()),
                      center_clear_fraction=float(valid[75:125,75:125].mean()),
                      worldcover_valid_fraction=float((wc != 0).mean()),
                      scene_id=item["id"], scene_datetime=item["properties"]["datetime"],
                      stac_item=item, rejected_scenes=attempts, worldcover_urls=urls)
        # Atomic replacement avoids treating an interrupted write as a cached chip.
        temporary = chip.with_suffix(".partial.npz")
        np.savez_compressed(temporary, reflectance=image, scl=scl, valid_mask=valid,
                            worldcover=wc, worldcover_valid=wc != 0,
                            transform=np.asarray(tuple(affine)[:6]), crs=np.asarray(crs), bands=np.asarray(BANDS))
        validate_chip(temporary, row, record)
        temporary.replace(chip)
    except (requests.RequestException, RasterioError, RuntimeError, KeyError, ValueError) as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    return record
