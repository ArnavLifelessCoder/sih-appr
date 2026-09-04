import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_07_context as context
import kg_imagery_io as imagery


def source_frame(country="Algeria", count=12):
    return pd.DataFrame({
        "source_id": [f"{country}_{i}" for i in range(count)], "country": country,
        "block_id": [f"b{i}" for i in range(count)], "lat": 30., "lon": 3.,
        "is_eog_flare": [1,1,1]+[0]*(count-3),
        "eog_flare_id": ["site_a","site_a","site_b"]+[None]*(count-3),
    })


def make_chip(root, row, missing_wc=False):
    root.mkdir(parents=True, exist_ok=True)
    x = np.broadcast_to(np.array([.1,.3,.2,.4,.2,.15],dtype="float32")[:,None,None], (6,200,200)).copy()
    wc = np.zeros((200,200),dtype="uint8")
    if not missing_wc:
        wc[:,:100]=10
        wc[:,100:]=50
    easting,northing = Transformer.from_crs("EPSG:4326","EPSG:32631",always_xy=True).transform(row["lon"],row["lat"])
    path = root / f"{row['chip_id']}.npz"
    np.savez_compressed(path, reflectance=x, valid_mask=np.ones((200,200),bool),
                        scl=np.full((200,200),5,dtype="uint8"), worldcover=wc,
                        worldcover_valid=wc!=0, bands=np.asarray(imagery.BANDS),
                        crs=np.asarray("EPSG:32631"), transform=np.array([10,0,easting-1000,0,-10,northing+1000]))
    record={k:row[k] for k in ["source_id","country","chip_id"]}
    record.update(status="ok",chip_file=path.name,crs="EPSG:32631",scene_id="test_scene",scene_datetime="2023-01-01T00:00:00Z",
                  stac_item={"id":"test_scene","collection":imagery.COLLECTION,"properties":{"datetime":"2023-01-01T00:00:00Z"}})
    context.write_json(root / f"{row['chip_id']}.json",record)
    return path,record


def prepared(tmp_path):
    inputs,root=tmp_path/"inputs",tmp_path/"output"
    inputs.mkdir()
    for country in context.COUNTRIES:
        source_frame(country).to_parquet(inputs / f"features_{country}_2022_2024.parquet")
    sample=context.prepare(inputs,root,n_per_country=4,positive_per_country=1)
    return inputs,root,sample


def test_sampler_deterministic_site_dedup_and_india_guard():
    frame=source_frame()
    one=context.select_sources(frame,"Algeria",n=8,positive_quota=3)
    two=context.select_sources(frame,"Algeria",n=8,positive_quota=3)
    pd.testing.assert_frame_equal(one,two)
    assert one.block_id.is_unique and one.chip_id.is_unique
    assert one.is_eog_flare.sum()==2
    with pytest.raises(ValueError,match="India is forbidden"):
        context.select_sources(source_frame("India"),"India",n=4,positive_quota=1)


def test_index_masks_negative_and_zero_denominators():
    values=context.normalized_difference(np.array([.4,-.1,0,np.nan]),np.array([.2,.2,0,.2]))
    assert np.isclose(values[0],1/3)
    assert np.isnan(values[1:]).all()


def test_extract_features_and_grid_validation(tmp_path):
    row=context.select_sources(source_frame(),"Algeria",n=4,positive_quota=1).iloc[0].to_dict()
    path,record=make_chip(tmp_path,row)
    imagery.validate_chip(path,row,record)
    features,qa=context.extract_features(path)
    assert len(features)==77
    assert all(k.startswith("img_") for k in features)
    assert np.isclose(features["img_full_ndvi_median"],1/3)
    assert np.isclose(features["img_full_ndbi_median"],-1/3)
    assert np.isclose(features["img_full_mndwi_median"],.2)
    assert features["img_full_wc_10_fraction"]==.5
    assert np.isclose(features["img_nearest_builtup_in_chip_m"],np.sqrt(50))
    assert qa["full_clear_fraction"]==1
    wrong=dict(row,lon=row["lon"]+.1)
    with pytest.raises(ValueError,match="center differs"):
        imagery.validate_chip(path,wrong,record)


def test_missing_landcover_is_nan_not_zero(tmp_path):
    row=context.select_sources(source_frame(),"Algeria",n=4,positive_quota=1).iloc[0].to_dict()
    path,_=make_chip(tmp_path,row,missing_wc=True)
    features,qa=context.extract_features(path)
    assert np.isnan(features["img_full_wc_10_fraction"])
    assert np.isnan(features["img_nearest_builtup_in_chip_m"])
    assert qa["full_worldcover_fraction"]==0


def test_offline_cache_resume_and_export_preserves_pending(tmp_path):
    inputs,root,sample=prepared(tmp_path)
    prior=inputs/"nb4"
    row=sample.iloc[0].to_dict()
    make_chip(prior,row)
    context.write_json(prior/"run_config.json",dict(sentinel_collection=imagery.COLLECTION,dates=imagery.DATES,
                                                  bands=imagery.BANDS,pixel_m=10))
    manifest=context.run_batch(root,inputs,offline=True)
    assert manifest.status.eq("ok").sum()==1
    assert manifest.status.eq("pending").sum()==23
    features,quality=context.export_features(root)
    assert len(features)==24 and len(features.columns)==78
    assert quality.status.eq("ok").sum()==1
    assert features.set_index("source_id").drop(index=row["source_id"]).isna().all().all()
    again=context.run_batch(root,inputs,offline=True)
    assert again.status.eq("ok").sum()==1
    assert context.bundle(root).is_file()


def test_attempt_cap_and_failed_resume(monkeypatch,tmp_path):
    inputs,root,sample=prepared(tmp_path)
    class Session:
        def close(self):
            pass
    monkeypatch.setattr(context,"make_session",Session)
    monkeypatch.setattr(context,"load_worldcover_keys",lambda *args:set())
    monkeypatch.setattr(context.importlib.metadata,"version",lambda _:"test")
    calls=[]
    def fail(row,*args):
        calls.append(row["source_id"])
        return {k:row[k] for k in ["source_id","country","chip_id"]} | {"status":"failed","error":"test"}
    monkeypatch.setattr(context,"acquire",fail)
    first=context.run_batch(root,inputs,max_new=2)
    assert len(calls)==2 and first.status.eq("failed").sum()==2
    second=context.run_batch(root,inputs,max_new=1)
    assert len(calls)==3 and len(set(calls))==3
    assert second.status.eq("failed").sum()==3


def test_changed_sample_is_rejected(tmp_path):
    inputs,root,sample=prepared(tmp_path)
    sample.loc[0,"lon"]+=1
    sample.to_csv(root/"pilot_sources.csv",index=False)
    with pytest.raises(ValueError,match="changed after preparation"):
        context.run_batch(root,inputs,offline=True)


def test_sentinel_selection_masks_clouds_and_applies_asset_scale(monkeypatch):
    rasterio = types.ModuleType("rasterio")
    errors = types.ModuleType("rasterio.errors")
    errors.RasterioError = OSError
    monkeypatch.setitem(sys.modules,"rasterio",rasterio)
    monkeypatch.setitem(sys.modules,"rasterio.errors",errors)
    def scene(name):
        assets={b:{"href":f"https://example.test/{name}/{b}","raster:bands":[{"scale":.0001,"offset":-.1}]} for b in imagery.BANDS}
        assets["scl"]={"href":f"https://example.test/{name}/scl"}
        return {"id":name,"assets":assets}
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"features":[scene("cloudy"),scene("clear")]}
    class Session:
        def post(self,url,json,timeout):
            assert json["collections"]==[imagery.COLLECTION]
            assert json["intersects"]["coordinates"]==[3.,30.]
            return Response()
    def crop(url,*args,**kwargs):
        value = (9 if "/cloudy/" in url else 5) if url.endswith("/scl") else 10000
        return np.full((200,200),value,dtype="float32")
    monkeypatch.setattr(imagery,"read_crop",crop)
    image,scl,valid,item,attempts=imagery.sentinel_crop(Session(),3.,30.,None,None)
    assert item["id"]=="clear" and len(attempts)==1
    assert valid.all() and np.allclose(image,.9)
