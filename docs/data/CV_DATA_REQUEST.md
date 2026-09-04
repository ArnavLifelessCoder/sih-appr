# Computer Vision Data Request

Do not download country-wide imagery. Export small source-centred chips for the
existing FIRMS source table. This keeps storage and compute proportional to the
candidate sources and makes the tabular and imagery branches joinable by `source_id`.

## Required data

1. Sentinel-2 Level-2A harmonized surface reflectance for 2022-2024.
2. ESA WorldCover 2021 v200 land-cover labels at 10 m.
3. A manifest mapping every chip to `source_id`, country, centre latitude and
   longitude, date window, projection, pixel size, cloud fraction, and source asset IDs.

For Sentinel-2, request bands B2, B3, B4, B8, B11, B12 and SCL. Use cloud and
cloud-shadow masking, then create dry-season and wet-season median composites.
Start with a 2 km by 2 km chip around each source, resampled to a common 10 m grid.
Preserve the original 20 m information for B11, B12, and SCL in the manifest.

WorldCover should be exported over the same chip footprints. At minimum, derive
fractions of built-up, cropland, bare ground, vegetation, and water, plus distance
to the nearest built-up pixel. These contextual features are likely more useful
than training a large image encoder immediately.

## Where to get it

- [Google Earth Engine Sentinel-2 surface reflectance](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Google Earth Engine ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200)
- [Copernicus Data Space STAC API](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)
- [ESA WorldCover direct data access](https://esa-worldcover.org/en/data-access)

Earth Engine is the simplest route for compositing and chip export. Copernicus
Data Space is the preferred direct-download route when Earth Engine export quotas
are restrictive.

## Pilot before full acquisition

Export a stratified pilot of 20,000 sources:

- all recoverable EOG-positive sources, capped by EOG site so fragmented flares do
  not dominate;
- high-score unmatched sources from every country;
- medium-score and low-score controls from every country;
- spatial grouping by the existing 10 km `block_id`.

Build the image branch only if this pilot improves corrected country-held-out
ranking or materially separates industrial-looking unmatched sources from crop and
wildfire backgrounds. Do not stack it merely because imagery is available.

## Delivery format

Preferred delivery is Cloud Optimized GeoTIFF chips or Zarr arrays plus one Parquet
manifest. Use this naming convention:

```text
cv/
  sentinel2/<country>/<source_id>.tif
  worldcover/<country>/<source_id>.tif
  chip_manifest.parquet
```

Do not include EOG labels in filenames or pixel data. Labels must be joined later by
`source_id` so the image storage itself cannot leak target status.
