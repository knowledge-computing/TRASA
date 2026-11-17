# trasa_ca24: the TRASA benchmark, rebuilt from public USGS sheets

This folder defines the benchmark. It holds no USGS rasters: only links, hashes, settings and the corrected road lines.

| File | What it is |
|---|---|
| `dataset.yaml` | The 8 sheets (USGS URL, year, quadrangle neatline), patch and graph settings, road widths, the 8 folds |
| `annotations/<map_id>.geojson` | Corrected road centerlines with a `road_type` field (the hand-made part of the benchmark) |
| `sources.lock.json` | SHA256 and size of every source file, written by `fetch_sources --lock` |
| `expected_checksums.json` | SHA256 of every built patch, graph and mask; others compare against it |

## What has to be filled in once (by the authors)

1. **The 8 sheets.** Download the USGS metadata table (`historicaltopo.zip` for scanned HTMC sheets, `ustopo_current.zip` / `ustopo_historical.zip` for US Topo) from `https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/`, then:

   ```bash
   python -m trasa.usgs_lookup --metadata historicaltopo.csv --products <scan_id> <scan_id> ...
   ```

   Paste the printed entries into `maps:`. Check the neatline datum (`crs`): NAD27 sheets use `EPSG:4267`, NAD83 sheets `EPSG:4269`. If a sheet's frame is not a clean quadrangle, give `neatline: {pixels: [[x,y], ...]}` in source-raster pixels instead.
2. **The corrected roads.** Save each sheet's corrected road lines as `annotations/<map_id>.geojson` (any CRS that pyproj knows; GeoPackage also works with `annotation: {path: ..., layer: roads}`). The type field must hold one of `highway`, `secondary_highway`, `light_duty_road`, `light_duty_road_paved`, `trail`, or be mapped with `road_type_map`. Crossings that are not intersections (bridges, overpasses) must not share a vertex. `python -m trasa.tiger_init` makes the first TIGER/Line 2021 draft for a sheet.
3. **The folds.** Replace the `folds:` placeholder with the eight folds used in the paper.
4. **Check the two `TODO-CHECK` settings** (`node_spacing`, `road_widths`) against what the paper labels used.
5. Lock, build, look at `data/trasa_ca24/previews/*.png` (frame and roads drawn in #536FBF over each sheet), then publish the checksums:

   ```bash
   python -m trasa.fetch_sources --spec datasets/trasa_ca24/dataset.yaml --lock
   python -m trasa.build_dataset --spec datasets/trasa_ca24/dataset.yaml --output data/trasa_ca24 --preview
   cp data/trasa_ca24/checksums.json datasets/trasa_ca24/expected_checksums.json
   git add datasets/trasa_ca24
   ```

## How others rebuild it

```bash
python -m trasa.reproduce_dataset --spec datasets/trasa_ca24/dataset.yaml --output data/benchmark
```

This downloads the 8 sheets (several GB, cached in `~/.cache/trasa` or `$TRASA_CACHE`), checks their SHA256, builds the patches, compares every file with `expected_checksums.json`, and writes `data/benchmark/folds/fold_00.json` ... `fold_07.json` for training.

If USGS ever re-issues a file, the hash check stops the build instead of silently making a different benchmark. USGS maps are public domain, so an archived copy (for example on Zenodo) can be listed under `raster: {mirrors: [...]}`; mirrors are tried when the USGS link fails and must have the same SHA256. If only `images/` differ in `verify_dataset`, the cause is a different GDAL/libjpeg build: install the pinned `requirements-research.lock`.
