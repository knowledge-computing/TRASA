# Cleaned LA Metro Topo Metadata

This folder is produced by `repair_la_topo_analysis.py`.

## What was fixed

- Original collected rows: **2,423**
- File/product records kept after removing non-map datasets: **2,013**
- Unique map editions after collapsing multiple file formats: **1,128**
- Removed NED contours, 3DHP, Combined Vector, and Map Indices.
- Reconstructed `map_name`, `scale`, `year`, and `product_group` from TNM titles.
- Collapsed duplicate file-format rows into one row per map edition.

## Counts by product group

- Historical Topographic Maps: **668**
- US Topo: **460**

## Counts by scale

- 1:24,000: **979**
- 1:62,500: **76**
- 1:250,000: **25**
- 1:31,680: **20**
- 1:125,000: **9**
- 1:100,000: **9**
- 1:20,000: **5**
- 1:48,000: **3**
- 1:50,000: **1**
- 1:96,000: **1**

## Main files

- `la_metro_topo_map_editions_clean.csv`
- `la_metro_topo_map_editions_clean.xlsx`
- `la_metro_topo_map_editions_clean.gpkg`, if GeoPandas can read the original GeoPackage
- `summary_tables/`
- `plots/`
