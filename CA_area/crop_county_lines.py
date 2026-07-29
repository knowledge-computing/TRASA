import geopandas as gpd
import rasterio
from shapely.geometry import box
from pathlib import Path

# --- PATHS (MSI) ---
MAPS_ROOT = Path("maps")
TIGER_ROOT = Path("shared_data/california_roads")
OUT_ROOT = Path("outputs/clipped")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- COUNTY MATCH ---
COUNTY_MAP = {
    "Lebec": "06029",
    "Whitaker_Peak": "06037",
    "Topanga": "06037",
    "Big_Sur": "06053",
    "Soberanes": "06053",
    "Ano_Nuevo": "06081",
}

# --- FIND MAPS ---
map_files = list(MAPS_ROOT.rglob("*.tif"))

for tif_path in map_files:
    name = tif_path.stem

    # --- detect county ---
    county = None
    for key, code in COUNTY_MAP.items():
        if key in name:
            county = code
            break

    if county is None:
        continue

    print(f"\nProcessing: {name} (county {county})")

    # --- raster extent polygon ---
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        raster_crs = src.crs
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    poly = gpd.GeoDataFrame(geometry=[geom], crs=raster_crs)

    # --- TIGER path ---
    tiger_dir = TIGER_ROOT / f"fe_2007_{county}_edges"
    shp_files = list(tiger_dir.glob("*.shp"))

    if not shp_files:
        print(f"  ❌ No shapefile in {tiger_dir}")
        continue

    roads = gpd.read_file(shp_files[0])

    # --- CRS match ---
    roads = roads.to_crs(poly.crs)

    # --- CLIP ---
    clipped = gpd.clip(roads, poly)

    # --- SAVE ---
    out_path = OUT_ROOT / f"{name}_roads.gpkg"
    clipped.to_file(out_path, driver="GPKG")

    print(f"  ✅ saved: {out_path}")