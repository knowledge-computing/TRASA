import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
import geopandas as gpd

# Load your CSV
df = pd.read_csv("master_inventory_la_plus_all_processed_with_prediction_links.csv")

# Drop rows without bbox
df = df.dropna(subset=["westbc", "eastbc", "southbc", "northbc"])

# Create polygons for each map
geoms = [
    box(row["westbc"], row["southbc"], row["eastbc"], row["northbc"])
    for _, row in df.iterrows()
]

# Compute union (true footprint)
union_geom = unary_union(geoms)

# Save as GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=[union_geom], crs="EPSG:4326")

# Export
gdf.to_file("map_coverage.geojson", driver="GeoJSON")
# or
gdf.to_file("map_coverage.gpkg", driver="GPKG")