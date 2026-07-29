import pandas as pd
import geopandas as gpd
from shapely.geometry import box

df = pd.read_csv("master_inventory_la_plus_all_processed_with_prediction_links.csv")

df = df.dropna(subset=["westbc", "eastbc", "southbc", "northbc"])

# create bbox polygons
df["geometry"] = df.apply(
    lambda r: box(r["westbc"], r["southbc"], r["eastbc"], r["northbc"]),
    axis=1
)

gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# save
gdf.to_file("map_boxes.gpkg", driver="GPKG")