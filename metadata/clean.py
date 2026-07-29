import pandas as pd

df = pd.read_csv("la_public_clean - la_public_clean.csv")

# Sort so preferred rows come first
df["is_preferred"] = df["map_download_link"].str.contains("_TM_geo", na=False)

df = df.sort_values(by="is_preferred", ascending=False)

# Deduplicate by prediction_link
df_clean = df.drop_duplicates(subset=["prediction_link"], keep="first")

# Optional: drop helper column
df_clean = df_clean.drop(columns=["is_preferred"])

df_clean.to_csv("la_public_deduplicated.csv", index=False)