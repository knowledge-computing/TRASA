import os
import csv
from pathlib import Path

CSV_PATH = "/users/4/kirsa002/IMOLA/CA_area/map_metadata_clean.csv"
SCANNED_DIR = Path("/projects/standard/yaoyi/shared/IMOLA/CA_area/outputs/scanned")
LIMIT_BYTES = 5 * 1024**3  # 5 GiB

def folder_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total

def yn(val) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "y"}

def find_folder(location_name, map_id, map_year):
    prefix = f"CA_{location_name}_{map_id}_{map_year}_"
    for p in SCANNED_DIR.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            return p
    return None

selected = []
used = 0

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader, start=1):
        if not yn(row["has_outputs"]):
            continue

        map_id = row["map_id"].strip()
        location_name = row["location_name"].strip()
        map_year = row["map_year"].strip()

        folder_name = f"CA_{location_name}_{map_id}_{map_year}_24000_geo"
        folder_path = find_folder(location_name, map_id, map_year)

        if folder_path is None:
            print(f"[SKIP missing] row {i}: {folder_name}")
            continue

        size = folder_size(folder_path)

        if used + size > LIMIT_BYTES:
            print("\n[STOP] Next folder would exceed 5 GiB")
            print(f"Current total: {used / 1024**3:.2f} GiB")
            break

        used += size
        selected.append((i, folder_name, folder_path, size))
        print(f"[ADD] row {i}: {folder_path.name} | {size / 1024**2:.1f} MB | total {used / 1024**3:.2f} GiB")

print("\nSelected folders:")
for i, folder_name, folder_path, size in selected:
    print(f"{i}\t{size / 1024**2:.1f} MB\t{folder_path}")

print(f"\nCount: {len(selected)}")
print(f"Final total: {used / 1024**3:.2f} GiB")