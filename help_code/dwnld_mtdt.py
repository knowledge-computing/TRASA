import os
import zipfile
import requests

metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata")
os.makedirs(metadata_dir, exist_ok=True)

zip_urls = [
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/historicaltopo.zip",
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/topomaps_all.zip",
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/topomaps_all_legacy.zip",
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/ustopo_current.zip",
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/ustopo_historical.zip",
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/ustopo_replacements.zip"
]

for url in zip_urls:
    zip_name = os.path.basename(url) 
    zip_path = os.path.join(metadata_dir, zip_name)
    extract_dir = os.path.join(metadata_dir, zip_name.replace(".zip", ""))  

    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name} ...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue
    else:
        print(f"✅ Already downloaded: {zip_name}")


    try:
        print(f"📂 Extracting {zip_name} to {extract_dir} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        print(f"Not a valid ZIP file: {zip_name}")

print("Done downloading and organizing metadata.")
