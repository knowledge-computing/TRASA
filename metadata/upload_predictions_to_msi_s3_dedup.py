#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import quote

import boto3
import pandas as pd
from botocore.exceptions import ClientError


def split_paths(cell) -> List[str]:
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [x.strip() for x in str(cell).split(" | ") if x.strip()]


def find_prediction_files(output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    pred_dir = output_dir / "prediction"
    if not pred_dir.exists():
        return None, None

    primary = pred_dir / "roads_prediction_sampled.gpkg"
    aux = pred_dir / "roads_clip_raster_crs.gpkg"

    primary_path = primary if primary.exists() else None
    aux_path = aux if aux.exists() else None

    if primary_path is None:
        gpkg_files = sorted(pred_dir.glob("*.gpkg"))
        if gpkg_files:
            primary_path = gpkg_files[0]
            if len(gpkg_files) > 1 and aux_path is None:
                aux_path = gpkg_files[1]

    return primary_path, aux_path


def build_s3_client(access_key: str, secret_key: str, endpoint_url: str):
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )


def presign_url(s3, bucket: str, key: str, expires_seconds: int) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


def public_url(endpoint_url: str, bucket: str, key: str) -> str:
    endpoint = endpoint_url.rstrip("/")
    # Keep slashes, encode spaces and other special chars safely.
    safe_key = quote(key, safe="/")
    return f"{endpoint}/{bucket}/{safe_key}"


def object_url(s3, endpoint_url: str, bucket: str, key: str, public_read: bool, expires_seconds: int) -> str:
    if public_read:
        return public_url(endpoint_url, bucket, key)
    return presign_url(s3, bucket, key, expires_seconds)


def upload_one_file(s3, local_path: Path, bucket: str, key: str, public_read: bool) -> None:
    extra_args = {"ACL": "public-read"} if public_read else None
    if extra_args:
        s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
    else:
        s3.upload_file(str(local_path), bucket, key)


def main():
    ap = argparse.ArgumentParser(
        description="Upload unique prediction layers to MSI S3 and write links back to spreadsheet."
    )
    ap.add_argument("--inventory", required=True, help="Path to master_inventory_la_plus_all_processed.csv")
    ap.add_argument("--bucket", required=True, help="MSI S3 bucket name")
    ap.add_argument("--prefix", required=True, help="Prefix inside bucket, e.g. imola/first_predictions")
    ap.add_argument("--access-key", default=os.environ.get("MSI_S3_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "")))
    ap.add_argument("--secret-key", default=os.environ.get("MSI_S3_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "")))
    ap.add_argument("--endpoint-url", default="https://s3.msi.umn.edu/")
    ap.add_argument("--expires-seconds", type=int, default=60 * 60 * 24 * 30,
                    help="Presigned URL lifetime. Ignored when --public-read is used.")
    ap.add_argument("--public-read", action="store_true",
                    help="Upload objects with ACL public-read and store stable public URLs instead of presigned URLs.")
    ap.add_argument("--upload-aux", action="store_true",
                    help="Also upload roads_clip_raster_crs.gpkg when present")
    ap.add_argument("--output-csv", default=None, help="Output spreadsheet path")
    args = ap.parse_args()

    if not args.access_key or not args.secret_key:
        raise SystemExit(
            "Missing credentials. Pass --access-key/--secret-key or set MSI_S3_ACCESS_KEY and MSI_S3_SECRET_KEY."
        )

    inventory_path = Path(args.inventory)
    output_csv = Path(args.output_csv) if args.output_csv else inventory_path.with_name(
        inventory_path.stem + "_with_prediction_links.csv"
    )

    s3 = build_s3_client(args.access_key, args.secret_key, args.endpoint_url)

    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError as e:
        raise SystemExit(f"Bucket not accessible or does not exist: {args.bucket}\n{e}")

    df = pd.read_csv(inventory_path, low_memory=False)

    new_cols = [
        "prediction_local_path",
        "prediction_aux_local_path",
        "prediction_s3_key",
        "prediction_aux_s3_key",
        "prediction_download_url",
        "prediction_aux_download_url",
        "prediction_unique_id",
        "prediction_is_canonical_row",
        "prediction_duplicate_count",
        "prediction_url_type",
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""

    groups = {}
    for idx, row in df.iterrows():
        if not bool(row.get("processed_output_found", False)):
            continue

        output_dirs = split_paths(row.get("matched_output_paths", ""))
        if not output_dirs:
            continue

        output_dir = Path(output_dirs[0])
        primary, aux = find_prediction_files(output_dir)
        if primary is None:
            continue

        unique_id = str(primary.resolve())
        groups.setdefault(unique_id, {"rows": [], "primary": primary, "aux": aux})
        groups[unique_id]["rows"].append(idx)

    uploaded_main = 0
    uploaded_aux = 0

    for unique_id, info in groups.items():
        primary = info["primary"]
        aux = info["aux"]
        row_ids = info["rows"]
        canonical_row = row_ids[0]

        output_dir = primary.parent.parent
        map_root_name = output_dir.name
        map_type = output_dir.parent.name

        main_key = f"{args.prefix}/{map_type}/{map_root_name}/prediction/{primary.name}"
        upload_one_file(s3, primary, args.bucket, main_key, args.public_read)
        main_url = object_url(s3, args.endpoint_url, args.bucket, main_key, args.public_read, args.expires_seconds)
        uploaded_main += 1

        aux_key = ""
        aux_url = ""
        if args.upload_aux and aux is not None and aux.exists():
            aux_key = f"{args.prefix}/{map_type}/{map_root_name}/prediction/{aux.name}"
            upload_one_file(s3, aux, args.bucket, aux_key, args.public_read)
            aux_url = object_url(s3, args.endpoint_url, args.bucket, aux_key, args.public_read, args.public_read and 0 or args.expires_seconds)
            uploaded_aux += 1

        duplicate_count = len(row_ids)

        for idx in row_ids:
            df.at[idx, "prediction_local_path"] = str(primary)
            df.at[idx, "prediction_s3_key"] = main_key
            df.at[idx, "prediction_download_url"] = main_url
            df.at[idx, "prediction_unique_id"] = unique_id
            df.at[idx, "prediction_is_canonical_row"] = bool(idx == canonical_row)
            df.at[idx, "prediction_duplicate_count"] = duplicate_count
            df.at[idx, "prediction_url_type"] = "public" if args.public_read else "presigned"

            if aux_key:
                df.at[idx, "prediction_aux_local_path"] = str(aux)
                df.at[idx, "prediction_aux_s3_key"] = aux_key
                df.at[idx, "prediction_aux_download_url"] = aux_url

    df.to_csv(output_csv, index=False)

    dedup = df[df["prediction_download_url"].astype(str).str.strip() != ""].copy()
    dedup = dedup.sort_values(
        by=["prediction_is_canonical_row", "map_name", "date_on_map"],
        ascending=[False, True, True],
    )
    dedup = dedup.drop_duplicates(subset=["prediction_unique_id"], keep="first")
    dedup_path = output_csv.with_name(output_csv.stem + "_deduplicated_predictions_only.csv")
    dedup.to_csv(dedup_path, index=False)

    print(f"Saved row-level spreadsheet: {output_csv}")
    print(f"Saved deduplicated spreadsheet: {dedup_path}")
    print(f"Unique prediction files uploaded: {uploaded_main}")
    print(f"Unique aux files uploaded: {uploaded_aux}")
    print(f"Row count with download links: {(df['prediction_download_url'].astype(str).str.strip() != '').sum()}")
    print(f"Deduplicated prediction count: {len(dedup)}")
    print(f"URL type used: {'public' if args.public_read else 'presigned'}")

    if args.public_read:
        print("\nNOTE: Public URLs only work long-term if MSI S3 allows public-read ACLs and/or bucket policy access for this bucket/prefix.")


if __name__ == "__main__":
    main()