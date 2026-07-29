#!/usr/bin/env python3
import argparse
import pathlib
import sys
import pyvista as pv

def process_one(vtp_path: pathlib.Path, outdir: pathlib.Path, keep_names: bool):
    vtp_path = vtp_path.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        mesh = pv.read(str(vtp_path))
    except Exception as e:
        print(f"[ERROR] Failed to read {vtp_path}: {e}", file=sys.stderr)
        return False

    # Extract nodes (points)
    points = mesh.points
    point_cloud = pv.PolyData(points)
  
    for name in mesh.point_data.keys():
        point_cloud.point_data[name] = mesh.point_data[name]

    # Extract edges (lines)
    edges = mesh.extract_all_edges()


    stem = vtp_path.stem if keep_names else "output"
    nodes_out = outdir / f"{stem}_nodes.vtp"
    edges_out = outdir / f"{stem}_edges.vtp"

    try:
        point_cloud.save(str(nodes_out))
        edges.save(str(edges_out))
    except Exception as e:
        print(f"[ERROR] Failed to write outputs for {vtp_path}: {e}", file=sys.stderr)
        return False

    print(f"[OK] {vtp_path.name}: saved {len(points)} nodes -> {nodes_out.name}, "
          f"{edges.n_cells} edges -> {edges_out.name}")
    return True

def read_manifest(path: pathlib.Path):
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(pathlib.Path(s))
    return lines

def main():
    p = argparse.ArgumentParser(
        description="Split VTP polyline graphs into nodes(.vtp) and edges(.vtp)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=pathlib.Path, help="Path to a single .vtp file")
    g.add_argument("--manifest", type=pathlib.Path,
                   help="Text file with one .vtp path per line (comments with #)")
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("./split_vtp"),
                   help="Output directory (default: ./split_vtp)")
    p.add_argument("--keep-names", action="store_true",
                   help="Keep original stem in output filenames (default uses 'output')")
    args = p.parse_args()

    tasks = []
    if args.input:
        tasks = [args.input]
    else:
        if not args.manifest.exists():
            p.error(f"Manifest not found: {args.manifest}")
        tasks = read_manifest(args.manifest)
        if not tasks:
            p.error("Manifest resolved to 0 files.")

    ok = 0
    for v in tasks:
        if not v.suffix.lower() == ".vtp":
            print(f"[WARN] Skipping non-VTP path: {v}", file=sys.stderr)
            continue
        if not v.exists():
            print(f"[WARN] Missing file: {v}", file=sys.stderr)
            continue
        ok += process_one(v, args.outdir, args.keep_names)

    print(f"\nDone. Successful files: {ok}/{len(tasks)}")
    sys.exit(0 if ok == len(tasks) else 1)

if __name__ == "__main__":
    main()
