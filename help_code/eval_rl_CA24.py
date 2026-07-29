#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass, asdict
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import heapq

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    from shapely.geometry import LineString, MultiLineString
    from shapely.strtree import STRtree
except Exception:
    LineString = None
    MultiLineString = None
    STRtree = None



@dataclass
class RasterMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float

@dataclass
class VectorMetrics:
    matched_pred: int
    matched_gt: int
    num_pred: int
    num_gt: int
    precision: float
    recall: float
    f1: float
    correctness: float
    completeness: float
    apls: float

@dataclass
class FileResult:
    file_id: str
    raster: Optional[RasterMetrics] = None
    vector: Optional[VectorMetrics] = None

@dataclass
class Summary:
    n_files: int
    raster_mean: Optional[Dict[str, float]] = None
    vector_mean: Optional[Dict[str, float]] = None



def _read_image_any(path: str) -> np.ndarray:

    if iio is not None:
        arr = iio.imread(path)
        return np.asarray(arr)
    if Image is not None:
        arr = np.array(Image.open(path))
        return arr
    raise RuntimeError("No image reader available")

def load_mask(path: str, threshold: float, assume_prob: bool) -> np.ndarray:

    arr = _read_image_any(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)

    if assume_prob:
        # normalize if looks like 0..255
        if arr.max() > 1.5:
            arr = arr / 255.0
        mask = arr >= threshold
    else:
        mask = arr > 0

    return mask

def _stem(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def pair_by_stem(gt_paths: Sequence[str], pred_paths: Sequence[str]) -> List[Tuple[str, str, str]]:
    """
    Returns list of (file_id, gt_path, pred_path) paired by filename stem
    """
    gt_map = {_stem(p): p for p in gt_paths}
    pr_map = {_stem(p): p for p in pred_paths}
    common = sorted(set(gt_map.keys()) & set(pr_map.keys()))
    pairs = [(cid, gt_map[cid], pr_map[cid]) for cid in common]
    return pairs

def load_polylines(path: str) -> List[np.ndarray]:
    """
    Loads polylines (pixel coords) from:
    - .p / .pkl: pickle of list-like objects. Each polyline is Nx2.
    - .geojson / .json: GeoJSON FeatureCollection of LineString/MultiLineString.
    Returns: list of (N,2) float arrays.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".p", ".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for k in ["lines", "polylines", "line_strings", "vectors"]:
                if k in obj:
                    obj = obj[k]
                    break

        polylines: List[np.ndarray] = []
        if isinstance(obj, (list, tuple)):
            for line in obj:
                arr = np.asarray(line, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                    polylines.append(arr[:, :2])
        return polylines

    if ext in [".geojson", ".json"]:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        polylines: List[np.ndarray] = []
        if gj.get("type") == "FeatureCollection":
            feats = gj.get("features", [])
        elif gj.get("type") == "Feature":
            feats = [gj]
        else:
            feats = [{"geometry": gj}]

        for feat in feats:
            geom = feat.get("geometry", feat)
            gtype = geom.get("type")
            coords = geom.get("coordinates")

            if gtype == "LineString":
                arr = np.asarray(coords, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] >= 2:
                    polylines.append(arr[:, :2])

            elif gtype == "MultiLineString":
                for ls in coords:
                    arr = np.asarray(ls, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] >= 2:
                        polylines.append(arr[:, :2])

        return polylines

    raise ValueError(f"Unsupported vector format: {path}")

def polylines_to_shapely(polylines: List[np.ndarray]):
    if LineString is None:
        raise RuntimeError("shapely not installed; cannot run vector metrics.")
    geoms = []
    for pl in polylines:
        if pl.shape[0] >= 2:
            geoms.append(LineString([(float(x), float(y)) for x, y in pl]))
    return geoms


# -----------------------------
# rasters
# -----------------------------

def compute_raster_metrics(gt: np.ndarray, pr: np.ndarray) -> RasterMetrics:
    """
    gt, pr are boolean arrays of same shape.
    """
    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pr.shape}")

    gt_flat = gt.reshape(-1)
    pr_flat = pr.reshape(-1)

    tp = int(np.sum(pr_flat & gt_flat))
    fp = int(np.sum(pr_flat & ~gt_flat))
    fn = int(np.sum(~pr_flat & gt_flat))
    tn = int(np.sum(~pr_flat & ~gt_flat))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return RasterMetrics(tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=precision, recall=recall, f1=f1, iou=iou, accuracy=acc)


# -----------------------------
# vectors
# -----------------------------

def _build_spatial_index(geoms):
    if STRtree is None:
        return None
    return STRtree(geoms)

def _geom_matches_any(geom, gt_index, gt_geoms, tol_px: float) -> bool:
    if gt_index is not None:
        minx, miny, maxx, maxy = geom.bounds
        # shapely 2: query returns geometries
        candidates = gt_index.query(
            geom.buffer(tol_px).envelope
        )
        for cand in candidates:
            if geom.distance(cand) <= tol_px:
                return True
        return False
    else:
        for g in gt_geoms:
            if geom.distance(g) <= tol_px:
                return True
        return False
def polyline_length(pl: np.ndarray) -> float:
    if len(pl) < 2:
        return 0.0
    diffs = np.diff(pl[:, :2], axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    return float(seg_lens.sum())


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def snap_point_to_nodes(pt: Tuple[float, float],
                        nodes: List[Tuple[float, float]],
                        snap_tol: float) -> int:
    for i, n in enumerate(nodes):
        if _euclidean(pt, n) <= snap_tol:
            return i
    nodes.append(pt)
    return len(nodes) - 1


def build_endpoint_graph(polylines: List[np.ndarray], snap_tol: float) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build graph using only polyline endpoints as nodes.
    Edge weight = full polyline length.
    """
    nodes: List[Tuple[float, float]] = []
    graph: Dict[int, List[Tuple[int, float]]] = {}

    for pl in polylines:
        if pl.shape[0] < 2:
            continue

        start = (float(pl[0, 0]), float(pl[0, 1]))
        end = (float(pl[-1, 0]), float(pl[-1, 1]))
        w = polyline_length(pl)

        u = snap_point_to_nodes(start, nodes, snap_tol)
        v = snap_point_to_nodes(end, nodes, snap_tol)

        if u == v:
            continue

        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, []).append((u, w))

    return graph


def dijkstra(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
    dist = {source: 0.0}
    pq = [(0.0, source)]

    while pq:
        cur_d, u = heapq.heappop(pq)
        if cur_d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            nd = cur_d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def approximate_apls(gt_polylines: List[np.ndarray],
                     pr_polylines: List[np.ndarray],
                     snap_tol: float,
                     min_path_length: float = 10.0) -> float:

    # APLS using endpoint-only graphs
    gt_nodes: List[Tuple[float, float]] = []
    pr_nodes: List[Tuple[float, float]] = []

    def build_graph_with_nodes(polylines: List[np.ndarray], snap_tol: float, nodes_store: List[Tuple[float, float]]):
        graph: Dict[int, List[Tuple[int, float]]] = {}
        for pl in polylines:
            if pl.shape[0] < 2:
                continue
            start = (float(pl[0, 0]), float(pl[0, 1]))
            end = (float(pl[-1, 0]), float(pl[-1, 1]))
            w = polyline_length(pl)

            u = snap_point_to_nodes(start, nodes_store, snap_tol)
            v = snap_point_to_nodes(end, nodes_store, snap_tol)

            if u == v:
                continue

            graph.setdefault(u, []).append((v, w))
            graph.setdefault(v, []).append((u, w))
        return graph

    gt_graph = build_graph_with_nodes(gt_polylines, snap_tol, gt_nodes)
    pr_graph = build_graph_with_nodes(pr_polylines, snap_tol, pr_nodes)

    if len(gt_nodes) < 2 or len(pr_nodes) < 2:
        return 0.0

    def nearest_pred_node(gt_pt: Tuple[float, float]) -> Optional[int]:
        best_i = None
        best_d = float("inf")
        for i, pt in enumerate(pr_nodes):
            d = _euclidean(gt_pt, pt)
            if d < best_d:
                best_d = d
                best_i = i
        if best_d <= snap_tol * 2:
            return best_i
        return None

    pair_scores = []

    for src in range(len(gt_nodes)):
        gt_dist = dijkstra(gt_graph, src)

        pred_src = nearest_pred_node(gt_nodes[src])
        if pred_src is None:
            continue
        pr_dist = dijkstra(pr_graph, pred_src)

        for dst, d_gt in gt_dist.items():
            if dst == src or d_gt < min_path_length:
                continue

            pred_dst = nearest_pred_node(gt_nodes[dst])
            if pred_dst is None:
                pair_scores.append(0.0)
                continue

            d_pr = pr_dist.get(pred_dst, None)
            if d_pr is None:
                pair_scores.append(0.0)
                continue

            score = 1.0 - min(1.0, abs(d_pr - d_gt) / d_gt)
            pair_scores.append(score)

    if not pair_scores:
        return 0.0
    return float(np.mean(pair_scores))


def compute_vector_metrics(gt_polylines: List[np.ndarray],
                          pr_polylines: List[np.ndarray],
                          tol_px: float) -> VectorMetrics:
    gt_geoms = polylines_to_shapely(gt_polylines)
    pr_geoms = polylines_to_shapely(pr_polylines)

    gt_index = _build_spatial_index(gt_geoms)
    pr_index = _build_spatial_index(pr_geoms)

    matched_pred = 0
    for p in pr_geoms:
        if _geom_matches_any(p, gt_index, gt_geoms, tol_px):
            matched_pred += 1

    matched_gt = 0
    for g in gt_geoms:
        if _geom_matches_any(g, pr_index, pr_geoms, tol_px):
            matched_gt += 1

    num_pred = len(pr_geoms)
    num_gt = len(gt_geoms)

    precision = matched_pred / num_pred if num_pred else 0.0
    recall = matched_gt / num_gt if num_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    correctness = precision
    completeness = recall

    apls = approximate_apls(
        gt_polylines=gt_polylines,
        pr_polylines=pr_polylines,
        snap_tol=tol_px
    )

    return VectorMetrics(
        matched_pred=matched_pred,
        matched_gt=matched_gt,
        num_pred=num_pred,
        num_gt=num_gt,
        precision=precision,
        recall=recall,
        f1=f1,
        correctness=correctness,
        completeness=completeness,
        apls=apls,
    )







def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for d in dicts for k in d.keys()})
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out

def raster_to_dict(m: RasterMetrics) -> Dict[str, float]:
    return {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "iou": m.iou,
        "accuracy": m.accuracy,
        "tp": float(m.tp),
        "fp": float(m.fp),
        "fn": float(m.fn),
        "tn": float(m.tn),
    }

def vector_to_dict(m: VectorMetrics) -> Dict[str, float]:
    return {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "correctness": m.correctness,
        "completeness": m.completeness,
        "apls": m.apls,
        "matched_pred": float(m.matched_pred),
        "matched_gt": float(m.matched_gt),
        "num_pred": float(m.num_pred),
        "num_gt": float(m.num_gt),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_raster", type=str, default=None)
    ap.add_argument("--pred_raster", type=str, default=None)
    ap.add_argument("--gt_raster_dir", type=str, default=None)
    ap.add_argument("--pred_raster_dir", type=str, default=None)

    ap.add_argument("--gt_vector", type=str, default=None)
    ap.add_argument("--pred_vector", type=str, default=None)
    ap.add_argument("--gt_vector_dir", type=str, default=None)
    ap.add_argument("--pred_vector_dir", type=str, default=None)

    ap.add_argument("--glob", type=str, default="*.png", help="Pattern used for *_dir inputs")
    ap.add_argument("--print_per_file", action="store_true")

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--assume_pred_prob", action="store_true",
                    help="Treat pred raster as prob map; apply threshold. If false, pred is treated as binary.")
    ap.add_argument("--assume_gt_prob", action="store_true",
                    help="Treat gt raster as prob map; apply threshold. Usually false (gt already binary).")

    ap.add_argument("--tol_px", type=float, default=3.0,
                    help="Distance tolerance in pixels for vector matching metrics")

    ap.add_argument("--out_json", type=str, default=None)
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    # Collect raster pairs
    raster_pairs: List[Tuple[str, str, str]] = []
    if args.gt_raster and args.pred_raster:
        raster_pairs = [("single", args.gt_raster, args.pred_raster)]
    elif args.gt_raster_dir and args.pred_raster_dir:
        gt_paths = sorted(glob(os.path.join(args.gt_raster_dir, args.glob)))
        pr_paths = sorted(glob(os.path.join(args.pred_raster_dir, args.glob)))
        raster_pairs = pair_by_stem(gt_paths, pr_paths)

    # Collect vector pairs
    vector_pairs: List[Tuple[str, str, str]] = []
    if args.gt_vector and args.pred_vector:
        vector_pairs = [("single", args.gt_vector, args.pred_vector)]
    elif args.gt_vector_dir and args.pred_vector_dir:
        gt_paths = sorted(glob(os.path.join(args.gt_vector_dir, "*.p"))) \
                 + sorted(glob(os.path.join(args.gt_vector_dir, "*.pkl"))) \
                 + sorted(glob(os.path.join(args.gt_vector_dir, "*.geojson"))) \
                 + sorted(glob(os.path.join(args.gt_vector_dir, "*.json")))
        pr_paths = sorted(glob(os.path.join(args.pred_vector_dir, "*.p"))) \
                 + sorted(glob(os.path.join(args.pred_vector_dir, "*.pkl"))) \
                 + sorted(glob(os.path.join(args.pred_vector_dir, "*.geojson"))) \
                 + sorted(glob(os.path.join(args.pred_vector_dir, "*.json")))
        vector_pairs = pair_by_stem(gt_paths, pr_paths)

    ids = sorted({fid for fid, _, _ in raster_pairs} | {fid for fid, _, _ in vector_pairs})
    raster_map = {fid: (gt, pr) for fid, gt, pr in raster_pairs}
    vector_map = {fid: (gt, pr) for fid, gt, pr in vector_pairs}

    results: List[FileResult] = []
    raster_dicts: List[Dict[str, float]] = []
    vector_dicts: List[Dict[str, float]] = []

    for fid in ids:
        fr = FileResult(file_id=fid)

        # Raster
        if fid in raster_map:
            gt_path, pr_path = raster_map[fid]
            gt = load_mask(gt_path, threshold=args.threshold, assume_prob=args.assume_gt_prob)
            pr = load_mask(pr_path, threshold=args.threshold, assume_prob=args.assume_pred_prob)
            rm = compute_raster_metrics(gt, pr)
            fr.raster = rm
            raster_dicts.append(raster_to_dict(rm))

        # Vector
        if fid in vector_map:
            if LineString is None:
                raise RuntimeError("Vector metrics requested but shapely is not installed.")
            gt_path, pr_path = vector_map[fid]
            gt_pl = load_polylines(gt_path)
            pr_pl = load_polylines(pr_path)
            vm = compute_vector_metrics(gt_pl, pr_pl, tol_px=args.tol_px)
            fr.vector = vm
            vector_dicts.append(vector_to_dict(vm))

        results.append(fr)

        if args.print_per_file:
            print(f"\n[{fid}]")
            if fr.raster:
                print("  Raster:",
                      f"P={fr.raster.precision:.4f}",
                      f"R={fr.raster.recall:.4f}",
                      f"F1={fr.raster.f1:.4f}",
                      f"IoU={fr.raster.iou:.4f}")
            if fr.vector:
                print("  Vector:",
                      f"P={fr.vector.precision:.4f}",
                      f"R={fr.vector.recall:.4f}",
                      f"F1={fr.vector.f1:.4f}",
                      f"(tol={args.tol_px}px)")

    summary = Summary(
        n_files=len(results),
        raster_mean=mean_dict([{k: v for k, v in d.items() if k in ["precision","recall","f1","iou","accuracy"]}
                               for d in raster_dicts]) if raster_dicts else None,
        vector_mean=mean_dict([
    {k: v for k, v in d.items() if k in ["precision", "recall", "f1", "correctness", "completeness", "apls"]}
    for d in vector_dicts]) if vector_dicts else None,
    )

    # Print aggregate
    print("\n=== IMOLA LDTR Evaluation Summary ===")
    print(f"Files evaluated: {summary.n_files}")
    if summary.raster_mean:
        print("Raster mean:",
              f"P={summary.raster_mean['precision']:.4f}",
              f"R={summary.raster_mean['recall']:.4f}",
              f"F1={summary.raster_mean['f1']:.4f}",
              f"IoU={summary.raster_mean['iou']:.4f}",
              f"Acc={summary.raster_mean['accuracy']:.4f}")
    if summary.vector_mean:
        print("Vector mean:",
              f"P={summary.vector_mean['precision']:.4f}",
              f"R={summary.vector_mean['recall']:.4f}",
              f"F1={summary.vector_mean['f1']:.4f}",
              f"Corr={summary.vector_mean['correctness']:.4f}",
              f"Comp={summary.vector_mean['completeness']:.4f}",
              f"APLS={summary.vector_mean['apls']:.4f}",
              f"(tol={args.tol_px}px)")

    if args.out_json:
        payload = {
            "summary": asdict(summary),
            "results": [
                {
                    "file_id": r.file_id,
                    "raster": raster_to_dict(r.raster) if r.raster else None,
                    "vector": vector_to_dict(r.vector) if r.vector else None,
                }
                for r in results
            ],
            "args": vars(args),
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote: {args.out_json}")

if __name__ == "__main__":
    main()