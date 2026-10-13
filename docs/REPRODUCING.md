# Reproducing TRASA

## Inputs and data preparation

Use RGB 256×256 map patches and corrected centerline graphs matching the mapped year. Modern road vectors require historical correction before serving as reference annotations. Every patch must retain its complete-map identifier and pixel origin `[x,y]`; spatial splits are made by complete map, not by patch.

A sample index uses the following fields:

```json
{"samples": [
  {"id":"map_a_0_0", "map_id":"map_a", "split":"train", "image":"images/map_a_0_0.png", "graph":"graphs/map_a_0_0.vtp", "origin":[0,0]}
]}
```

Graphs may be JSON (`nodes: [[x,y],...], edges: [[i,j],...]`) or VTP polylines. Declare pixel or normalized coordinates explicitly. Install the optional VTK reader for compressed/appended VTP:

```bash
pip install -r requirements-data.lock
python -m trasa.prepare_data --index dataset/index.json --output data/prepared --coordinates normalized --copy-images
python -m trasa.folds --manifest data/prepared/manifest.json --output data/folds --schedule dataset/fold_schedule.json
```

Pixel-coordinate JSON inputs use `--coordinates pixels`. Unused VTP points may be removed with the explicit `--drop-isolated` flag. Self loops, invalid coordinates and graphs exceeding 100 queries are rejected. Roles derive from patch graph degree: endpoint=1, intermediate=2, intersection>=3. Clipping a road at a patch exit creates an endpoint.

An optional per-sample `road_mask` names the corrected pixel annotation for geometry supervision; it preserves road-type-dependent label widths. Road foreground is encoded as one or white (255), with zero background. Preparation, fold generation and caching retain its path, and resume fingerprints include its content. If absent, geometry targets are rasterized from the corrected graph. Reference masks supervise geometry pretraining only and are not used as graph input guidance.

The paper uses eight maps and eight folds: six complete maps for training, one for validation and one for testing. `fold_schedule.json` is a list of eight `{train:[...], val:map_id, test:map_id}` objects. Omitting `--schedule` selects a recorded deterministic sorted-map cyclic schedule, rather than asserting the original study's unspecified assignment.

## Train and freeze geometry

Start from the official SAM ViT-B checkpoint. Fine-tune the SAM-Road image encoder and geometry decoder on each fold's training maps:

```bash
python -m trasa.train_geometry --manifest data/folds/fold_00.json --sam-checkpoint weights/sam_vit_b_01ec64.pth --output runs/geometry/fold_00 --device cuda
python -m trasa.cache_guidance --manifest data/folds/fold_00.json --checkpoint runs/geometry/fold_00/best.ckpt --output data/guidance/fold_00 --device cuda
```

Geometry training uses the released SAM-Road transposed-convolution decoder and binary cross entropy. Defaults follow SAM-Road: Adam LR 1e-3, encoder LR factor 0.1, 3-pixel road labels and 3-pixel keypoint radius. Per-sample `road_width` overrides the label width; road-symbol labels can use their annotated widths. Epochs, batch size and label settings are configurable because the TRASA manuscript does not specify the geometry fine-tuning schedule. Geometry resume uses `--resume runs/geometry/fold_00/last.ckpt` and restores optimizer/RNG state.

This entry point trains the geometry branch. The original end-to-end SAM-Road implementation is included for experiments that also train its topology objective; a matching checkpoint from that procedure may instead supply frozen guidance. Record which recipe produced the geometry checkpoint.

Upstream keypoint labels use degree != 2, including endpoints. `--keypoint-semantics intersection_only` trains only intersection labels if that is the intended experiment. Checkpoint provenance records the actual setting and training maps. Caching infers those fields from checkpoints produced here and rejects inconsistent declarations or held-out training maps. External checkpoints require explicit `--training-map-ids` and `--keypoint-semantics`, as well as a matching variant, patch resolution and decoder. Vanilla SAM is initialization, not trained road guidance.

Caches contain named `road` and `intersection` probability arrays in [0,1]. The second name denotes the declared SAM-Road keypoint semantics; no thresholding or per-patch min/max normalization is applied. During TRASA graph training, geometry is frozen and supplies input guidance only; no geometry loss enters the graph objective.

## Train graph variants

```bash
python -m trasa.train --config configs/paper.yaml --manifest data/guidance/fold_00/manifest.json --output runs/trasa/fold_00 --device cuda --ablation trasa
python -m trasa.predict_dataset --manifest data/guidance/fold_00/manifest.json --checkpoint runs/trasa/fold_00/best.pt --output results/trasa/fold_00 --device cuda
```

Variants are `ldtr` (RGB), `ldtr-sam` (guidance), `ldtr-ntc` (RGB + roles), and `trasa` (guidance + roles). They share the paper graph configuration. Graph training uses AdamW LR/weight decay 1e-4, batch size 8, 100 epochs and loss weights 1/1/1/0.5. The lowest validation objective selects `best.pt`; `last.pt` retains optimizer, epoch, data-content fingerprints and RNG/loader state. Resume requires unchanged configuration and data.

For all four variants and eight prepared fold-specific guidance manifests:

```bash
python -m trasa.benchmark --fold-manifests data/guidance/fold_*/manifest.json --config configs/paper.yaml --output runs/benchmark --device cuda --dry-run
```

Remove `--dry-run` to execute training and held-out-map evaluation. Fold metrics are averaged arithmetically over the eight held-out maps. The upstream LDTR and SAM-Road trainers, inference pipelines and native baseline evaluators are additionally retained under `third_party/`; their original READMEs describe their separate dependencies and datasets.

## Evaluation protocol

`predict_dataset` merges patch graphs by map origins and evaluates complete maps. It reports geometric agreement, intersections and degree accuracy, connectivity, splits, spurious branches, symmetric APLS, and query-level four-class Type F1 for NTC variants. Query-level matching retains the no-node class before graph filtering. Final graph construction uses binary valid-node and direct edge confidence, independently of NTC or auxiliary connectivity.

Paper thresholds are 0.5 for nodes/edges, 5 pixels for matching/merging/buffers, and 8 pixels for unsupported branches. Boundary clusters use a 5-pixel band and merge nodes from different patches; overlapping interiors are not fused. APLS uses control-point insertion onto graph edges, 5-pixel subdivision, at most 500 deterministic controls per direction, missing-path penalties and the harmonic mean of directional scores. These sampling details are explicit implementation settings absent from the manuscript; validate them against the study's evaluator before asserting identical scores. Undefined IDA/connectivity denominators are reported as null with eligible counts.

`infer` predicts new images without reference graphs. Cached guidance accepts a single patch; full-map inference requires embedded geometry or its matching checkpoint. An optional GDAL affine and CRS transform output coordinates to geospatial pixel centers. Native training checkpoints contain trusted training state; load trusted checkpoint files.

The annotated benchmark and fold-specific trained weights are external inputs. Tests verify executable code and contracts; they do not reproduce the paper's accuracy tables. CUDA grid-sampling backward can be nondeterministic, so record hardware, seeds and package versions with results.
