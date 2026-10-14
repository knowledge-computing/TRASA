# Reproducing TRASA

This guide covers the eight-map experiment, all four graph variants, complete-map inference, and tables calculated from saved artifacts. Start with the [installation instructions](README.md#installation). The corrected benchmark, original fold schedule, trained study weights, and original evaluator must be supplied separately. The commands support reproduction; execution tests do not establish agreement with published scores.

## Dataset format

The manuscript describes approximately 7,300 RGB 256×256 patches from eight USGS map sheets, five road-symbol classes, and road annotations initialized from 2021 TIGER/Line and manually corrected against historical map evidence. Uncorrected modern road vectors are unsuitable reference labels.

Create `dataset/index.json`. Paths resolve relative to this file:

```json
{
  "schema_version": 1,
  "dataset_kind": "benchmark",
  "reference_provenance": {
    "source": "2021 TIGER/Line",
    "historical_correction": "manually_corrected",
    "review_protocol": "overlay inspection against mapped-year road evidence"
  },
  "maps": {"map_a": {"width": 512, "height": 256, "map_year": 1950}},
  "samples": [
    {
      "id": "map_a_0_0",
      "map_id": "map_a",
      "split": "train",
      "image": "images/map_a_0_0.png",
      "graph": "graphs/map_a_0_0.json",
      "road_mask": "labels/map_a_0_0.png",
      "origin": [0, 0]
    }
  ]
}
```

This is a format example. Supply all patches and actual metadata for the eight maps. The initial `split` is replaced when generating folds. IDs must be unique, images must be RGB, and `origin` is the nonnegative integer top-left `[x,y]` offset in the complete map. Repeated patch origins and image-file reuse across partitions are rejected. Optional map metadata includes actual CRS and GDAL-order affine `[x0,a,b,y0,d,e]`.

A graph JSON uses `[x,y]` node coordinates and zero-based undirected edges:

```json
{
  "nodes": [[0,128], [128,128], [255,128]],
  "edges": [[0,1], [1,2]],
  "edge_widths": [3,7],
  "metadata": {"correction_status": "manually_reviewed"}
}
```

Graph metadata and optional label widths are preserved. Duplicate/reversed edges are canonicalized; conflicting duplicate widths are rejected. Self-loops, isolated reference points, nonfinite or out-of-range coordinates, fractional/invalid indices, zero-length edges, and graphs exceeding 100 nodes fail validation. Disconnected road components are allowed. No-node class 0 belongs to unmatched queries; reference roles follow degree: endpoint 1, intermediate 2, and intersection 3 for degree ≥3.

ASCII, compressed, and appended VTP polylines are supported; compressed/appended VTP needs the `data` extra. Use `--coordinates pixels` or `--coordinates normalized` explicitly. Normalized coordinates are multiplied by `patch_size - 1`; unused VTP points are removed only with `--drop-isolated`.

Optional corrected road masks are binary single-channel 0/1 or 0/255 images. Masks take precedence over graph rasterization during geometry training. Otherwise use `edge_widths`, sample-level `road_width`, or the configured fallback width. The manuscript describes road-dependent widths around 2–10 pixels; a uniform fallback does not recover original labels. Declare historical correction only when it actually occurred.

```bash
python -m trasa.prepare_data --index dataset/index.json --coordinates pixels --copy-images --output data/prepared
python -m trasa.validate_data --manifest data/prepared/manifest.json --without-guidance --paper --output results/data_validation.json
```

Validation reports invalid samples and exits nonzero on failure. `--paper` requires eight maps, 256×256 dimensions, benchmark provenance, and the historical-correction declaration; it cannot independently verify manual review. Prepared manifests retain map IDs, origins, annotation metadata, masks, and provenance.

## Eight-fold partitions

Create `dataset/fold_schedule.json` as a list of eight objects. Each object has six distinct training maps, one validation map, and one test map:

```json
[{"train":["map_a","map_b","map_c","map_d","map_e","map_f"],"val":"map_g","test":"map_h"}]
```

The example shows one fold; provide all eight. Each map must be tested exactly once, and every fold must partition the same eight IDs. All patches from a map stay together.

```bash
python -m trasa.folds --manifest data/prepared/manifest.json --schedule dataset/fold_schedule.json --output data/folds
```

Outputs are `fold_00.json` through `fold_07.json` and `schedule.json`. Without `--schedule`, the command generates a deterministic sorted-map cyclic schedule and records `generated_cyclic`. That schedule is an alternative experiment design; it is not the recovered original study assignment.

## Geometry training and guidance

Download the [official SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `weights/sam_vit_b_01ec64.pth`. Train a separate geometry model for each fold, using its six training maps and selecting the checkpoint by validation BCE:

```bash
for fold_id in 00 01 02 03 04 05 06 07; do
  python -m trasa.train_geometry --manifest "data/folds/fold_${fold_id}.json" --sam-checkpoint weights/sam_vit_b_01ec64.pth --output "runs/geometry/fold_${fold_id}" --variant vit_b --patch-size 256 --epochs 100 --batch-size 8 --learning-rate 0.001 --encoder-lr-factor 0.1 --keypoint-semantics intersection_only --device cuda
  python -m trasa.cache_guidance --manifest "data/folds/fold_${fold_id}.json" --checkpoint "runs/geometry/fold_${fold_id}/best.ckpt" --output "data/guidance/fold_${fold_id}" --sam-variant vit_b --patch-size 256 --device cuda
  python -m trasa.validate_data --manifest "data/guidance/fold_${fold_id}/manifest.json" --paper --output "results/guidance_validation_${fold_id}.json"
done
```

The trainer optimizes SAM's encoder and the released SAM-Road geometry decoder with Adam and BCE. It does not train native SAM-Road's TopoNet objective. The geometry recipe, variant, and label defaults include choices not fully specified by TRASA; see [implementation conventions](#implementation-conventions).

`intersection_only` supervises degree ≥3 nodes, following the TRASA channel description. `degree_ne_2` includes endpoints and intersections, following upstream SAM-Road. Preserve the actual semantics of an existing checkpoint; no silent relabeling occurs. Fallback road rasterization uses PIL with `--road-width 3`; keypoint discs use `--keypoint-radius 3`. Supplied masks and per-edge widths override the road fallback.

Outputs include validation-selected `best.ckpt`, resumable `last.ckpt`, `history.jsonl`, `config.json`, `environment.json`, and `provenance.json`. Resume with the original options and `--resume runs/geometry/fold_00/last.ckpt`. Geometry epoch budgets can be extended; changed settings or data are checked.

For a compatible external geometry checkpoint, declare its actual training and checkpoint-selection maps:

```bash
python -m trasa.cache_guidance --manifest data/folds/fold_00.json --checkpoint weights/external_geometry.ckpt --output data/guidance/external_fold_00 --training-map-ids map_a map_b map_c map_d map_e map_f --validation-map-ids map_g --keypoint-semantics degree_ne_2 --sam-variant vit_b --patch-size 256 --device cuda
```

Replace the illustrative IDs with true producer identities. Variant, resolution, encoder, and decoder must match. A compatible SAM mask-decoder checkpoint additionally uses `--sam-decoder`. Raw SAM initialization alone, missing geometry weights, and unconverted LoRA checkpoints are rejected. External provenance declarations cannot independently prove an unavailable training history.

Cache generation rejects geometry training on held-out maps and checkpoint selection on test maps. Local checkpoints also bind producer data fingerprints. Each NPZ binds continuous probabilities to the image hash, sample/map IDs, geometry checkpoint hash, and label semantics. Channel order is **road, intersection/keypoint**. SAM receives RGB only, normalized in 0–255 space by mean `[123.675,116.28,103.53]` and standard deviation `[58.395,57.12,57.375]`. Probabilities are not thresholded or rescaled. Reference masks supervise geometry training and are unnecessary for guidance inference.

## Graph training and tables

The four variants toggle geometry guidance and NTC as listed in the [README](README.md#training). All use [configs/paper.yaml](configs/paper.yaml): 256×256 patches, 100 queries, hidden dimension 256, three encoder/decoder layers, eight heads, four feature levels, five-hop targets, AdamW, learning rate/weight decay 1e-4, batch size 8, and 100 epochs.

For a single fold:

```bash
python -m trasa.train --config configs/paper.yaml --manifest data/guidance/fold_00/manifest.json --ablation trasa --output runs/trasa/fold_00 --device cuda
python -m trasa.predict_dataset --manifest data/guidance/fold_00/manifest.json --checkpoint runs/trasa/fold_00/best.pt --output results/trasa/fold_00 --device cuda
```

`best.pt` selects the lowest validation objective; `last.pt` preserves optimizer, RNG, loader-generator, epoch, and best-objective state. History contains per-epoch losses and validation metrics. Resume an interruption with the same options and `--resume runs/trasa/fold_00/last.pt`. Changed configuration, data, geometry, variant, or implementation fingerprints invalidate resume. `--max-epochs-this-run` permits a short run while retaining the intended total budget. A completed budget adds no epoch.

After generating all eight guidance manifests, inspect and execute all 32 graph runs:

```bash
python -m trasa.benchmark --fold-manifests data/guidance/fold_*/manifest.json --config configs/paper.yaml --output runs/benchmark --device cuda --dry-run
python -m trasa.benchmark --fold-manifests data/guidance/fold_*/manifest.json --config configs/paper.yaml --output runs/benchmark --device cuda
python -m trasa.tables --reports runs/benchmark/*/fold_*/test/metrics.json --output results/tables
```

Per-fold `test/metrics.json` reports include test-map IDs, map partitions, configuration/seed, checkpoint and data hashes, geometry provenance, execution metadata, and prediction/reference paths. Tables require eight unique held-out maps, identical cross-method split assignments, compatible metric protocols, and comparable within-method configurations. Missing metrics remain null; missing baseline results are not replaced by published values.

Table outputs are `results.json`, `main_results.csv`/`.md`, and `ablations.csv`/`.md`, with means, population standard deviations, and eligible counts. Keep alternative evaluation protocols in separate table invocations. Synthetic and benchmark reports cannot be mixed. Use fresh output directories unless a command supports resume.

## Complete-map inference

Inference needs an RGB map and trained weights, without reference road annotations:

```bash
python -m trasa.infer --checkpoint runs/trasa/fold_00/best.pt --geometry-checkpoint runs/geometry/fold_00/best.ckpt --image dataset/new_map.png --output results/new_map_graph.json --device cuda
```

The geometry checkpoint must match the training hash. RGB-only variants omit it. Tiles use a regular 256-pixel grid and black padding at image boundaries; predictions outside the map are clipped. `--stride 128` enables overlap, partitioned at patch-center midpoints. Boundary nodes merge within five pixels in bounded clusters containing at most one node from each patch. Duplicate edges are removed without inserting connector roads. Nearby distinct roads remain a possible ambiguous case for proximity matching.

For one matching 256×256 patch, use `--guidance matching.npz` instead of a geometry checkpoint; bindings are checked. A single cache cannot cover an unseen multi-patch map. JSON preserves pixel coordinates, prediction attributes, and inference provenance.

With a georeferenced RGB GeoTIFF:

```bash
python -m trasa.infer --checkpoint runs/trasa/fold_00/best.pt --geometry-checkpoint runs/geometry/fold_00/best.ckpt --image dataset/new_map.tif --read-georeferencing --output results/new_map.geojson --device cuda
```

Use `.gpkg` for source-CRS roads/nodes layers. GeoJSON reprojects to WGS84 and retains source metadata. Explicit `--affine` and `--crs` are available in `trasa.infer` and `trasa.export`. Values must come from the source raster; pixel centers use `[x+0.5,y+0.5]`. Keep pixel-space JSON for evaluation.

## Evaluation protocol

```bash
python -m trasa.evaluate --reference reference_graph.json --prediction results/new_map_graph.json --output results/new_map_metrics.json
```

Evaluate complete-map graphs in **pixels**, with five-pixel geometric/matching tolerance and an eight-pixel unsupported-branch cutoff. `predict_dataset` additionally measures Type F1 from raw queries. Reports identify all protocol parameters. The original study evaluator is unavailable, so the following matching, sampling, branch-counting, and undefined-case conventions require verification before claiming exact numerical equivalence.

Let `R` and `P` be unions of reference and predicted road segments, `L` geometric length, and `B5` a five-pixel buffer:

| Metric | Definition |
|---|---|
| Completeness `C` | `L(R ∩ B5(P)) / L(R)` |
| Correctness `K` | `L(P ∩ B5(R)) / L(P)` |
| Quality | `CK / (C + K - CK)` |
| Intersection precision / recall | Matched degree ≥3 nodes / predicted or reference intersection count |
| Intersection F1 | Harmonic mean of intersection precision and recall |
| Intersection Degree Accuracy | Equal-degree fraction among matched intersection pairs |
| Connectivity | Fraction of reference-connected, spatially matched node pairs connected in the prediction |
| Split errors | Reference components whose matched nodes occupy multiple predicted components |
| Spurious branches | Maximal degree-two-interior paths with an unsupported connected portion longer than eight pixels |

Geometric unions avoid double-counting coincident segments; buffers use Shapely's rounded caps/joins. At zero tolerance, original lines are intersected directly. Both empty road graphs score 1 for geometric metrics; exactly one empty graph scores 0. One-to-one Hungarian matching maximizes matches within tolerance before minimizing Euclidean distance. Topological intersections use graph degree, independent of NTC. Both empty intersection sets score 1; exactly one empty set scores 0. Degree accuracy and connectivity are null without eligible pairs. Connectivity excludes unmatched nodes; reports include its eligible-pair count. Split errors count affected components once and omit wholly unmatched components. Closed loops are eligible branch paths.

**APLS — `control_point_apls_v1`** is the explicitly named pixel-space approximation:

1. Subdivide source edges at spacing ≤5 pixels; connected original nodes and inserted vertices are controls.
2. Cap controls at 500 using seeded sampling without replacement (default seed 10).
3. Project controls onto target edges within five pixels and subdivide those existing edges.
4. For each source-connected unordered control pair with positive reference shortest-path length `dR`, use Euclidean edge weights and penalty `min(1, abs(dR-dP)/dR)`. Missing projections or disconnected target paths receive penalty 1.
5. Each directional score is one minus mean penalty; the final score is their harmonic mean.

Directions without eligible pairs yield 0 unless both graphs have no edges, which yields 1. The pinned SAM-Road geographic/meter evaluator is a separate protocol. Large graphs may require substantial shortest-path memory; benchmark resource consumption has not been measured.

**Type F1** uses all raw queries before validity filtering or merging. Hungarian assignment labels matched queries by reference degree and unmatched queries as no-node 0; predictions use four-class NTC argmax. Per-class precision/recall/F1 and support are reported. Undefined class ratios are 0; macro-F1 averages all four classes, including absent ones. Queries are pooled within each held-out map before scores are averaged across maps. NTC-disabled variants have no type score. Interpolated boundary vertices have no invented type prediction.

## External baselines

Full LDTR and SAM-Road sources are [included at pinned revisions](third_party/README.md). Native baseline environments are separate from TRASA; the controlled `--ablation ldtr` uses the manuscript's ResNet-50 architecture rather than the upstream default lineament configuration. The geometry branch alone is not the full SAM-Road graph baseline.

| Method | Implementation reference | Required external artifacts / setup |
|---|---|---|
| LDTR | [Included source](third_party/ldtr/README.md) | Historical-road weights and original baseline settings |
| SAM-Road | [Included source](third_party/sam_road/README.md) | Native topology training recipe, historical-data adaptation, and baseline weights |
| RelationFormer | [RGB source revision](https://github.com/suprosanna/relationformer/tree/5035d2a524e6402d410f970d9b60cc973cd3e16f) | Separate native environment/CUDA operator, historical-data adaptation, and weights |
| Sat2Graph | [Source revision](https://github.com/songtaohe/Sat2Graph/tree/7ca4a345a2ceae4c62c120661608a60e9e962be4) | Legacy TensorFlow environment, historical preprocessing/weights; upstream usage terms apply |
| SAM3 | [Source revision](https://github.com/facebookresearch/sam3/tree/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc) | Separate environment following that revision, gated weights, and actual road-type prompts |
| GPT5.5 | Service identified by the manuscript | Original model snapshot, exact prompt/training example, and actual responses |

These external reference revisions are not verified as the study's revisions. Native baseline training and proprietary-service inference have not been reproduced. Follow each implementation's README for its native commands; no missing weights, integrations, or service responses are simulated.

Executable adapters convert supplied predictions into the shared pixel-graph format:

```bash
python -m trasa.baselines --kind graph-json --input external/prediction.json --output external/converted.json --method relationformer --source-revision 5035d2a524e6402d410f970d9b60cc973cd3e16f
python -m trasa.baselines --kind vtp --coordinates normalized --input external/prediction.vtp --output external/converted_vtp.json --method ldtr
python -m trasa.baselines --kind sat2graph-pickle --trusted-pickle --axis-order yx --input external/prediction.p --output external/converted_sat2graph.json --method sat2graph
python -m trasa.baselines --kind mask --input external/road_probability.tif --output external/converted_sam3.json --method sam3
python -m trasa.baselines --kind polylines-json --input external/response.json --output external/converted_gpt.json --method gpt5.5
```

Mask conversion thresholds at 0.5, removes 8-connected components smaller than 20 pixels, applies Zhang skeletonization, removes endpoint spurs shorter than eight pixels, and clusters connected junction pixels. Polylines are clamped to patch bounds and rasterized at a recorded width (default 3). Invalid individual polylines are counted/discarded; malformed whole responses fail. VTP/pickle edges are already decoded; retained confidence 1 describes geometry rather than a neural score. Declare `xy`/`yx` for native pickle data and load trusted files only.

Create an external prediction index with actual provenance and either all held-out patch samples or complete maps:

```json
{
  "training_map_ids": [],
  "validation_map_ids": [],
  "example_map_ids": [],
  "source_revision": "actual_revision",
  "checkpoint_sha256": "actual_digest_if_available",
  "configuration": {},
  "samples": [{"id":"actual_test_patch_id", "graph":"converted.json"}]
}
```

Use empty ID lists only when none were used. For map predictions, replace `samples` with `maps:[{map_id,graph}]`. Graph paths resolve relative to the index. Evaluation requires exact held-out coverage and rejects test-map training, selection, or prompting examples:

```bash
python -m trasa.baseline_evaluate --manifest data/folds/fold_00.json --predictions external/fold_00/index.json --method relationformer --output results/relationformer/fold_00
python -m trasa.tables --reports runs/benchmark/*/fold_*/test/metrics.json results/relationformer/fold_*/metrics.json --output results/comparison_tables
```

## Implementation conventions

The manuscript specifies the main architecture and outer objective `L_node + L_edge + L_conn + 0.5 L_type`. Details below make the executable implementation explicit where original experiment settings are unavailable:

| Setting | Implemented convention / verification needed |
|---|---|
| Geometry architecture | SAM ViT-B and released 2×2 transposed-convolution decoder. SAM-Road's manuscript describes 3×3 kernels; TRASA does not resolve the discrepancy. |
| Geometry supervision | `intersection_only` by default; upstream endpoint-plus-intersection `degree_ne_2` is supported. The actual study setting needs verification. |
| Geometry schedule | Adam, BCE, LR 0.001, encoder factor 0.1, 100 epochs, batch 8; configurable recipe, without native TopoNet loss. Confirm original teacher training. |
| Graph input | ImageNet-normalized RGB plus unchanged road/keypoint probabilities; preserved RGB stem weights and mean-RGB extra filters. Geometry stays frozen. |
| Transformer inheritance | Four sampling points, FFN 1024, dropout 0.1, centered sine positions, and LDTR relation-token attention. |
| Hungarian cost | `5 × coordinate L1 − 3 × p(valid)`. Assignment is shared by all supervised heads. |
| Node loss | Weighted validity CE `[0.2,0.8]` plus matched coordinate L1 sum divided by reference-node count. Internal term weights default to 1; upstream LDTR uses 3/5. Confirm study weights. |
| Pair losses | Direct-edge CE weights `[1,20]`; unweighted connectivity CE. Nonempty graphs supervise matched-query pairs; empty graphs supervise absent unordered query pairs. |
| Five-hop targets | Shortest-path distances 1–5 inclusive, following TRASA's wording; pinned LDTR uses a strict `<5` test. |
| NTC reduction | Unweighted mean CE, configurable as `type_reduction: sum`. The manuscript's summed expression does not specify batch normalization. |
| Pair order | Random endpoint order in training; average both orders' logits for direct-edge inference. NTC and auxiliary connectivity do not replace validity/direct-edge decoding. |
| Run selection | Seed 10, fully trainable graph backbone, lowest validation objective. Verify original-run seed, freezing, and selection settings. |
| Evaluation | The explicit pixel protocol above, including approximate APLS and raw-query macro-F1; compare with the original evaluator. |

Checkpoints record mathematical settings, input/data provenance, and execution metadata. Older checkpoints retain their stored legacy position, pair-order, and relation-token defaults; loading them does not make their original training equivalent to the current configuration. GPU memory/runtime and CUDA parity have not been measured.

For a standalone execution check, run `python -m trasa.demo --output runs/synthetic_demo`. Guided checks additionally require real SAM initialization as shown in the README. Demos use smaller models and explicitly labeled synthetic data; published numerical reproduction remains dependent on the unavailable study artifacts and verification above.
