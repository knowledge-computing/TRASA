# TRASA

Paper implementation of dual-mask-guided, node-role-supervised road graph extraction from historical topographic maps. The model consumes RGB and frozen SAM-Road road/keypoint probabilities and predicts nodes and direct edges with an LDTR deformable transformer. A separate NTC head predicts no-node, endpoint, intermediate and intersection roles; an auxiliary connectivity head supervises paths of up to five hops.

This is the active implementation at `research/IMOLA/TRASA`, connected to `knowledge-computing/TRASA` on `main`. The old `research/TRASA` path now links to this same folder. IMOLA's data, workflows, baseline copies and documents are organized alongside it; the original TRASA checkout and all its local edits are preserved in `IMOLA/legacy/trasa-workspace-before-cleanup`. See [workspace layout](docs/WORKSPACE_LAYOUT.md). No map collections, training weights or derived data are bundled in the public code repository.

## Install and verify

Use Python 3.10. For CPU verification, create a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m pytest -q
```

`requirements-cpu.lock` additionally pins every installed transitive dependency from the fresh verification environment; use it in place of `requirements-cpu.txt` to repeat that exact package set. Verification used CPython 3.10.14 on Linux x86_64.

For GPU training, install the matching PyTorch 2.5.1 / torchvision 0.20.1 CUDA wheels for your host, then `python -m pip install -e '.[test]'`. No custom CUDA extension, Lightning, WandB, VTK or legacy IMOLA imports are required. The vendored LDTR attention uses its differentiable PyTorch reference kernel when the optional compiled operator is absent. GPU performance and CUDA determinism have not been verified in this implementation session.

## Paper settings

`configs/paper.yaml` uses 256×256 patches, 100 node queries plus one LDTR relation token, ImageNet ResNet-50, three encoder/decoder layers, eight attention heads, hidden dimension 256 and N=5. AdamW uses LR 1e-4, weight decay 1e-4, batch size 8, 100 epochs. Node/edge/connectivity weights are 1; NTC weight is 0.5. Node/edge thresholds are 0.5 and boundary merge tolerance is 5 pixels.

The RGB filters in the five-channel stem are copied exactly from ImageNet weights. Both mask-channel filters use their RGB-filter mean. RGB receives ImageNet normalization; mask probabilities stay in [0,1]. The geometry module is frozen, held in evaluation mode and detached. It receives only RGB; reference masks are never accepted as a substitute for predicted guidance.

Read [paper-to-code mapping](docs/PAPER_IMPLEMENTATION.md) before claiming experimental reproduction: the paper does not specify every architecture, geometry-checkpoint or evaluation detail. This code implements its method, but has not reproduced the result tables.

## Data contract

Supply a version-1 JSON manifest. Image, graph and mask paths resolve relative to that file. Example:

```json
{
  "schema_version": 1,
  "guidance_provenance": {
    "checkpoint_sha256": "REPLACE_WITH_64_CHARACTER_SHA256",
    "channel_order": ["road", "intersection"],
    "keypoint_semantics": "degree_ne_2",
    "training_map_ids": ["training_sheet_A"]
  },
  "samples": [
    {"id": "A_0_0", "map_id": "training_sheet_A", "split": "train", "image": "images/A_0_0.png", "graph": "graphs/A_0_0.json", "guidance": "masks/A_0_0.npz"},
    {"id": "B_0_0", "map_id": "validation_sheet_B", "split": "val", "image": "images/B_0_0.png", "graph": "graphs/B_0_0.json", "guidance": "masks/B_0_0.npz"},
    {"id": "C_0_0", "map_id": "test_sheet_C", "split": "test", "image": "images/C_0_0.png", "graph": "graphs/C_0_0.json", "guidance": "masks/C_0_0.npz"}
  ]
}
```

Graph annotations are `{"nodes": [[x,y], ...], "edges": [[i,j], ...]}` in **pixel (x,y)** coordinates, from 0 to 255. Optional `types` must agree with the patch graph degree. Duplicate/reversed edges are deduplicated; self loops, isolated reference nodes, invalid coordinates and >100 reference nodes are rejected rather than silently discarded. Crop/clip road graphs before computing roles: patch exits have degree one.

NPZ masks have named `road` and `intersection` float arrays of shape 256×256. Use probabilities, not thresholded masks. The upstream keypoint mask is trained on **degree != 2**, which includes endpoints. Declare `degree_ne_2` for upstream checkpoints or `intersection_only` only for checkpoints actually trained that way. Guidance provenance must identify training maps and the checkpoint hash; the loader rejects held-out maps in geometry training and any sheet crossing train/val/test boundaries. Empty `training_map_ids` is appropriate only for externally pretrained geometry with no benchmark-sheet training.

For the eight-map benchmark, make eight fold manifests with six training maps, one validation map and one held-out test map each. The manuscript does not identify the eight sheets or exact fold schedule, so no invented split is shipped. Any benchmark fine-tuning of geometry must be fold-specific and then frozen.

## Cache frozen guidance

Use a **trained SAM-Road checkpoint**, with matching variant, patch resolution and decoder architecture. A vanilla SAM checkpoint lacks the road decoder and is rejected. The default is SAM-Road's ViT-B plus transposed-convolution decoder; `--sam-decoder` selects its alternate SAM decoder. LoRA checkpoints are explicitly rejected.

```bash
python -m trasa.cache_guidance --manifest /path/fold-input.json \
  --checkpoint /path/samroad-fold.ckpt --output /path/new-guidance-cache \
  --training-map-ids training_sheet_A --keypoint-semantics degree_ne_2 --device cuda
```

This writes a new manifest with predicted guidance and its checkpoint provenance. It does not create reference annotations. Images/graphs stay in their original locations.

## Train later

```bash
python -m trasa.train --config configs/paper.yaml \
  --manifest /path/fold.json --output /path/new-run --device cuda
```

Use `--geometry-checkpoint /path/model.ckpt` for online frozen geometry instead of cached masks. Its hash must match the manifest. `--ablation ldtr`, `ldtr-sam`, `ldtr-ntc` or `trasa` selects the four paper variants within the same implementation.

`last.pt` contains model, optimizer, epoch, Python/NumPy/Torch/CUDA RNG states and shuffled-loader generator state. `best.pt` is selected by validation loss. Resuming requires identical configuration and actual data-content hashes:

```bash
python -m trasa.train --config configs/paper.yaml \
  --manifest /path/fold.json --output /path/run --device cuda --resume /path/run/last.pt
```

Checkpoints contain trusted Python/NumPy training state; load only your own or trusted files. CUDA `grid_sample` backward can be nondeterministic: seeding and deterministic settings cannot guarantee bitwise A100 reproduction.

## Inference and evaluation

For one patch with cached geometry:

```bash
python -m trasa.infer --checkpoint /path/best.pt --image /path/patch.png \
  --guidance /path/masks.npz --output /path/new-graph.json
```

For a whole map, omit `--guidance` and provide `--geometry-checkpoint` if the TRASA model was trained using cached masks. Embedded online geometry is restored automatically. Patch size must match the geometry checkpoint. Use `--stride 256` by default; overlap behavior is described in the mapping document. Optional `--affine x0 a b y0 d e --crs EPSG:...` transforms pixel centers through a GDAL affine into geospatial coordinates. No reference graph is required for inference.

Graphs contain pixel/geospatial nodes, undirected edges, predicted role attributes, node/edge confidence and source hashes. NTC classes and N-hop predictions do not gate or repair final edges.

```bash
python -m trasa.evaluate --reference /path/corrected-map-graph.json \
  --prediction /path/merged-pixel-graph.json --output /path/new-metrics.json
```

The evaluator computes buffered completeness/correctness/quality, intersection precision/recall/F1 and degree accuracy, matched-pair connectivity, split components, spurious branches and symmetric APLS. It expects already merged **pixel-coordinate** graphs. Undefined IDA/connectivity denominators produce JSON null and eligible counts are retained. APLS defaults (5-pixel control spacing, at most 500 controls, seed 10) are disclosed implementation choices; report them with results. `type_macro_f1` operates on raw matched queries including no-node; it cannot be computed faithfully from filtered final graph attributes alone.

## Next steps

See [TODO.md](TODO.md) for the short training/data-production checklist and [source provenance](docs/sources.json) for exact upstream commits.
