# TRASA

Implementation of **TRASA** for road graph extraction from topographic maps. Frozen SAM-Road road and keypoint probabilities guide an LDTR deformable transformer, with auxiliary five-hop connectivity and four-class node type supervision.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pytest -q
```

For GPU training, install matching PyTorch 2.5.1 and torchvision 0.20.1 CUDA wheels, then `pip install -e '.[test]'`.

## Training and inference

```bash
python -m trasa.train --config configs/paper.yaml --manifest data/fold.json --output runs/trasa --device cuda
python -m trasa.infer --checkpoint runs/trasa/best.pt --image data/patch.png --guidance data/guidance.npz --output results/graph.json
python -m trasa.evaluate --reference data/reference.json --prediction results/graph.json --output results/metrics.json
```

Use `--ablation ldtr`, `ldtr-sam`, `ldtr-ntc`, or `trasa` to select a paper variant. The paper configuration uses 256×256 patches, 100 node queries, ResNet-50, three encoder/decoder layers, dimension 256, and eight attention heads.

See [reproduction instructions](docs/REPRODUCING.md) for data preparation, geometry training, fold manifests, checkpoint provenance, and map-level evaluation. Dataset annotations and trained weights are supplied separately; synthetic tests verify execution, not the reported benchmark accuracy.

## Upstream code

Pinned LDTR, SAM-Road, and SAM source snapshots are included in `third_party/`. The TRASA runtime uses the adapted components in `src/trasa/vendor/`; the complete snapshots retain the upstream training, preprocessing, inference, and evaluation implementations. Revisions and attribution are documented in [third-party sources](docs/THIRD_PARTY.md).
