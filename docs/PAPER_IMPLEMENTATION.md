# Paper-to-code mapping and unresolved experimental details

Primary specification: the supplied `sonya_sigspatial_2026.zip`, especially `04_method.tex`, `05_experiment.tex` and `03_dataset.tex`. Text sources are preserved locally under `docs/paper/` and excluded from the code publication snapshot; figures remain in the original archive, so that directory alone is not a compilable paper bundle. `sources.json` records the archive SHA256 and exact upstream revisions.

## Implemented requirements

| Paper requirement | Implementation |
|---|---|
| Frozen SAM-Road geometry only, no TopoNet/NMS final predictor | `geometry.SAMRoadGeometry` |
| RGB + road + intersection probabilities; resize without thresholding | `model.TRASA.forward` |
| ImageNet ResNet-50, RGB filters preserved, two RGB-mean filters | `model.guidance_stem` |
| LDTR multi-scale deformable graph encoder/decoder | vendored `ldtr/deformable_detr_2D.py` + `model.TRASA` |
| 100 queries; three encoder and decoder layers; dim 256; eight heads | `configs/paper.yaml` |
| Separate valid-node score and coordinate regression | `valid_head`, `coordinate_head` |
| Separate four-class NTC MLP, unmatched query = no-node | `type_head`, `losses.GraphCriterion` |
| Hungarian matching shared by geometry/edge/connectivity/type supervision | `GraphCriterion.match` |
| N=5 shortest-path supervision; auxiliary head only during training | `graph.connectivity_labels`, `connectivity_head` |
| Node/edge/connectivity weights 1, NTC weight .5; no mask loss | `GraphCriterion.forward` |
| Direct validity/edge threshold .5; NTC remains an attribute | `graph.decode_graphs` |
| Merge nearby boundary nodes at 5 pixels; deduplicate edges | `graph.merge_patch_graphs` |
| Optional spatial transform of output graph | `graph.georeference_graph` |
| AdamW 1e-4, weight decay 1e-4, batch 8, 100 epochs | `train.main` |
| Complete-map train/validation/test split, fold-specific geometry provenance | `data.PatchDataset` |
| Four ablations | `train --ablation` |
| Map-level geometry, topology and routing evaluation | `metrics`, `evaluate` |

## Upstream analysis and compatibility

The official [LDTR repository](https://github.com/knowledge-computing/LDTR) supplies the deformable transformer, Hungarian cost structure, node CE/L1, pairwise relation head and N-hop CE formulation. It predicts binary validity separately from coordinates. The local legacy `LDTR_mine` instead modified this into a four-class validity/type head; using that as-is would conflate NTC with the paper's independent validity task. It also contains CPU-incompatible device handling and fragile empty-graph branches. The clean losses avoid those paths.

The standalone [LDTR paper](https://doi.org/10.1007/978-3-032-04617-8_3) was available only as a publisher abstract/preview during this work. Its full text was not read. The authors' open [DIGMAPPER account, section 2.3](https://arxiv.org/html/2506.16006v1) supports the deformable-attention/N-hop formulation. Detailed implementation decisions here come from the supplied TRASA manuscript and pinned official LDTR code, not an asserted full-text verification.

The [SAM-Road paper](https://arxiv.org/html/2403.16051v3) and [official code](https://github.com/htcr/sam_road) split dense geometry from topology. The upstream model normalizes 0–255 RGB pixels, then returns **[keypoint, road]** masks. TRASA's input order is **[RGB, road, intersection]**. The adapter explicitly swaps mask channels. It does not import Lightning, WandB, torchmetrics, the upstream `model` module or TopoNet.

SAM-Road's `sam/segment_anything` directory is a Git submodule, absent after a nonrecursive clone. That caused the interrupted session's FileNotFoundError. It was resolved by initializing `sam` at its pinned commit, using HTTPS for retrieval, then vendoring only its modeling package and license. Runtime imports are repository-relative; no `/tmp` or sibling-source paths are required.

SAM-Road's default code uses kernel-2 transposed convolutions despite its paper describing kernel-3. We follow the released **code** for checkpoint compatibility. Its alternate prompt-free SAM decoder is also supported and must be selected explicitly. Its keypoint label is degree!=2 (including endpoints). The TRASA draft calls this an intersection mask, but does not resolve that difference; checkpoint/cache metadata records the actual semantics.

Vendored LDTR attention uses the original PyTorch reference implementation when its optional CUDA extension is unavailable. The sampling/weighting formulation is preserved and differentiable. Explicit `meshgrid(indexing="ij")` fixes a compatibility warning without changing coordinates. The rest of the decoder is retained at the pinned revision.

## Choices the manuscript does not determine

| Missing/ambiguous detail | Implemented decision |
|---|---|
| N-hop inclusive vs official code's hardcoded `distance < 5` | Manuscript takes precedence: `1 <= distance <= 5`. Regression test distinguishes five from six hops. |
| Feature levels, sampling points, FFN, dropout, relation-token image attention | LDTR defaults: four levels, four points, FFN 1024, dropout .1, one relation token with ordinary decoder cross-attention (`rln_attn=False`). |
| ImageNet weight version and RGB normalization | torchvision `IMAGENET1K_V1`, ImageNet mean/std; masks remain probabilities. |
| Separate node validity and NTC heads | Binary validity + two-coordinate MLP + separate four-class type MLP. Unused upstream box width/height outputs are omitted because they have no active paper objective. |
| Loss reduction and inner class balancing | Node loss = weighted validity CE + L1 sum per matched node; inherited LDTR validity weights [.2,.8], edge weights [1,20], connectivity/NTC unweighted CE. Outer weights follow TRASA. NTC CE averages queries/batch; the draft writes a sum without normalization details. |
| Undirected pair orientation | Average logits from both endpoint orders at training and inference. Upstream randomizes training order and averages inference; averaging training too enforces symmetry without random supervision noise. |
| Empty graphs and incomplete matching | Empty targets supervise all query pairs as absent. Oversized targets are rejected, never truncated. No fabricated random negative graph. |
| Training schedule, clipping, augmentation, model-selection metric | Constant AdamW LR, no extra clipping/augmentation; select lowest validation objective. These are exposed choices, not recovered experiment metadata. |
| Geometry variant/checkpoint/fine-tuning schedule | ViT-B/default decoder by default, explicit alternative options. No invented checkpoint or training recipe; geometry must already be trained and frozen, with per-fold provenance. |
| Boundary band, cluster coordinates and duplicate-edge confidence | 5-pixel band; bounded greedy cross-patch clusters, mean coordinates, highest-confidence type, max duplicate-edge confidence. No transitive cluster collapse; no same-patch node merges. |
| Overlapping patch interiors | Only boundary nodes merge, per literal manuscript. Overlapping interiors can duplicate geometry; default stride=patch size. Border coverage uses an overlapping final tile if the image dimensions are not multiples of patch size. More elaborate overlap stitching is future work. |
| Graph-to-world pixel convention | Normalized coordinate 1 maps to W−1/H−1; GDAL transform uses pixel centers (+.5). |
| Exact APLS settings | Symmetric control insertion onto edges, length-relative shortest-path penalty, harmonic mean of both directions; 5-pixel subdivision, <=500 deterministic controls/direction, no minimum path-length exclusion. This is not validated against unpublished TRASA evaluation outputs. |
| Node/intersection matching tie handling | Maximize valid one-to-one assignments within 5 pixels, then minimize distance. No many-to-one matches. |
| Unmatched branch definition and metric empty cases | Maximal degree-2-interior path; count if >8 pixels and its entire geometry has no buffered reference support. Null for undefined matched-intersection degree accuracy or connectivity. |
| Exact benchmark identities and fold schedule | User-provided manifests; no inferred splits or uncorrected TIGER data. |

## Reproducibility boundaries

The implementation and synthetic CPU tests verify method wiring, gradients, geometry import/checkpoint contracts, correct target remapping, inference rules, data leakage checks and serialized state restoration. A separate fresh virtual environment verifies packaging and pinned dependencies. No real training run, model accuracy, A100 throughput, GPU kernel equivalence or paper-table reproduction is claimed.

Each run stores configuration, manifest and actual training/validation file-content hashes, geometry provenance, optimizer, epoch and RNG states. Cached probability masks must be generated with the declared checkpoint; cache provenance is an explicit data contract rather than a cryptographic proof that a third-party file was generated correctly. Do not mix geometry trained on held-out maps into a fold.

The implementation does not train SAM-Road, reconstruct the eight-map benchmark, run the six comparison models, create IMOLA's US1884+ data, estimate production/application uncertainty or calibrate confidence. These are later experimental/data-production tasks. Node-role labels are topology labels; they are not road-symbol classes.
