# Third-party sources

Exact source commits are in `sources.json`. Existing file-level copyright notices are retained.

- `src/trasa/vendor/ldtr`: selected models/ops files from knowledge-computing/LDTR. The upstream repository has no top-level LICENSE at the pinned commit. Its transformer/attention files carry Deformable DETR / DETR copyright and Apache-2.0 notices; those projects' license texts are included as `LICENSE_DEFORMABLE_DETR` and `LICENSE_DETR`. No new license is asserted for the entire upstream LDTR repository.
- `src/trasa/vendor/segment_anything/modeling`: the SAM modeling package from SAM-Road's exact `sam` submodule revision. Its Apache-2.0 license is included as `segment_anything/LICENSE` and `SAM_LICENSE`.
- `src/trasa/geometry.py`: geometry construction and decoder layout follow SAM-Road's released model; SAM-Road's MIT license is retained as `SAM_ROAD_LICENSE`.

TRASA-specific model composition, data contract, objectives, runtime, CLI, graph operations and tests are added here. The snapshot does not assign a new blanket license to unrelated upstream material or manuscript text.
