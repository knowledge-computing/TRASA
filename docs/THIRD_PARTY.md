# Third-party source code

| Source | Included snapshot | Revision |
|---|---|---|
| [LDTR](https://github.com/knowledge-computing/LDTR) | `third_party/ldtr/` | `eaf5c9f3e66ca0181ea98fe4378420107b8af296` |
| [SAM-Road](https://github.com/htcr/sam_road) | `third_party/sam_road/` | `04b67c503f120ecff2b2dc5a2041ea6d300e306a` |
| [SAM](https://github.com/htcr/segment-anything-road) | `third_party/sam_road/sam/` | `6fdee8f2727f4506cfbbe553e23b895e27956588` |

The snapshots retain upstream source, configurations, training/inference code, preprocessing, evaluation and documentation. Generated Python bytecode and package-build metadata are excluded. SAM-Road's two SAM submodule names reference the same included SAM tree. `upstream-files.json` records SHA256 for every included source/reference file; tests verify these contents.

TRASA imports the adapted transformer/attention and SAM modeling components from `src/trasa/vendor/`, avoiding dependencies on the upstream experiment frameworks during normal execution. Adaptations include a differentiable attention fallback, explicit meshgrid indexing, a five-channel backbone, separate NTC, corrected empty-graph handling, frozen geometry and portable experiment entry points. Complete upstream snapshots provide the original baseline implementations and a reference for these changes; they are not imported implicitly.

File-level copyright notices and upstream license texts are retained. SAM-Road is MIT licensed; SAM and the Deformable DETR/DETR components carry their included Apache-2.0 licenses. The pinned LDTR repository has no top-level license; its existing notices remain unchanged. No blanket license is assigned to unrelated upstream material.
