# Upstream implementations

These pinned snapshots provide the original LDTR and SAM-Road source, including SAM-Road's SAM dependency. Revisions, file checksums and attribution are in `../docs/THIRD_PARTY.md` and `../docs/upstream-files.json`.

- `ldtr/`: graph model, Hungarian matching, graph losses, dataset generation, training, inference, evaluation and optional CUDA attention extension.
- `sam_road/`: geometry/topology model, dataset handling, training, sliding-window inference and APLS/TOPO evaluators.
- `sam_road/sam/`: SAM builders, image/prompt/mask encoders, utilities and examples.

The `trasa` package uses adapted components with its own pinned runtime dependencies. To execute an original upstream baseline, follow that snapshot's README and install its separate dependencies. Baseline configuration paths and preprocessing settings must be adapted to the same benchmark folds. Checkpoint files and datasets are not bundled.
