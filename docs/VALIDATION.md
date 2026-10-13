# Implementation validation

Verified on Linux x86_64, CPython 3.10.14, PyTorch 2.5.1+cpu and torchvision 0.20.1+cpu. A fresh virtual environment was installed from `requirements-cpu.txt`; its complete package set is pinned in `requirements-cpu.lock`.

```text
python -m pytest -q
24 passed in 18.18s

python -m pip check
No broken requirements found.
```

The same 24 tests also passed in the pre-existing SAM-Road environment. Tests cover:

- Full 256×256, 100-query, dim-256, 3+3-layer TRASA forward/backward, gradients through all graph heads and the two added input channels, detached guidance and correct outer loss weights.
- All four guidance/NTC ablation combinations and empty-image losses.
- Five-hop inclusion, six-hop exclusion, disconnected pairs, self-pair exclusion and patch-degree node roles.
- Hungarian permutation applied consistently to node validity, NTC, edges and connectivity.
- Undirected pair symmetry and final graphs constructed without the connectivity head or NTC gating.
- Geometry checkpoint loading, upstream mask channel reordering, RGB normalization, evaluation mode, no geometry gradients and rejection of vanilla SAM/random guidance generators.
- An actual SAM ViT-B geometry forward pass at reduced 32×32 resolution with synthetic weights. This verifies vendored architecture/imports, not pretrained extraction accuracy.
- Map-level split leakage and geometry-training leakage rejection; actual image/graph/mask content fingerprints for resume validation.
- Optimizer update, checkpoint round trip, Python/NumPy/Torch/data-loader RNG restoration and unchanged inference after reload.
- Synthetic train/resume/infer/evaluate CLI workflow, including refusal to resume after a cached mask changes.
- Boundary-only cross-patch merging, duplicate-edge removal, empty merges, GDAL pixel-center transform and tile coverage.
- Geometric metrics, perfect/differently subdivided graph APLS, split/spurious-branch penalties, intersection degree accuracy and threshold-constrained one-to-one matching.

`compileall` passed for source/tests. CLI help loads without legacy imports. A distribution wheel built successfully. The optional SAM decoder also passed a two-image geometry forward check. The original sibling TRASA checkout's complete Git status was compared byte-for-byte before/after implementation. During subsequent folder cleanup it was relocated intact into IMOLA's `legacy/trasa-workspace-before-cleanup`; all 264 non-Git file checksums and that same Git status were verified after relocation. Its former path now links to the active implementation.

After reorganization, the suite was rerun through the old `research/TRASA` compatibility path: **24 passed in 10.81s**, with no broken package requirements. All 36 filesystem moves, compatibility aliases and four script data-path fixes were checked. Pytest now retains only failed temporary fixtures, preventing successful checkpoint tests from filling the server's small `/tmp` filesystem.

No real benchmark training or IMOLA data production was run. Full LDTR paper access, actual geometry checkpoint selection, exact fold identities, A100 verification and validation against the paper's original evaluator remain experimental prerequisites, as described in `PAPER_IMPLEMENTATION.md` and `TODO.md`.
