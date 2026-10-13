# Validation

The automated tests cover the full-size graph model's forward/backward pass, all four ablations, Hungarian label remapping, five-hop targets, frozen geometry, upstream mask ordering, checkpoint/RNG restoration, data leakage checks, graph merging and evaluation. A synthetic train/resume/infer/evaluate workflow verifies the command-line entry points.

Run:

```bash
python -m pytest -q
python -m pip check
```

Tests use synthetic inputs and do not establish benchmark accuracy. Reproducing the reported results additionally requires the annotated map benchmark, fold-specific geometry weights, graph training and validation-selected evaluation settings.

The verified CPU configuration (Python 3.10, PyTorch 2.5.1+cpu) passes **35 tests**, including compressed VTP conversion, geometry training/resume/cache provenance, corrected pixel-label preservation, eight-map fold generation and held-out map aggregation. Dependency checks and wheel builds pass. Upstream snapshot tests verify all 205 included source/reference files.
