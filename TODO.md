# Next tasks

- Identify the eight annotated map sheets and freeze eight fold manifests (6 train / 1 validation / 1 test), with historically corrected graphs and five road-symbol subsets.
- Select or fine-tune a SAM-Road geometry checkpoint for each fold; confirm mask semantics, patch resolution and checkpoint hashes. Obtain the full LDTR paper to finish the remaining source audit.
- Run TRASA and its three ablations for 100 epochs on the A100; validate graph merging and metric settings, then evaluate held-out maps and report fold means. No paper result has been reproduced yet.
- Begin IMOLA data production separately: plan the 1:62,500 series and sampled 1:24,000/1:250,000 series; retain scale/year/CRS, road-symbol attributes, provenance and raster/vector linkage. Define calibrated processing, production and application uncertainty before US1884+ publication.
