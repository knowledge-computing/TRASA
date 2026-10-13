# Method and implementation

| Component | Code |
|---|---|
| SAM-Road geometry and probability guidance | `geometry.py` |
| Fold-specific geometry training | `train_geometry.py` |
| JSON/VTP data conversion and complete-map folds | `prepare_data.py`, `folds.py` |
| Five-channel ResNet-50 and LDTR deformable graph transformer | `model.py` |
| Hungarian matching, node/edge/N-hop objectives, NTC | `losses.py` |
| Direct graph decoding, boundary merging, georeferencing | `graph.py` |
| Paper configuration and ablations | `configs/paper.yaml`, `train.py` |
| Map-level metrics | `metrics.py`, `evaluate.py` |
| Held-out map prediction and fold aggregation | `predict_dataset.py`, `benchmark.py` |

RGB filters are initialized from ImageNet ResNet-50; the two added channels use the mean RGB filter. RGB receives ImageNet normalization and guidance stays in [0,1]. SAM-Road receives RGB only and stays frozen during graph training. Its output order [keypoint, road] is converted to [road, keypoint].

Binary node validity and the four-class NTC head are separate. Hungarian assignments align every supervised head; unmatched queries receive no-node labels. Direct edge and auxiliary connectivity heads average both endpoint orders. N-hop targets include shortest-path distances from one through five hops. Final graphs use node/edge confidence thresholds of 0.5; NTC remains an attribute and the connectivity head is used only during training.

The default SAM-Road decoder follows its released transposed-convolution code for checkpoint compatibility. Upstream keypoint labels include degree-one endpoints and intersections (degree != 2). Record the actual label semantics when training or selecting geometry; do not relabel these probabilities as intersection-only output.

The graph implementation inherits four feature levels, four sampling points, FFN dimension 1024, dropout 0.1 and one relation token. Node loss combines weighted validity CE ([0.2,0.8]) and coordinate L1; edge CE uses inherited [1,20] class weights; connectivity and NTC use unweighted CE. Outer paper weights are 1/1/1/0.5. Undirected training averages endpoint-order logits; empty graphs supervise absent query pairs. The manuscript does not specify every internal reduction, checkpoint, fold identity or evaluation sampling parameter; these settings are explicit in the configuration and reproduction instructions.

The PyTorch deformable-attention reference kernel is available without a compiled CUDA extension. CUDA grid-sampling backward may be nondeterministic. Report hardware, dependencies and evaluation protocol with experiments.
