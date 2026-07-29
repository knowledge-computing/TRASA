SAM3 sample dataset for Leeje
================================

This sample was created from the Beverly Hills 1995 1:24000 map.

reference/
  Raw 128x128 map patches used as SAM3 reference examples.
  Each reference patch has a matching *_lines.json file.
  The JSON stores road-line coordinates in patch-local pixel coordinates.

inference_patches/
  Raw 128x128 map patches used as SAM3 inference inputs.
  These are not overlaid and do not include JSON annotations.
  In this version, inference patches are selected from diverse road-containing areas across the map.
  They are also kept away from the map boundary and away from the selected reference patches.

debug_overlays/
  Visualization only.
  These images show the road-line coordinates drawn on top of the raw reference patch.
  These are not model inputs.

manifest.json
  Summary of generated files, including row/column offsets.
