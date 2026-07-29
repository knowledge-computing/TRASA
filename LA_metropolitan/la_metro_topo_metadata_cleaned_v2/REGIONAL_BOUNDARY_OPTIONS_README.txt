Regional boundary options added.

Main file:
  la_metro_topo_metadata_cleaned_v2/regional_boundary_options.gpkg

Layers:
  option_a_la_county
  option_b_la_msa_current
  option_c_la_csa
  option_d_csa_component_msas
  csa_component_counties
  all_boundary_options

Also added these layers to:
  la_metro_topo_metadata_cleaned_v2/la_metro_topo_map_editions_clean.gpkg

Useful tables:
  summary_tables/regional_boundary_options_area.csv
  summary_tables/current_inventory_overlap_with_boundary_options.csv

Useful plot:
  plots/regional_boundary_options_overview.png

Interpretation:
  Option B is the boundary used for the current collected inventory.
  Option C, the CSA, is much larger. If the team chooses the CSA as the expansion boundary,
  the TNM metadata collection should be rerun using the CSA geometry so that Riverside,
  San Bernardino, and Ventura-area map sheets are not missed.
