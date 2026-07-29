# Regional Scope Comparison

This folder is exploratory and separate from the final LA metro statistics.

It compares the number of topo map editions intersecting several possible study boundaries:

1. City of Los Angeles
2. Los Angeles County
3. LA–Long Beach–Anaheim MSA/current inventory boundary
4. Greater LA CSA
5. Greater LA CSA outside the current LA MSA

Important:
- This does not modify `la_metro_topo_metadata_cleaned_v2/`.
- This reruns TNM metadata collection over the Greater LA CSA extent.
- Use this only for meeting discussion about possible future expansion scope.
- The advisor-requested LA metro statistics remain in `la_metro_topo_metadata_cleaned_v2/`.

Main outputs:
- `summary_tables/boundary_option_map_counts.csv`
- `summary_tables/boundary_option_counts_by_product_group.csv`
- `summary_tables/boundary_option_counts_by_scale.csv`
- `summary_tables/additional_maps_greater_la_csa_outside_current_msa.csv`
- `plots/boundary_option_unique_map_editions.png`
- `plots/boundary_option_by_product_group.png`
- `plots/boundary_option_by_scale_heatmap.png`
- `greater_la_csa_map_editions_with_boundary_flags.csv`
- `greater_la_csa_map_editions_comparison.gpkg`

