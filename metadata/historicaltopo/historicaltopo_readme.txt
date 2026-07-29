Metadata Overview
Topics .... USGS Historical Topographic Map Collection (HTMC)
Contents .. Product metadata and summary storage estimates.
Formats ... Plain-text and comma-separated value (CSV) files.

Available Series
Historical Topographic Maps
historicaltopo.zip
Products from the Historical Topographic Map Collection (HTMC).
Scanned paper topographic map sheets originally published by the USGS and other 
federal agencies between 1884 and 2006. 
Products are distributed as raster images in TerraGo GeoPDF format. 
Including maps with original printed boundaries not formerly registered in GNIS.

Note: Historical editions may also include maps with content or names that are 
not directly associated with the modern Geographic Names Information System.

Data elements describing the content of historical maps were collected manually 
through visual inspection of the scanned map images and are subject to errors 
inherent to this method. All recorded information is subject to inaccuracies of 
the original print version of every map product. 

This file is intended to provide a succinct summary of the information found 
in the associated report. Some fields may have further expanded definitions to 
account for other printed variations. This document does not attempt to define 
any existing or established terms outside the scope of this report. Refer to 
related public specifications or metadata glossaries for more details.


Column Descriptions
Availability may vary by series.

scan_id .................. Central primary key for all historical map records. 
                           A sequence generated integer assigned to a digital 
                           image and the paper map sheet from which it was 
                           created.
                           Type: bigint 
                           Domain: positive non-zero integers

product_inventory_uuid ... Universally unique identifier for a product within 
                           the inventory system. Associates a spatial dataset 
                           to content within a product class or series. 
                           Retained between replacements for same edition.
                           Note: Designed to be re-usable if the products are 
                           re-issued under different headings. Retention for 
                           non-cell-based product may be impacted by changes 
                           in reported spatial boundary.
                           Registered as an identifier in ScienceBase.
                           Type: UUID
                           Domain: RFC-4122

series ................... General collection or category of maps.
                           Type: text
                           Domain: US Topo, HTMC.

edition .................. Indicates whether this map is the latest 
                           representation of a map extent or cell.
                           US Topo only: At most one current product may be 
                           defined for each 7.5-Minute cell unit.
                           Type: text
                           Domain: Current, Historical

map_name ................. The printed map title.
                           Printed names on historical maps may vary from the 
                           official current geographic names record.
                           Type: text
                           Domain: place names and printed sheet titles

primary_state ............ The current official GNIS state name. 
                           Type: text 
                           Domain: common names of U.S. states and 
                           territories, provinces, and other political regions.
                           Note: registered values may vary from printed text 
                           in historical printed maps.

westbc ................... The west bounding coordinate of the product spatial 
                           extent, measured in angle of longitude.
                           Type: numeric
                           Domain: -180.0, 180.0 (decimal degrees)

eastbc ................... The east bounding coordinate of the product spatial 
                           extent, measured in angle of longitude.
                           Type: numeric
                           Domain: -180.0, 180.0 (decimal degrees)

northbc .................. The north bounding coordinate of the product spatial 
                           extent, measured in angle of latitude.
                           Type: numeric
                           Domain: -90.0, 90.0 (decimal degrees)

southbc .................. The south bounding coordinate of the product spatial 
                           extent, measured in angle of latitude.
                           Type: numeric 
                           Domain: -90.0, 90.0 (decimal degrees)

geom_wkt ................. Well-Known Text representation of the standardized 
                           geometry. All polygons expressed in WGS84 SRS.
                           Type: text

date_on_map .............. The primary edition year of the map product. 
                           Also reported as "publication date".
                           Type: integer
                           Domain: calendar years, 1897-present

imprint_year ............. Year the original map was completed or imprinted for 
                           reproduction. Multiple physical copies may be 
                           produced over time. 
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available

survey_year .............. Year of the latest field survey.
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available
                           
field_check_year ......... Year the map content was last checked in the field.
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available

edit_year ................ Year of the last revision of change in content.
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available

aerial_photo_year ........ Year of latest aerial imagery used in production.
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available

photo_inspection_year .... Year of the last photo inspection.
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available

photo_revision_year ...... Year when newer photos were used to make revisions.
                           Type: integer
                           Domain: calendar years, 1897-present
                           Default: Null/Empty, when no data is available

metadata_date ............ Last update to the distributed metadata record. 
                           Type: date
                           Domain: standard calendar dates
                           Note: Data distribution activities are not 
                           reflective of changes to content of the original 
                           map product

print_releases ........... Printed label designating the purpose or intent for 
                           this print edition.
                           Type: text
                           Domain: All printed values meeting this criterium,
                           "advance", "provisional", "interim", etc.
                           Representation: comma-delimited list
                           Default: Standard, no apparent designations to 
                           indicate otherwise.
                           
variations ............... Added spatial data layers or distinguishing 
                           characteristics of the printed map. 
                           Typically designated in the positive for the 
                           availability of new information in print form.
                           Type: text
                           Domain: All applicable printed visual qualities
                           "Woodland", "ShadedRelief", "Planimetric", etc.
                           Representation: comma-delimited list
                           Default: "None"; when no variations have been 
                           recorded.

publishers ............... Federal publishers identified in print for their 
                           contribution to the production of the map product.
                           Other cooperating agencies, including local or state 
                           governments, are not included.
                           Type: text
                           Domain: Organization names and abbreviations
                           "USGS", "USFS", "BIA", "MIL" (Military), etc.
                           Representation: comma-delimited list

languages ................ Primary language(s) used on the printed map. 
                           Noted: Labeled in plural as more than one language 
                           may be present in the data source. 
                           Does not cover local or regional place names. 
                           Type: text
                           Domain: Common languages used in map production.
                           Default: "Undetermined", in cases where the data may 
                           not have been available in the final metadata record.
                           Representation: comma-delimited list of names if 
                           more than one language was recorded for that sheet.

hdatum_printed ........... Original printed horizontal datum given for the 
                           spatial coordinate system. 
                           Type: text
                           Domain: Recognized spatial reference systems,
                           "NAD27", "NAD83", "Regional", "Old Hawaiian", etc.
                           Default: "Unstated"; when a printed value cannot be 
                           located on the map sheet.
                           Note: All digital products are prepared using 
                           standard official coordinate systems with applicable 
                           toWGS84 translation arrays. All values are shown as 
                           printed or with a close approximation to accurately 
                           reflect the original material. In many cases, NAD27 
                           is used as a stand-in for georeferencing when other 
                           data is not available.

projection_printed ....... Original printed projection used with the map.
                           Type: text
                           Domain: Recognized map projection types,
                           "Polyconic", "Transverse Mercator", "Lambert 
                           Conformal Conic", etc.
                           Default: "Unstated"; when a printed value cannot be 
                           located on the map sheet.
                           Note: In many cases, Polyconic is used for purposes 
                           of processing digital georeferenced images.

map_scale ................ The quotient of ratio for the scale to which the 
                           spatial was reproduced within the map frame in 
                           relation to the printed page.
                           Type: integer
                           Domain: varies by series
                           US Topo Only: {24000, 25000, 20000}

gnis_cell_id ............. The ID of the cell record matching the map frame 
                           boundary in the geographic name information system 
                           cells database.
                           Bounding coordinates of cell-based maps are expected 
                           to match those associated with this cell.
                           Type: bigint
                           Domain: integers
                           Default: -1; when not available for maps without 
                           a matching extent in the cells database.

gnis_cell_name ........... The current official GNIS cell or printed name.
                           Printed names on historical maps may vary from the 
                           official current geographic names record.
                           Type: text
                           Domain: place names and printed sheet titles
                           Default: "NA"; when Not Applicable. 

gnis_primary_state ....... The current official GNIS state name. 
                           Type: text 
                           Domain: common names of U.S. states and 
                           territories, provinces, and other political regions.
                           Note: registered values may vary from printed text 
                           in historical printed maps.
                           Default: "NA"; when Not Applicable. 

grid_size ................ The standard spatial dimension of the GNIS cell unit.
                           Type: text
                           Domain: GNIS cell grids
                           US Topo: 7.5 x 7.5 Minute
                           Default: "NA"; when Not Applicable. 

cell_type ................ Relation of the map spatial extent to the fixed 
                           GNIS cell grid.
                           Type: text
                           Domain: GNIS cell types
                           Default: "NA"; when Not Applicable. 

state_list ............... List of state abbreviations based off the geographic 
                           names feature database.
                           Type: text 
                           Domain: short code abbreviations of U.S. states and 
                           territories, provinces, and other political regions.
                           Default: "NA"; when Not Applicable. 

county_list .............. List of counties or equivalent regions that intersect 
                           the boundary of the map according to the geographic 
                           names feature database.
                           Type: text
                           Domain: government units
                           Default: "NA"; when Not Applicable.

page_width_inches ........ The calculated dimension (left to right) of the 
                           digital document page in inches.
                           Note: Products generated from scanned images of 
                           historical paper sheets may not match the original 
                           print dimensions of the source material.
                           Type: numeric
                           Domain: positive non-zero numeric values

page_height_inches ....... The calculated dimension (left to right) of the 
                           digital document page in inches.
                           Note: Products generated from scanned images of 
                           historical paper sheets may not match the original 
                           print dimensions of the source material.
                           Type: numeric
                           Domain: positive non-zero numeric values

product_filename ......... Name of the primary digital deliverable product.
                           Type: text
                           Domain: varies by series
                           ustopo_current: "<CellState> <CellName>"
                           ustopo_historical: "<CellState> <CellName> 
                           <ProductionDate> <Projection>"
                           historicaltopo: "<PrintedState> <PrintedTitle> 
                           <SourceSystemID> <DateOnMap> <MapScale>"

product_filesize ......... Calculated measure of storage capacity occupied by 
                           the primary deliverable within a digital filesystem. 
                           Type: numeric
                           Domain: positive non-zero integers

product_url .............. Internet location of the primary deliverable.
                           Type: text
                           Domain: RFC-1738

thumbnail_url ............ Internet location of a raster graphic representing 
                           the digital map product.
                           Type: text
                           Domain: RFC-1738
 
geotiff_url .............. Internet location of a georeferenced TIFF raster 
                           graphic representing the digital map product.
                           Note on variation: files in this category are sorted 
                           by GNIS primary state, defaulting to printed state 
                           when not applicable.
                           Type: text
                           Domain: RFC-1738
 
sciencebase_url .......... Internet location of the published entry in the 
                           ScienceBase public catalog. Existing items will 
                           often receive updates between editions. 
                           Note: URLs are not persistent and cannot be reused 
                           once a product is removed.
                           Product Inventory UUID values will be retained for 
                           the same original content.
                           Type: text
                           Domain: RFC-1738

metadata_url ............. Internet location of the distributed version of the 
                           product metadata in XML format as maintained by the 
                           inventory system.
                           Type: text
                           Domain: RFC-1738


Examples
historicaltopo
1. Digitized scan of a paper copy printed at earlier date.

All values presented were accurate at the time of reporting.

-[ RECORD 1 ]----------+-------------------------------------------------------
scan_id                | 500828
product_inventory_uuid | 65bb10d6-ed99-1f63-0fcc-79bd5d2080a0
series                 | HTMC
edition                | Historical
map_name               | Madison West
primary_state          | Wisconsin
westbc                 | -89.50000000
eastbc                 | -89.37500000
northbc                | 43.12500000
southbc                | 43.00000000
geom_wkt               | POLYGON((-89.37500000 43.00000000,-89.37500000 
                         43.12500000,-89.50000000 43.12500000,-89.50000000 
                         43.00000000,-89.37500000 43.00000000))
date_on_map            | 1959
imprint_year           | 1975
survey_year            |
field_check_year       | 1959
edit_year              |
aerial_photo_year      | 1974
photo_inspection_year  |
photo_revision_year    | 1974
metadata_date          | 2022-11-16
print_releases         | Standard
variations             | Woodland Tint
hdatum_printed         | NAD27
projection_printed     | Polyconic
map_scale              | 24000
publishers             | USGS
languages              | English
gnis_cell_id           | 27199
gnis_cell_name         | Madison West
gnis_primary_state     | Wisconsin
grid_size              | 7.5 X 7.5 Minute
cell_type              | Standard on Grid
state_list             | WI
county_list            | Dane
page_width_inches      | 21.21
page_height_inches     | 27.36
product_filename       | WI_Madison West_500828_1959_24000_geo.pdf
product_filesize       | 11847193

product_url            | https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps
                         /HistoricalTopo/PDF/WI/24000
                         /WI_Madison%20West_500828_1959_24000_geo.pdf
                        
thumbnail_url          | https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps
                         /HistoricalTopo/PDF/WI/24000
                         /WI_Madison%20West_500828_1959_24000_tn.jpg
                         
geotiff_url            | https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps
                         /HistoricalTopo/GeoTIFF/WI
                         /WI_Madison%20West_500828_1959_24000_geo.tif
                         
metadata_url           | https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/maps
                         /historicaltopo/pdf/WI/24000
                         /WI_Madison_West_500828_1959_24000_geo.xml
                         
sciencebase_url        | https://www.sciencebase.gov/catalog/item/
                         5a8a2472e4b00f54eb3bf13e
