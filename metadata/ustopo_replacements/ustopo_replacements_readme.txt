US Topo Replacements
Product inventory changes within the US Topo map series.
Focusing on variations between major editions.

Column Definitions
gnis_cell_id ............. Single spatial unit within 7.5-Minute grid. At most 
                           one current edition per cell unit. 
                           
product_inventory_uuid ... Universally unique identifier for a product within 
                           the inventory system.
                           
db_uuid .................. Identifies a single product metadata record loaded 
                           to the inventory database.
                           
inv_uuid ................. Inventory instance of a loaded product metadata 
                           record. A new instance will be created with each 
                           processing event of the loaded record. At most one 
                           active per loaded record.
                           
product_filename ......... Published filename of primary deliverable for this 
                           product.
                           
unit_auth_code ........... Authorization code under which work on this cell 
                           was conducted. New submissions are handled as 
                           replacements to editions using the same 
                           authorization code. 
                           
publication_date ......... Date recorded in the product metadata for when this 
                           version of the product was produced. Typically 
                           coincides with the date used in the filename and 
                           year printed on map.
                           
metadata_date ............ Last update to the distributed public metadata. 
                           Changes may occur for a variety of reasons, 
                           including edition status.


Terms
GNIS, Geographic Names Information System

