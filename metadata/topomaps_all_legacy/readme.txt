Last updated: 05-Nov-2018

This bundle contains plain-text, comma-separated value (CSV) files that
describe the complete collections of two USGS topographic map products:

1) US Topo (http://nationalmap.gov/ustopo) -- Current topographic quadrangle
maps. These are maps produced in 2010 or later. They are created and published
as digital products. The format is layered PDF with geospatial extensions.

2) Historical Topographic Map Collection (HTMC,
http://nationalmap.gov/historical) -- Scanned images of legacy topographic
quadrangle maps. These maps were originally published on paper in the period
1884-2006. The HTMC images are in PDF format with geospatial extensions.

The CSV files are essentially a database dump of selected fields in the USGS
internal archive database. The CSV files are refreshed nightly. They can be used
to keep private collections of map files current, write download scripts, and
for other kinds of advanced data management.


TOPOMAPS_ALL.CSV

Some fields in topomaps_all.csv are used only for HTMC maps, and some are used
for both US Topo and HTMC maps. 

----------------------------------------

The following fields are generally populated for both US Topo and HTMC maps.
The preceeding number is the field number, which is useful for script writers.
Programmers note that this list starts counting at 1.

 1  Series
 2  Version
 3  Cell ID
 4  Map Name
 5  Primary State
 6  Scale
 7  Date On Map
17  Datum
18  Projection
44  Cell Name
45  Primary State Name
46  N Lat
47  W Long
48  S Lat
49  E Long
51  Download GeoPDF
52  View FGDC Metadata XML
53  View Thumbnail Image
55  GDA Item ID
56  Create Date
57  Byte Count
58  Grid Size
59  Download Product S3
60  View Thumbnail Image S3
61  NRN
62  NSN

Fields 59 and 60 were added in January 2018. These URLs point to USGS data in the 
Amazon S3 cloud. The files retrieved through the field 59 URL are the same as 
those retrieved through the field 51 URL, but will be delivered without 
zipping and with different names (the names of delivered files may change in 
the near future, and we strongly recommend that applications not depend on permanance 
of the current names; field 55 (GDA Item ID) is a primary-key product identifier
than can be relied on to be permanent). The service pointed to by the field 51 URL 
remains up as of January 2018, but may be under threat due to outdated software and 
security concerns. We recommend applications use the cloud URL from field 59.

-----------------------------------------

The following fields are generally populated only for HTMC maps. Most of the
values were transcribed from the printed map collar at the time the
paper map was scanned. These attributes were not used consistently throughout the
USGS topographic mapping programs of 1884-2006, so population of some of these
fields is not consistent or predictable, nor are the meanings of the values
necessarily consistent. These fields are discussed in the HTMC product
standard, available at http://pubs.usgs.gov/tm/11b03/


 8  Imprint Year
 9  Woodland Tint
10  Visual Version Number
11  Photo Inspection Year
12  Photo Revision Year
13  Aerial Photo Year
14  Edit Year
15  Field Check Year
16  Survey Year
19  Advance
20  Preliminary
21  Provisional
22  Interim
23  Planimetric
24  Special Printing
25  Special Map
26  Shaded Relief
27  Orthophoto
28  Pub USGS
29  Pub Army Corps Eng
30  Pub Army Map
31  Pub Forest Serv
32  Pub Military Other
33  Pub Reclamation
34  Pub War Dept
35  Pub Bur Land Mgmt
36  Pub Natl Park Serv
37  Pub Indian Affairs
38  Pub EPA
39  Pub Tenn Valley Auth
40  Pub US Commerce
41  Keywords
42  Map Language
43  Scanner Resolution
50  Link to HTMC Metadata
54  Scan ID


TOPOMAPS_REPLACED_BY.CSV

'topomaps_replaced_by.csv' is a more specialized file that will be of interest
only to advanced users of these data. This file patches an oversight in USGS
processes for publishing these maps. Following is an explanation:

The field 'GDA Item ID' is a database primary key to each product instance.
This key is guaranteed to be unique within the domain of US Topo and HTMC
maps. However, it is not guaranteed to be permanent in the sense of forever
referencing a distributable product. Maps in both series are sometimes
replaced to correct technical errors. In such a case, the new map is made from
exactly the same source material, and the replaced map ceases to have any
value. The replaced map is therefore flagged as 'non-distributable' and for
all practical purposes is deleted. However, this creates a problem for anyone
using these CSV files for ongoing operations. New versions of topomaps_all.csv
may be missing records that were present in earlier versions of the file, and
the download URLs in the earlier version may now be dead links.
topomaps_replaced_by.csv addresses this problem by listing all 'replaced' GDA
IDs, and in some cases relates these to the GDA ID of the new product
instance. The fields in this file are:

GDA Item ID - The database key of a product that has been deleted.

Replaced by ID - The database key of the new product. If this field is null,
it means that a bad product has been deleted but has not yet been replaced.

Scan ID - For HTMC maps only. This is an additional piece of information that
may be helpful in some cases. For most HTMC corrections the map does not need
to be rescanned, so the Scan ID remains the same between the old and new
product instances. However, this is not absolutely guaranteed; there are
occasionally cases that require rescanning, which means that the same paper
map gets both a new Scan ID and a new GDA ID.

Cell ID - Unique ID for the standard cell associated with a map extent. This
is redundant information, and not strictly necessary to include in this file.


Change History:
20181105 Add: new fields (columns 61 and 62)
20180128 Add: new fields (columns 59 and 60)
20140910 Minor Revisions
20140519 First Edition

