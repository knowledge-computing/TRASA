from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
import numpy as np
from sklearn.cluster import DBSCAN

def deduplicate_coords(coords, tolerance=2.0):
    coords_np = np.array(coords)
    if len(coords_np) == 0:
        return []

    clustering = DBSCAN(eps=tolerance, min_samples=1).fit(coords_np)
    labels = clustering.labels_
    deduped = []

    for label in set(labels):
        cluster_points = coords_np[labels == label]
        center = cluster_points.mean(axis=0)
        deduped.append(tuple(center))

    return deduped

def extract_nodes_from_lines(line_coords_list, tolerance=2.0):

    lines = [LineString(coords) for coords in line_coords_list]

    index = STRtree(lines)

    endpoints = set()
    for line in lines:
        endpoints.add((line.coords[0][0], line.coords[0][1]))  # start
        endpoints.add((line.coords[-1][0], line.coords[-1][1]))  # end

    intersections = set()
    for line in lines:
        for other in index.query(line):
            if line == other:
                continue
            inter = line.intersection(other)
            if isinstance(inter, Point):
                intersections.add((inter.x, inter.y))

    all_nodes_raw = list(endpoints.union(intersections))
    deduped_nodes = deduplicate_coords(all_nodes_raw, tolerance=tolerance)

    return deduped_nodes
nodes = extract_nodes_from_lines(example_lines, tolerance=2.0)
nodes
