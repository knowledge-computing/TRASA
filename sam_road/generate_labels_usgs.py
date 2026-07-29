import os
import sys
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
import networkx as nx

image_size = 256
KEYPOINT_RADIUS = 3
ROAD_WIDTH = 1

def create_directory(path, delete=False):
    if delete and os.path.isdir(path):
        # be careful with delete=True, default is false
        import shutil
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def draw_points_on_image(size, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for (x, y) in points:
        x = int(round(x))
        y = int(round(y))
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=255)
    return img

def draw_line_segments_on_image(size, line_segments, width):
    """
    Draws line segments on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - line_segments: A list of tuples, where each tuple represents a line segment as ((x1, y1), (x2, y2)).
    - width: The width of the lines to be drawn, in pixels.

    Returns:
    - A square image with the given line segments drawn.
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for (p1, p2) in line_segments:
        x1, y1 = p1
        x2, y2 = p2
        draw.line(
            (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            fill=255,
            width=width,
        )
    return img

def load_graph_from_vtp(vtp_path, size):
    """
    reads a vtp polyline graph with normalized 0..1 coords
    and returns a networkx graph in pixel coordinates.
    """
    mesh = pv.read(vtp_path)

    pts_norm = np.array(mesh.points)[:, :2]  # (x_norm, y_norm)
    px = pts_norm[:, 0] * (size - 1)
    py = pts_norm[:, 1] * (size - 1)
    pts_pix = np.stack([px, py], axis=1)

    g = nx.Graph()

    # add nodes
    for i, (x, y) in enumerate(pts_pix):
        g.add_node(i, xy=(x, y))

    # mesh.lines encodes polylines: [npts, id0, id1, ..., npts, id0, ...]
    lines = mesh.lines
    i = 0
    n = len(lines)
    while i < n:
        npts = int(lines[i])
        idxs = lines[i + 1 : i + 1 + npts]
        i += 1 + npts

        if npts < 2:
            continue

        for j in range(len(idxs) - 1):
            u = int(idxs[j])
            v = int(idxs[j + 1])
            g.add_edge(u, v)

    return g

def main():
    if len(sys.argv) != 3:
        print("usage: python generate_railroad_labels_from_vtp.py <vtp_dir> <out_dir>")
        sys.exit(1)

    vtp_dir = sys.argv[1]
    out_dir = sys.argv[2]

    create_directory(out_dir, delete=False)

    for fname in sorted(os.listdir(vtp_dir)):
        if not fname.endswith(".vtp"):
            continue

        vtp_path = os.path.join(vtp_dir, fname)
        base = os.path.splitext(fname)[0]  # e.g. sample_8_5_5_graph

        print(f"processing {fname}")

        g = load_graph_from_vtp(vtp_path, image_size)

        # collect edges as pixel segments
        segments = []
        for u, v in g.edges():
            x1, y1 = g.nodes[u]["xy"]
            x2, y2 = g.nodes[v]["xy"]
            segments.append(((x1, y1), (x2, y2)))

        # key nodes: degree != 2
        key_nodes = []
        for node, degree in g.degree():
            if degree != 2:
                x, y = g.nodes[node]["xy"]
                key_nodes.append((x, y))

        keypoint_img = draw_points_on_image(
            size=image_size, points=key_nodes, radius=KEYPOINT_RADIUS
        )
        road_img = draw_line_segments_on_image(
            size=image_size, line_segments=segments, width=ROAD_WIDTH
        )

        key_out = os.path.join(out_dir, f"{base}_keypoint_mask.png")
        road_out = os.path.join(out_dir, f"{base}_road_mask.png")

        keypoint_img.save(key_out)
        road_img.save(road_out)

        print(f"saved {key_out} and {road_out}")

if __name__ == "__main__":
    main()
