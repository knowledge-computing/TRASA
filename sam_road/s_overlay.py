
import cv2, numpy as np, pyvista as pv, os
imgp="/users/4/kirsa002/IMOLA/data/train_data_fault/raw/sample_000001_data.png"
vtpp="/users/4/kirsa002/IMOLA/data/train_data_fault/vtp/sample_000001_graph.vtp"
img=cv2.imread(imgp); h,w=img.shape[:2]; overlay=img.copy()
mesh=pv.read(vtpp)
def to_xy(p): return int(round(p[0])), int(round(p[1]))
if mesh.lines.size:
    data=mesh.lines; i=0
    while i<len(data):
        n=int(data[i]); i+=1
        ids=[int(x) for x in data[i:i+n]]; i+=n
        for a,b in zip(ids[:-1], ids[1:]):
            ax,ay=to_xy(mesh.points[a][:2]); bx,by=to_xy(mesh.points[b][:2])
            cv2.line(overlay,(ax,ay),(bx,by),(0,0,255),1)
cv2.imwrite("/users/4/kirsa002/IMOLA/data/overlay_check.png", overlay)
print("saved overlay_check.png")