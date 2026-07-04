#!/usr/bin/env python3
# some scratch code for figs
from gafa_licsar import *

# 2 Amplitude of an extended single burst with outline of standard SLC truncation



# 4 a. Single burst overlap in radar coords (Queen Maud’s Land?) using 100% extension and outline of standard overlap.
# b. Coherence variation across burst in rg.
# c. Same in az.
ifgs=glob.glob('ifgm*')
for fname in ifgs:
    ifg, meta = load_gdr(fname)
    plt.imshow(np.log1p(np.abs(ifg)), cmap="gray")
sigmas=glob.glob('sigma*')

import numpy as np
import matplotlib.pyplot as plt
# Example: your arrays
# A: (h1, w1) or (h1, w1, c)
# B: (h2, w2) or (h2, w2, c)
# C: (h3, w3) or (h3, w3, c)
# A, B, C = arr1, arr2, arr3
# coh
imgs = [np.abs(load_gdr(ifgs[0])[0]),
        np.abs(load_gdr(ifgs[1])[0]),
        np.abs(load_gdr(ifgs[2])[0]),]
cmap = 'gray'
#pha
imgs = [np.angle(load_gdr(ifgs[0])[0]),
        np.angle(load_gdr(ifgs[1])[0]),
        np.angle(load_gdr(ifgs[2])[0]),]
cmap = 'hsv'
#backscatter
imgs = [load_gdr(sigmas[0])[0],
        load_gdr(sigmas[1])[0],
        load_gdr(sigmas[2])[0]]
cmap = 'gray'

hs = [im.shape[0] for im in imgs]
ws = [im.shape[1] for im in imgs]

# Figure layout parameters (tweak if you want different margins/gaps)
left, right = 0.08, 0.98
bottom, top = 0.05, 0.95
gap = 0.01  # vertical gap fraction in figure coordinates

fig = plt.figure(figsize=(6, 6))

total_h = sum(hs)
y = top
for i, im in enumerate(imgs):
    h_frac = hs[i] / total_h
    w_frac = ws[i] / max(ws)

    ax_h = (top - bottom) * h_frac
    ax_w = (right - left) * w_frac

    ax_x = left
    ax_y = y - ax_h

    ax = fig.add_axes([ax_x, ax_y, ax_w, ax_h])
    ax.imshow(im, aspect='equal', interpolation='nearest', cmap=cmap)
    # ax.axis('off')

    y = ax_y - gap


plt.show()





# b
axx, meaning = 0, 'Range sample'
axx, meaning = 1, 'Azimuth line'
y0 = imgs[0].mean(axis=axx)
y1 = imgs[1].mean(axis=axx)
y2 = imgs[2].mean(axis=axx)
y0[y0 == 0] = np.nan
y1[y1 == 0] = np.nan
y2[y2 == 0] = np.nan

x0 = range(len(y0))
x1 = range(len(y1))
x2 = range(len(y2))

plt.figure()
plt.plot(x0, y0, label="IW1")
plt.plot(x1, y1, label="IW2")
plt.plot(x2, y2, label="IW3")

plt.xlabel(meaning)
plt.ylabel("Mean coherence")
plt.legend()
plt.tight_layout()
plt.show()







import numpy as np

def polygon_width_height(path, x_col=0, y_col=1, id_col=2):
    """
    Reads a GAMMA-style polygon vertex list:
      x y id
    Returns a list of (polygon_id, width_px, height_px).
    Polygons are assumed to be delineated by repeating point_id = 1..k within each polygon,
    i.e., each polygon's vertices all share the same set of ids {1..5} in your example.
    """
    data = np.loadtxt(path)

    x = data[:, x_col]
    y = data[:, y_col]
    pid = data[:, id_col].astype(int)

    # Group vertices by polygon index = how many times id_col cycles back to the first id value
    # (works for your format where each polygon uses ids 1..5)
    first_id = pid.min()
    # polygon_index increments when pid returns to first_id
    polygon_index = np.cumsum((pid == first_id) & (np.arange(len(pid)) != 0))

    # Build per-polygon width/height from bounding box in pixel coords
    results = []
    for p in np.unique(polygon_index):
        mask = polygon_index == p
        xi = x[mask]
        yi = y[mask]
        width_px = int(xi.max() - xi.min() + 1)
        height_px = int(yi.max() - yi.min() + 1)
        results.append((int(p), width_px, height_px))

    return results

# Example usage:
# for poly_i, w, h in polygon_width_height("your_polygon.txt"):
#     print(poly_i, w, h)

# 5 a. Wrapped geocoded burst and swath overlaps for Türkiye for single geometry (with edges trimmed where coh drops)
# b. Same with standard BOI.
# c and d. Same for an ice example: Queen Maud’s Land?


# 6 a. Unwrapped Türkiye with asc and desc on same image b. 3d decomposition for Türkiye including az and rg offsets c. profile across 3d image that shows diffierence in noise between areas covered by burst overlaps and az offsets
