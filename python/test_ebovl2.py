#!/usr/bin/env python3
# some scratch code for figs
from gafa_licsar import *
from pygmt.params import Pattern
# 2 Amplitude of an extended single burst with outline of standard SLC truncation
f = 'sigma_2026-03-06_002-1693-IW3-HH.gdr'
f = 'sigma_2026-03-06_002-1694-IW3-HH.gdr'

# this particular burst would have standard burst window in SLC as:
# 437   22560      81    1344
# i.e.
# slc_window = [[1344-81+1, 22560 - 437+1]]  # but must be [xmin, ymin, xmax, ymax]
slc_window = [437, 81, 22560, 1344]
slc_bovl = [0, 0, 22883, 114] # 22883 x 114
full_bovl = [0,0, 29520, 798] # 29520x798

sigma, meta = load_gdr(f)
sigma = 10*np.log10(sigma)

ys=np.arange(sigma.shape[0])
xs=np.arange(sigma.shape[1])
ys = ys*3
xs = xs*12

# prep the grid:
import xarray as xr
grid = xr.DataArray(
    data=sigma,
    coords={"y": ys, "x": xs},
    dims=("y", "x")
)

import matplotlib.cm

vmin = -22
vmin = -27
vmax = 0

cmap = matplotlib.cm.get_cmap("viridis")
c0 = cmap((vmin - vmin) / (vmax - vmin))[:3]   # color at vmin
c1 = cmap((vmax - vmin) / (vmax - vmin))[:3]   # color at vmax

def rgb(c):  # c is 0..1 floats
    return f"rgb({int(round(c[0]*255))},{int(round(c[1]*255))},{int(round(c[2]*255))})"


import pygmt
from pygmt.params import Position
# pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="10p")
axis_annot = "8p"   # tick labels
axis_label = "6p"  # axis title
#cb_annot = "6p"     # colorbar tick labels
#cb_label = "10p"    # colorbar title

pygmt.config(
    FONT_ANNOT_PRIMARY=axis_annot,
    FONT_LABEL=axis_label,
    FONT_ANNOT_SECONDARY=axis_annot,
    FONT_TITLE=axis_label,
    #FONT_ANNOT_COLORBAR=cb_annot,
    #FONT_LABEL_COLORBAR=cb_label,
)
fig = pygmt.Figure()

# pygmt.config(COLOR_BACKGROUND=rgb(c0), COLOR_FOREGROUND=rgb(c1))
r0,g0,b0 = 68, 1, 84      # example: your desired background color
r1,g1,b1 = 253, 231, 36  # example: your desired foreground color

pygmt.config(
    COLOR_BACKGROUND=f"{r0}/{g0}/{b0}",
    COLOR_FOREGROUND=f"{r1}/{g1}/{b1}",
)
pygmt.makecpt(
    cmap="matplotlib/viridis",
    series=[vmin, vmax, 0.5],
    continuous=True,
)

# plot the grid
fig.grdimage(grid,cmap=True,
             frame=["WS", "xaf+lAcross-Track [px]", "yaf+lAlong-Track [px]"],
             region=[0, xs[-1], 0, ys[-1]],
             projection="X12c/3c")

# plot the black rectangle
#slc_window_aligned =
# plot the overlap rectangle(s):
# Large rectangle: red with one hatch direction
slc_bovl_aligned = [xs[-1] - slc_bovl[2], ys[-1] - slc_bovl[3], xs[-1], ys[-1]] # 22883 x 114
full_bovl_aligned = [xs[-1] - full_bovl[2], ys[-1] - full_bovl[3], xs[-1], ys[-1]]# 29520x798
fig.plot(
    data=[full_bovl_aligned],
    style="r+s",
    pen="1p,red",
    fill=Pattern(19, fgcolor="red", bgcolor=""),
)

# Smaller rectangle: orange with a different hatch pattern
fig.plot(
    data=[slc_bovl_aligned],
    style="r+s",
    pen="1p,orange",
    fill=Pattern(20, fgcolor="orange", bgcolor=""),
)


fig.colorbar(
    cmap=True,
    position=Position("MR", cstype="outside", offset=(0.5, 0)),
    frame=["x+lIntensity [dB]"],
    length='3c',
    width='0.35c',
)

fig.savefig('pgmtest2.png', dpi=200)
fig.savefig('pgmtest2b.pdf')
fig.savefig('pgm_20260720b.pdf')


















fig.show()

from gdar.io.native import read_gdr  # GDAR internal format
from pathlib import Path
dr = read_gdr(Path('.')/f)
ref = {}
bid = f.split("_")[-1]
ref[bid] = read_gdr(f)



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
imgs = [10 * np.log10(np.maximum(load_gdr(sigmas[0])[0], 1e-10)),
        10 * np.log10(np.maximum(load_gdr(sigmas[1])[0], 1e-10)),
        10 * np.log10(np.maximum(load_gdr(sigmas[2])[0], 1e-10))]
cmap = 'viridis'


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

plt.colorbar()
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
