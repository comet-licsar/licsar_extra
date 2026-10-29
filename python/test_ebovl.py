#!/usr/bin/env python3
# some scratch code for figs
from gafa_licsar import *
import numpy as np
import matplotlib.pyplot as plt

### THUR:
# amplitude (sigma naught -> dB) and change labels to *4 and *16
# also add the precise extents rectangle

# QML coh, pha
# 2 Amplitude of an extended single burst with outline of standard SLC truncation
sigmas=glob.glob('sigma*')
#backscatter
imgs = [10 * np.log10(np.maximum(load_gdr(sigmas[0])[0], 1e-10)),
        10 * np.log10(np.maximum(load_gdr(sigmas[1])[0], 1e-10)),
        10 * np.log10(np.maximum(load_gdr(sigmas[2])[0], 1e-10))]
cmap = 'viridis'

# final fig - only IW1:


from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

rect_wids = [20260, 23612, 24232]
rect_heis = [1420, 1424, 1422]
rect_wids2 = [20520, 24372, 24792]
rect_heis2 = [1476, 1480, 1478]
## attempt 2
left, right = 0.08, 0.98
bottom, top = 0.05, 0.95
gap = 0.01  # vertical gap fraction in figure coordinates
gap = 0.005
vmin = -35
vmin = -22 # NESZ
#vmin = -20
vmax = -5
vmax = 0
dy = 4
dx = 16

fig = plt.figure(figsize=(8.5, 8))

mappable = None

available_h = (top-bottom) - gap*(len(imgs)-1)

y = top
add_smaller = False

for i, im in enumerate(imgs):

    h_frac = hs[i] / sum(hs)
    w_frac = ws[i] / max(ws)

    ax_h = available_h * h_frac
    ax_w = (right-left) * w_frac

    ax = fig.add_axes([
        left,
        y-ax_h,
        ax_w,
        ax_h
    ])
    
    if i == 0:
        cb_x = left + ax_w + 0.025
        cb_y = y - ax_h
        cb_h = ax_h
    
    ny, nx = im.shape

    mappable = ax.imshow(
        im,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    # add rectangle:
    if add_smaller:
        rect_wid = rect_wids[i]
        rect_hei = rect_heis[i]

        rect_wid_px = rect_wid / dx
        rect_hei_px = rect_hei / dy

        rect = Rectangle(
            (
                nx/2 - rect_wid_px/2,
                ny/2 - rect_hei_px/2
            ),
            rect_wid_px,
            rect_hei_px,
            fill=False,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
    rect_wid = rect_wids2[i]
    rect_hei = rect_heis2[i]

    rect_wid_px = rect_wid / dx
    rect_hei_px = rect_hei / dy
    
    rect2 = Rectangle(
        (
            nx/2 - rect_wid_px/2,
            ny/2 - rect_hei_px/2
        ),
        rect_wid_px,
        rect_hei_px,
        fill=False,
        edgecolor='black',
        linewidth=2
    )
    
    ax.add_patch(rect2)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x*dx:.0f}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y*dy:.0f}")
    )

    if i == 0:
        ax.set_ylabel("Azimuth [px]")

    if i == 0: #len(imgs)-1:
        ax.set_xlabel("Range [px]")
        
    y -= ax_h + gap


cax = fig.add_axes([cb_x, cb_y+0.025, 0.02, cb_h-0.05])

cb = fig.colorbar(
    mappable,
    cax=cax,
    label=r"$\sigma^0$ [dB]"
)

#cb.set_ticks([-22, -20, -15, -10, -5])

plt.show()

'''
img = 10 * np.log10(np.maximum(load_gdr(sigmas[0])[0], 1e-10))
im = img
i = 0
vmin = -20
fig = plt.figure(figsize=(8.5, 2.5))
cb_x = left + ax_w + 0.025
cb_y = y - ax_h
cb_h = ax_h
ax = fig.add_axes([
        left,
        y-ax_h,
        ax_w,
        ax_h
    ])
mappable = ax.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
ny, nx = im.shape
rect_wid = rect_wids2[i]
rect_hei = rect_heis2[i]
rect_wid_px = rect_wid / dx
rect_hei_px = rect_hei / dy
rect2 = Rectangle(
    (
        nx/2 - rect_wid_px/2,
        ny/2 - rect_hei_px/2
    ),
    rect_wid_px,
    rect_hei_px,
    fill=False,
    edgecolor='black',
    linewidth=2
)

ax.add_patch(rect2)
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x*dx:.0f}")
)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f"{y*dy:.0f}")
)

ax.set_ylabel("Azimuth [px]")

ax.set_xlabel("Range [px]")
#cax = fig.add_axes([cb_x, cb_y+0.025, 0.02, cb_h-0.05])

cb = fig.colorbar(
    mappable,
    #cax=cax,
    label=r"$\sigma^0$ [dB]"
)

#cb.set_ticks([-22, -20, -15, -10, -5])

plt.show()
'''



'''
# previous version - all three swaths:

'''








# 4 a. Single burst overlap in radar coords (Queen Maud’s Land?) using 100% extension and outline of standard overlap.
# b. Coherence variation across burst in rg.
# c. Same in az.
ifgs=glob.glob('ifgm_*')
#for fname in ifgs:
#    ifg, meta = load_gdr(fname)
#    plt.imshow(np.log1p(np.abs(ifg)), cmap="gray")



# Example: your arrays
# A: (h1, w1) or (h1, w1, c)
# B: (h2, w2) or (h2, w2, c)
# C: (h3, w3) or (h3, w3, c)
# A, B, C = arr1, arr2, arr3
# coh
cohs = [np.abs(load_gdr(ifgs[0])[0]),
        np.abs(load_gdr(ifgs[1])[0]),
        np.abs(load_gdr(ifgs[2])[0]),]
cmap = 'gray'
#pha
phas = [np.angle(load_gdr(ifgs[0])[0]),
        np.angle(load_gdr(ifgs[1])[0]),
        np.angle(load_gdr(ifgs[2])[0]),]
cmap = 'hsv'






# b
axx, meaning = 0, 'Range sample'
#axx, meaning = 1, 'Azimuth line'
y0 = cohs[0].mean(axis=axx)
y1 = cohs[1].mean(axis=axx)
y2 = cohs[2].mean(axis=axx)
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




# only IW2, plot on vik0?

# now using for phase:

from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

#rect_wids = [20260, 23612, 24232]
#rect_heis = [1420, 1424, 1422]
# these ones are needed!!
# these are for the bovl ifg:
rect_wids2 = [19833, 23501, 22882]
rect_heis2 = [123, 116, 114]

## attempt 2
left, right = 0.08, 0.98
bottom, top = 0.05, 0.95
gap = 0.01  # vertical gap fraction in figure coordinates
gap = 0.005
vmin = -35
vmin = -22 # NESZ
vmax = -5
imgs = phas
imgs_coh = cohs
vmin = -np.pi
vmax = np.pi
# cmap = 'vik0'
import cmcrameri.cm as cmc
cmap = cmc.vikO

# to do only one:
imgs = [phas[1]]
imgs_coh = [cohs[1]]
dy = 4
dx = 16
hs, ws = [], []
for im in imgs:
    hs.append(im.shape[0])
    ws.append(im.shape[1])

####### option 1 - plot with coh avgs
fig = plt.figure(figsize=(8.5, 8))

mappable = None

available_h = (top - bottom) - gap * (len(imgs) - 1)

y = top

for i, im in enumerate(imgs):

    # -----------------------
    # coherence statistics
    # -----------------------
    im_coh = imgs_coh[i]

    coh_az = im_coh.mean(axis=1)
    coh_rg = im_coh.mean(axis=0)

    h_frac = hs[i] / sum(hs)
    w_frac = ws[i] / max(ws)

    ax_h = available_h * h_frac
    ax_w = (right - left) * w_frac

    img_x = left
    img_y = y - ax_h

    ny, nx = im.shape

    # -----------------------
    # main image
    # -----------------------
    ax = fig.add_axes([
        img_x,
        img_y,
        ax_w,
        ax_h
    ])

    mappable = ax.imshow(
        im,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )

    # -----------------------
    # rectangles
    # -----------------------
    if add_smaller:

        rect_wid_px = rect_wids[i] / dx
        rect_hei_px = rect_heis[i] / dy

        ax.add_patch(
            Rectangle(
                (
                    nx/2 - rect_wid_px/2,
                    ny/2 - rect_hei_px/2
                ),
                rect_wid_px,
                rect_hei_px,
                fill=False,
                edgecolor='black',
                linewidth=1
            )
        )

    rect_wid_px = rect_wids2[i] / dx
    rect_hei_px = rect_heis2[i] / dy

    ax.add_patch(
        Rectangle(
            (
                nx/2 - rect_wid_px/2,
                ny/2 - rect_hei_px/2
            ),
            rect_wid_px,
            rect_hei_px,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
    )

    # -----------------------
    # coordinate formatting
    # -----------------------
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x * dx:.0f}")
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y * dy:.0f}")
    )

    ax.set_xlabel("Range [px]")
    ax.set_ylabel("Azimuth [px]")

    # =====================================================
    # coherence profiles aligned to displayed image
    # =====================================================

    pos = ax.get_position()

    prof_gap = 0.01
    prof_w = 0.08
    prof_h = 0.08 * pos.height

    # -----------------------
    # TOP coherence profile
    # -----------------------

    ax_top = fig.add_axes(
        [
            pos.x0,
            pos.y1 + 0.003,
            pos.width,
            prof_h
        ],
        sharex=ax
    )

    ax_top.plot(
        np.arange(nx),
        coh_rg,
        "k-",
        lw=1
    )

    ax_top.set_ylim(0.5, 1.0)

    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position("right")
    ax_top.set_ylabel("mean coh")

    ax_top.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False
    )

    ax_top.spines["left"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    # -----------------------
    # RIGHT coherence profile
    # -----------------------

    ax_right = fig.add_axes(
        [
            pos.x1 + prof_gap,
            pos.y0,
            prof_w,
            pos.height
        ],
        sharey=ax
    )

    ycoord = np.linspace(
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        ny
    )

    ax_right.plot(
        coh_az,
        ycoord,
        "k-",
        lw=1
    )

    ax_right.set_xlim(0.5, 1.0)
    ax_right.set_ylim(ax.get_ylim())

    ax_right.set_xlabel("mean\ncoh")

    ax_right.tick_params(
        axis="y",
        left=False,
        labelleft=False
    )

    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # -----------------------
    # colorbar position
    # -----------------------

    if i == 0:
        cb_x = pos.x1 + prof_gap + prof_w + 0.015
        cb_y = pos.y0
        cb_h = pos.height

    y -= ax_h + gap

# =========================================================
# colorbar
# =========================================================

cax = fig.add_axes([
    cb_x,
    cb_y,
    0.02,
    cb_h
])

cb = fig.colorbar(
    mappable,
    cax=cax,
    label="phase [rad]"
)

plt.show()



####### option 2 - plot without coh avgs

# fig = plt.figure(figsize=(8.5, 4))
fig = plt.figure(figsize=(8.5, 8))
mappable = None
available_h = (top-bottom) - gap*(len(imgs)-1)
y = top
add_smaller = False

for i, im in enumerate(imgs):
    # coh
    im_coh = imgs_coh[i]
    coh_az = im_coh.mean(axis=1)  # one value per azimuth line
    coh_rg = im_coh.mean(axis=0)  # one value per range sample
    #
    h_frac = hs[i] / sum(hs)
    w_frac = ws[i] / max(ws)
    ax_h = available_h * h_frac
    ax_w = (right-left) * w_frac
    ax = fig.add_axes([
        left,
        y-ax_h,
        ax_w,
        ax_h
    ])
    if i == 0:
        cb_x = left + ax_w + 0.025
        cb_y = y - ax_h
        cb_h = ax_h
    ny, nx = im.shape
    mappable = ax.imshow(
        im,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    # add rectangle:
    if add_smaller:
        rect_wid = rect_wids[i]
        rect_hei = rect_heis[i]
        rect_wid_px = rect_wid / dx
        rect_hei_px = rect_hei / dy
        rect = Rectangle(
            (
                nx/2 - rect_wid_px/2,
                ny/2 - rect_hei_px/2
            ),
            rect_wid_px,
            rect_hei_px,
            fill=False,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
    rect_wid = rect_wids2[i]
    rect_hei = rect_heis2[i]
    rect_wid_px = rect_wid / dx
    rect_hei_px = rect_hei / dy
    rect2 = Rectangle(
        (
            nx/2 - rect_wid_px/2,
            ny/2 - rect_hei_px/2
        ),
        rect_wid_px,
        rect_hei_px,
        fill=False,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(rect2)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x*dx:.0f}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y*dy:.0f}")
    )
    # coh
    prof_gap = 0.015
    prof_w = 0.10
    #
    ax_right = fig.add_axes([
        left + ax_w + prof_gap,
        y,
        prof_w,
        ax_h
    ])
    ax_right.plot(coh_az, np.arange(ny), 'k-')
    ax_right.set_ylim(ny-1, 0)   # match imshow orientation
    ax_right.set_xlabel('mean coherence')
    ax_right.set_xlim(0.5, 1)
    # only keep left y labels on the image
    ax_right.set_yticklabels([])
    #
    # coh rg:
    prof_h = 0.12
    ax_bottom = fig.add_axes([
        left,
        y - prof_h - 0.01,
        ax_w,
        prof_h
    ])
    ax_bottom.plot(np.arange(nx), coh_rg, 'k-')
    ax_bottom.set_xlim(0, nx-1)
    ax_bottom.set_ylabel('mean coherence')
    ax_right.set_ylim(0.5, 1)
    # use same metre formatting
    ax_bottom.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x*dx:.0f}")
    )
    ax_bottom.set_xlabel("Range [m]")
    if i == 1 or len(imgs) == 1:
        ax.set_ylabel("Azimuth [px]")
    if i == len(imgs)-1:
        ax.set_xlabel("Range [px]")
    y -= ax_h + gap


cax = fig.add_axes([cb_x, cb_y+0.025, 0.02, cb_h-0.05])
cb = fig.colorbar(
    mappable,
    cax=cax,
    label = 'phase [rad]'
)

plt.show()
















# final fig 2 - only IW1:


from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

rect_wids = [20260, 23612, 24232]
rect_heis = [1420, 1424, 1422]
rect_wids2 = [20520, 24372, 24792]
rect_heis2 = [1476, 1480, 1478]
## attempt 2
left, right = 0.08, 0.98
bottom, top = 0.05, 0.95
gap = 0.01  # vertical gap fraction in figure coordinates
gap = 0.005
vmin = -35
vmin = -22 # NESZ
#vmin = -20
vmax = -5
vmax = 0
dy = 4
dx = 16

fig = plt.figure(figsize=(8.5, 8))

mappable = None

available_h = (top-bottom) - gap*(len(imgs)-1)

y = top
add_smaller = False

for i, im in enumerate(imgs):

    h_frac = hs[i] / sum(hs)
    w_frac = ws[i] / max(ws)

    ax_h = available_h * h_frac
    ax_w = (right-left) * w_frac

    ax = fig.add_axes([
        left,
        y-ax_h,
        ax_w,
        ax_h
    ])
    
    if i == 0:
        cb_x = left + ax_w + 0.025
        cb_y = y - ax_h
        cb_h = ax_h
    
    ny, nx = im.shape

    mappable = ax.imshow(
        im,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    # add rectangle:
    if add_smaller:
        rect_wid = rect_wids[i]
        rect_hei = rect_heis[i]

        rect_wid_px = rect_wid / dx
        rect_hei_px = rect_hei / dy

        rect = Rectangle(
            (
                nx/2 - rect_wid_px/2,
                ny/2 - rect_hei_px/2
            ),
            rect_wid_px,
            rect_hei_px,
            fill=False,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
    rect_wid = rect_wids2[i]
    rect_hei = rect_heis2[i]

    rect_wid_px = rect_wid / dx
    rect_hei_px = rect_hei / dy
    
    rect2 = Rectangle(
        (
            nx/2 - rect_wid_px/2,
            ny/2 - rect_hei_px/2
        ),
        rect_wid_px,
        rect_hei_px,
        fill=False,
        edgecolor='black',
        linewidth=2
    )
    
    ax.add_patch(rect2)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x*dx:.0f}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y*dy:.0f}")
    )

    if i == 0:
        ax.set_ylabel("Azimuth [px]")

    if i == 0: #len(imgs)-1:
        ax.set_xlabel("Range [px]")
        
    y -= ax_h + gap


cax = fig.add_axes([cb_x, cb_y+0.025, 0.02, cb_h-0.05])

cb = fig.colorbar(
    mappable,
    cax=cax,
    label=r"$\sigma^0$ [dB]"
)

#cb.set_ticks([-22, -20, -15, -10, -5])

plt.show()






# below is on how to get the bovl width/length values:


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

# to get bovls:
mtab=tab/20260306R_tab
ScanSAR_burst_overlap $mtab tata 1 1 3 1

# Example usage:
for poly_i, w, h in polygon_width_height("tata.SLC_az_ovr.poly"):
    print(poly_i, w, h)

# 5 a. Wrapped geocoded burst and swath overlaps for Türkiye for single geometry (with edges trimmed where coh drops)
# b. Same with standard BOI.
# c and d. Same for an ice example: Queen Maud’s Land?


# 6 a. Unwrapped Türkiye with asc and desc on same image b. 3d decomposition for Türkiye including az and rg offsets c. profile across 3d image that shows diffierence in noise between areas covered by burst overlaps and az offsets
