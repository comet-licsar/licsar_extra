#!/usr/bin/env python3

################################################################################
# LiCStamps ps_plot
# by Milan Lazecky, 2026+, University of Leeds
#
# translation of STAMPS ps_plot.m to python (supported by MS Copilot)
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.io import loadmat
from licstamps import *


def ps_plot(
    value_type,
    plot_flag=1,
    lims=None,
    ref_ifg=0,
    ifg_list=None,
    n_x=0,
    cbar_flag=0,
    textsize=0,
    textcolor=None,
    lon_rg=None,
    lat_rg=None,
    units=None,
    *varargin
):
    """
    Python equivalent of ps_plot.m
    """

    # ---------- defaults ----------
    if lims == 0:
        lims = None
    if ifg_list is None:
        ifg_list = []
    if lon_rg is None:
        lon_rg = []
    if lat_rg is None:
        lat_rg = []

    # ---------- load PS core ----------
    ps = load_ps_data()
    n_ps = ps["n_ps"]
    n_ifg = ps["n_ifg"]

    # ---------- dispatch value_type ----------
    ph_all, fig_name, units, ref_ps = dispatch_value_type(
        ps, value_type, ref_ifg, ifg_list
    )

    # ---------- referencing ----------
    ph_disp = ph_all[:, ifg_list] if ifg_list else ph_all

    if np.isrealobj(ph_disp):
        if isinstance(ref_ps, (list, tuple, np.ndarray)) and len(ref_ps) > 0:
            if value_type != "v":
                mean_ref = np.nanmean(ph_disp[ref_ps, :], axis=0)
                ph_disp -= mean_ref
        if lims is None:
            v = ph_disp[~np.isnan(ph_disp)]
            lims = np.percentile(v, [0.1, 99.9])
    else:
        lims = (-np.pi, np.pi)

    # ---------- data-only mode (MATLAB BACKGROUND = -1) ----------
    if plot_flag == -1:
        # ph_disp is the final data after referencing
        return ph_disp, lims, None, None

    # ---------- plotting ----------
    fig, h_axes_all = plot_ifg_grid(
        ph_disp,
        ps,
        lims,
        fig_name,
        plot_flag,
        n_x,
        textsize,
        textcolor,
        lon_rg,
        lat_rg,
        units,
        cbar_flag,
    )

    # ---- ACTUALLY PLOT EACH IFG ----
    if ph_disp.ndim == 1:
        # single panel (e.g. 'v')
        ax = list(h_axes_all.values())[0]
        plt.sca(ax)
        ps_plot_ifg(
            ph_disp,
            plot_flag,
            lims,
            lon_rg,
            lat_rg,
            ext_data=None,
        )
    else:
        # multiple interferograms (e.g. 'u')
        for i, ax in h_axes_all.items():
            plt.sca(ax)
            ps_plot_ifg(
                ph_disp[:, i - 1],   # MATLAB-style indexing was preserved
                plot_flag,
                lims,
                lon_rg,
                lat_rg,
                ext_data=None,
            )

    #plt.show()

    return fig, lims, None, h_axes_all




def dispatch_value_type(ps, value_type, ref_ifg, ifg_list):
    value_type = value_type.lower()
    if value_type == "d":
        scla = loadmat("scla2.mat", squeeze_me=True)

        if "K_ps_uw" not in scla:
            raise KeyError("K_ps_uw missing from scla2.mat")

        ph_all = scla["K_ps_uw"].reshape(-1, 1)

        ref_ps = ps_setref(ps)
        fig_name = "d"
        units = "rad/m"

        return ph_all, fig_name, units, ref_ps
    
    # ============================================================
    # UNWRAPPED PHASE (u) – already assumed implemented by you
    # ============================================================
    if value_type == "u":
        phuw = loadmat("phuw2.mat", squeeze_me=True)
        ph_all = phuw["ph_uw"]
        fig_name = "u"
        units = "rad"
        ref_ps = ps_setref(ps)
        return ph_all, fig_name, units, ref_ps
    if value_type == "u-d":
        # ---- load data ----
        phuw = loadmat("phuw2.mat", squeeze_me=True)
        scla = loadmat("scla2.mat", squeeze_me=True)

        ph_all = phuw["ph_uw"] - scla["ph_scla"]

        # ---- zero master IFG ----
        ph_all[:, ps["master_ix"]] = 0.0

        ref_ps = ps_setref(ps)
        fig_name = "u-d"
        units = "rad"

        return ph_all, fig_name, units, ref_ps
    if value_type == "u-do":
        # ---- load data ----
        phuw = loadmat("phuw2.mat", squeeze_me=True)
        scla = loadmat("scla2.mat", squeeze_me=True)

        # subtract DEM error
        ph_all = phuw["ph_uw"] - scla["ph_scla"]

        # ---- deramp interferograms ----
        ph_all, ph_ramp = ps_deramp(ps, ph_all)

        # ---- zero master IFG ----
        ph_all[:, ps["master_ix"]] = 0.0

        ref_ps = ps_setref(ps)
        fig_name = "u-do"
        units = "rad"

        return ph_all, fig_name, units, ref_ps
    if value_type == "m":
        # ---- load master atmosphere ----
        scla = loadmat("scla2.mat", squeeze_me=True)

        if "C_ps_uw" not in scla:
            raise KeyError("C_ps_uw not found in scla2.mat")

        # master AOE is 1D (n_ps,)
        ph_all = scla["C_ps_uw"].reshape(-1, 1)

        ref_ps = ps_setref(ps)
        fig_name = "m"
        units = "rad"

        return ph_all, fig_name, units, ref_ps
    if value_type == "u-dmo":
        # ---- load data ----
        phuw = loadmat("phuw2.mat", squeeze_me=True)
        scla = loadmat("scla2.mat", squeeze_me=True)

        if "ph_uw" not in phuw:
            raise KeyError("ph_uw missing from phuw2.mat")
        if "ph_scla" not in scla or "C_ps_uw" not in scla:
            raise KeyError("Required fields missing from scla2.mat")

        # ---- subtract DEM error and master atmosphere ----
        ph_all = (
            phuw["ph_uw"]
            - scla["ph_scla"]
            - scla["C_ps_uw"][:, None]
        )

        # ---- deramp ----
        ph_all, ph_ramp = ps_deramp(ps, ph_all)

        # ---- zero master IFG ----
        ph_all[:, ps["master_ix"]] = 0.0

        ref_ps = ps_setref(ps)
        fig_name = "u-dmo"
        units = "rad"

        return ph_all, fig_name, units, ref_ps
    # ============================================================
    # VELOCITY (v) – SM only
    # ============================================================
    if value_type == "v":
        # ---- load unwrapped phase
        phuw = loadmat("phuw2.mat", squeeze_me=True)
        ph_uw = phuw["ph_uw"]            # shape (n_ps, n_ifg)

        n_ps, n_ifg = ph_uw.shape
        master_ix = ps["master_ix"]

        # ---- reference PS
        ref_ps = ps_setref(ps)
        if isinstance(ref_ps, int) and ref_ps == 0:
            raise RuntimeError("No reference PS available for velocity inversion")

        # ---- exclude master interferogram
        unwrap_ifg_index = [i for i in range(n_ifg) if i != master_ix]
        ph_uw = ph_uw[:, unwrap_ifg_index]
        day = ps["day"][unwrap_ifg_index]

        # ---- reference remove
        ph_uw = ph_uw - np.nanmean(ph_uw[ref_ps, :], axis=0)

        # ---- design matrix
        G = np.column_stack([
            np.ones_like(day),
            day - ps["master_day"]
        ])

        # ---- least squares (MATLAB lscov → lstsq)
        m, *_ = lstsq(G, ph_uw.T)

        # ---- wavelength
        lambda_ = getparm("lambda")[0]

        # ---- velocity (mm/yr)
        ph_all = (
            -m[1, :] * 365.25 / (4 * np.pi) * lambda_ * 1000
        )

        fig_name = "v"
        units = "mm/yr"

        return ph_all.reshape(-1, 1), fig_name, units, ref_ps
    if value_type == "v-d":
        ph = dispatch_value_type(ps, "u-d", ref_ifg, ifg_list)[0]
        ph_all = estimate_velocity_from_phase(ps, ph)
        return ph_all, "v-d", "mm/yr", ps_setref(ps)
    if value_type == "v-do":
        ph = dispatch_value_type(ps, "u-do", ref_ifg, ifg_list)[0]
        ph_all = estimate_velocity_from_phase(ps, ph)
        return ph_all, "v-do", "mm/yr", ps_setref(ps)
    if value_type == "v-m":
        ph = dispatch_value_type(ps, "u-m", ref_ifg, ifg_list)[0]
        ph_all = estimate_velocity_from_phase(ps, ph)
        return ph_all, "v-m", "mm/yr", ps_setref(ps)
    if value_type == "v-dmo":
        ph = dispatch_value_type(ps, "u-dmo", ref_ifg, ifg_list)[0]
        ph_all = estimate_velocity_from_phase(ps, ph)
        return ph_all, "v-dmo", "mm/yr", ps_setref(ps)
    
    # ============================================================
    raise ValueError(f"Unknown value_type: {value_type}")


"""
plot_ifg_grid.py
Grid-layout plotting helper for ps_plot (STAMPS)
"""

"""
ps_plot_ifg.py
Core Python translation of MATLAB ps_plot_ifg.m
Supports bg_flag 0 and 1 (lon/lat scatter) for u and v plotting
"""


def ps_plot_ifg(in_ph, bg_flag=1, col_rg=None, lon_rg=None, lat_rg=None, ext_data=None):
    """
    Plot PS interferogram values (Python / matplotlib)

    Parameters
    ----------
    in_ph : ndarray (n_ps,)
        Phase / value vector
    bg_flag : int
        0 = black bg, lon/lat
        1 = white bg, lon/lat
    col_rg : (min, max)
        Color limits
    lon_rg, lat_rg : (min, max)
        Geographic limits
    ext_data : dict or None
        External data overlay

    Returns
    -------
    ph_lims : (max, min)
    """

    # ----------------- parameters -----------------
    plot_pixel_m = getparm("plot_scatterer_size")[0]
    plot_pixel_size = getparm("plot_pixels_scatterer")[0]
    plot_color_scheme = getparm("plot_color_scheme")[0]
    lonlat_offset = getparm("lonlat_offset")[0]
    ref_radius = getparm("ref_radius")[0]
    ref_centre = getparm("ref_centre_lonlat")[0]

    ps = load_ps_data()
    lonlat = ps["lonlat"].copy()

    # apply lon/lat offset
    lonlat[:, 0] += lonlat_offset[0]
    lonlat[:, 1] += lonlat_offset[1]

    # ----------------- geographic windowing -----------------
    mask = np.ones(len(in_ph), dtype=bool)

    if lon_rg is not None and len(lon_rg) == 2:
        mask &= (lonlat[:, 0] >= lon_rg[0]) & (lonlat[:, 0] <= lon_rg[1])

    if lat_rg is not None and len(lat_rg) == 2:
        mask &= (lonlat[:, 1] >= lat_rg[0]) & (lonlat[:, 1] <= lat_rg[1])

    in_ph = in_ph[mask]
    lonlat = lonlat[mask]

    # ----------------- color scaling -----------------
    if col_rg is not None and len(col_rg) == 2:
        min_ph, max_ph = col_rg
    else:
        if np.isrealobj(in_ph):
            min_ph = np.nanmin(in_ph)
            max_ph = np.nanmax(in_ph)
        else:
            min_ph, max_ph = -np.pi, np.pi
            in_ph = np.angle(in_ph)

    if max_ph == min_ph:
        min_ph, max_ph = -np.pi, np.pi

    ph_range = max(max_ph - min_ph, np.finfo(float).eps)

    col_ix = np.floor((in_ph - min_ph) / ph_range * 63).astype(int)
    col_ix = np.clip(col_ix, 0, 63)
    print("ph_range:", ph_range)
    print("unique col_ix:", np.unique(col_ix[~np.isnan(col_ix)])[:10])
    
    # ----------------- colormap -----------------
    if plot_color_scheme.lower().startswith("gray"):
        cmap = plt.cm.gray_r
    elif plot_color_scheme.lower().startswith("inflation"):
        cmap = plt.cm.jet_r
    else:
        cmap = plt.cm.jet

    colors = cmap(np.linspace(0, 1, 64))

    # ----------------- background -----------------
    if bg_flag == 0:
        plt.gca().set_facecolor("black")
        plt.gca().tick_params(colors="white")
    else:
        plt.gca().set_facecolor("white")

    # ----------------- scatter plot -----------------
    valid = ~np.isnan(col_ix)
    plt.scatter(
        lonlat[valid, 0],
        lonlat[valid, 1],
        c=colors[col_ix[valid]],
        s=max(15, plot_pixel_size**2),
        marker="s",
        linewidths=0,
    )

    # ----------------- reference marker -----------------
    if ref_radius not in (np.inf, -np.inf):
        plt.plot(ref_centre[0], ref_centre[1], "k*", markersize=10)

    # ----------------- external data overlay (basic) -----------------
    if ext_data is not None and isinstance(ext_data, dict):
        if "lonlat" in ext_data:
            plt.scatter(
                ext_data["lonlat"][:, 0],
                ext_data["lonlat"][:, 1],
                marker="^",
                s=30,
                facecolor="none",
                edgecolor="k",
            )

    plt.axis("equal")
    plt.axis("tight")

    # ----------------- color limits -----------------
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(min_ph, max_ph)
    plt.colorbar(sm, ax=plt.gca())

    return max_ph, min_ph


def plot_ifg_grid(
    ph_disp,
    ps,
    lims,
    fig_name,
    plot_flag,
    n_x,
    textsize,
    textcolor,
    lon_rg,
    lat_rg,
    units,
    cbar_flag,
):
    """
    Create a STAMPS-style grid of interferogram plots.

    Parameters
    ----------
    ph_disp : ndarray (n_ps, n_ifg)
        Data to plot
    ps : dict
        ps structure from load_ps_data()
    lims : (min, max)
        Color limits
    fig_name : str
        Figure title
    plot_flag : int
        Background style selector
    n_x : int
        Max number of plots per row (0 = auto)
    textsize : int
        Text size
    textcolor : tuple or None
        Text color
    lon_rg, lat_rg : ignored here (handled in ps_plot_ifg)
    units : str
        Colorbar units
    cbar_flag : int
        Colorbar flag

    Returns
    -------
    fig : matplotlib.figure.Figure
    h_axes_all : dict
        Dictionary mapping ifg index → Axes
    """

    n_ifg_plot = ph_disp.shape[1]

    # ---------- figure geometry (STAMPS-like) ----------
    fig_ar = 4 / 3  # aspect ratio
    useratio = 1.0

    # crude spatial aspect estimate from lon/lat
    lonlat = ps["lonlat"]
    ar = (lonlat[:, 0].max() - lonlat[:, 0].min()) / max(
        lonlat[:, 1].max() - lonlat[:, 1].min(), 1e-6
    )

    if n_x == 0:
        n_y = int(np.ceil(np.sqrt(n_ifg_plot * ar / fig_ar)))
        n_x = int(np.ceil(n_ifg_plot / n_y))
    else:
        n_y = int(np.ceil(n_ifg_plot / n_x))

    d_x = useratio / n_x
    d_y = d_x / ar * fig_ar

    # manual axes positions (MATLAB-style)
    x_pos = np.linspace(0.05, 0.95 - d_x, n_x)
    y_pos = np.linspace(0.95 - d_y, 0.05, n_y)

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(fig_name)

    h_axes_all = {}

    # ---------- plot each interferogram ----------
    idx = 0
    for iy in range(n_y):
        for ix in range(n_x):
            if idx >= n_ifg_plot:
                break

            left = x_pos[ix]
            bottom = y_pos[iy]

            ax = fig.add_axes([left, bottom, d_x * 0.95, d_y * 0.95])

            h_axes_all[idx + 1] = ax  # MATLAB-style indexing

            idx += 1

    return fig, h_axes_all


'''


ax = fig.add_axes([x, y, w, h])
im = ax.imshow(data, vmin=lims[0], vmax=lims[1], origin="upper")

if show_cbar:
    fig.colorbar(im, ax=ax, orientation="horizontal")


from ps_parms_default import ps_parms_default
from getparm import getparm
from ps_plot import ps_plot

ps_parms_default()   # initialize parms.mat
value, _ = getparm("lambda")

ps_plot("u")
ps_plot("v")

ps_parms_default()
ps_plot("u")
ps_plot("v")

u_do_data, lims, _, _ = ps_plot("u-do", plot_flag=-1)

print(u_do_data.shape)   # (n_ps, n_ifg)
'''
