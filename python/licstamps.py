#!/usr/bin/env python3

################################################################################
# LiCStamps hacks
# by Milan Lazecky, 2026+, University of Leeds
#
# STAMPS functions being translated with support of MS Copilot (ChatGPT 5)
#
################################################################################


from datetime import datetime, timedelta, date
import os
import inspect
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.io import loadmat, savemat
from scipy.linalg import lstsq

def get_height_wrt_dem():
    scla = loadmat("scla2.mat", squeeze_me=True)
    ph_scla = scla["ph_scla"]            # rad
    K_ps = scla["K_ps_uw"]               # rad/m
    height_wrt_dem = ph_scla.mean(axis=1) / K_ps   # meters
    return height_wrt_dem



def export_ps_to_csv(outfile, correction_flag=None, use_detrend_for_coher=True):

    ps = load_ps_data()
    lambda_ = getparm("lambda")[0]

    # ---- correction mapping ----
    corr = correction_flag or ""
    u_type = "u" + (f"-{corr}" if corr else "")
    v_type = "v" + (f"-{corr}" if corr else "")

    # ---- load products ----
    u_data = ps_plot(u_type, plot_flag=-1)[0]      # rad, (n_ps, n_ifg)
    v_data = ps_plot(v_type, plot_flag=-1)[0]      # mm/yr

    # ---- HEIGHT from DEM ----
    psver = int(loadmat("psver.mat", squeeze_me=True)["psver"])
    hgt = loadmat(f"hgt{psver}.mat", squeeze_me=True)
    height_abs = hgt["hgt"]                        # meters

    # ---- HEIGHT WRT DEM (meters) ----
    scla = loadmat("scla2.mat", squeeze_me=True)
    ph_scla = scla["ph_scla"]                      # rad
    K_ps = scla["K_ps_uw"]                         # rad/m

    height_wrt_dem = ph_scla.mean(axis=1) / K_ps   # meters

    # ---- CUMULATIVE DISPLACEMENT ----
    cum_disp = 1000 * -np.sum(u_data, axis=1) * lambda_ / (4 * np.pi)

    # ---- temporal coherence ----
    coher = temporal_coherence(u_data, use_detrend_for_coher)

    # ---- dates (MATLAB datenum → Python) ----
    dates = [
        (
            datetime.fromordinal(int(d))
            + timedelta(days=float(d) - int(d))
            - timedelta(days=366)
        ).strftime("%Y-%m-%d")
        for d in ps["day"]
    ]

    # ---- build dataframe ----
    df = pd.DataFrame({
        "ID": np.arange(ps["n_ps"]) + 1,
        "LAT": np.round(ps["lonlat"][:, 1], 6),
        "LON": np.round(ps["lonlat"][:, 0], 6),
        "HEIGHT": np.round(height_abs, 2),
        "HEIGHT WRT DEM": np.round(height_wrt_dem, 2),
        "VEL": np.round(v_data[:, 0], 2),
        "CUM DISP": np.round(cum_disp, 2),
        "COHER": np.round(coher, 2),
    })

    # ---- add time series ----
    for i, date in enumerate(dates):
        df[date] = np.round(
            1000 * -u_data[:, i] * lambda_ / (4 * np.pi), 2
        )  # mm

    df.to_csv(outfile, index=False)
    print(f"Exported {len(df)} PS points to {outfile}")



"""
load_ps_data.py
Load STAMPS ps*.mat data in a Python-friendly way
"""


"""
ps_deramp.py
Faithful Python translation of MATLAB ps_deramp.m (STAMPS)

Original: David Bekaert
Python translation: SM & SB compatible
"""


def estimate_velocity_from_phase(ps, ph_uw):
    """
    Estimate LOS velocity from unwrapped phase time series
    ph_uw shape: (n_ps, n_ifg)
    """

    ref_ps = ps_setref(ps)
    lambda_ = getparm("lambda")[0]

    # remove master IFG
    unwrap_ix = [i for i in range(ps["n_ifg"]) if i != ps["master_ix"]]
    ph_uw = ph_uw[:, unwrap_ix]
    day = ps["day"][unwrap_ix]

    # reference
    ph_uw = ph_uw - ph_uw[ref_ps, :].mean(axis=0)

    # design matrix
    G = np.c_[np.ones_like(day), day - ps["master_day"]]

    m, *_ = lstsq(G, ph_uw.T)

    vel = -m[1, :] * 365.25 / (4 * np.pi) * lambda_ * 1000
    return vel.reshape(-1, 1)



def temporal_coherence(ph_uw, use_detrend=True):
    """
    Compute temporal coherence per PS
    ph_uw shape: (n_ps, n_ifg)
    """
    n_itf = ph_uw.shape[1]
    coher = np.zeros(ph_uw.shape[0])

    for i in range(ph_uw.shape[0]):
        ts = ph_uw[i, :]
        if use_detrend:
            ts = detrend(ts)
        coher[i] = np.abs(np.sum(np.exp(-1j * ts)) / n_itf)

    return coher


def ps_deramp(ps, ph_all, degree=None):
    """
    Deramp interferograms using a polynomial surface.

    Parameters
    ----------
    ps : dict
        STAMPS ps structure (from load_ps_data)
    ph_all : ndarray, shape (n_ps, n_ifg)
        Phase data to deramp
    degree : float or int, optional
        Polynomial degree (1, 1.5, 2, 3).
        If None, tries to load deramp_degree.mat, else defaults to 1.

    Returns
    -------
    ph_all : ndarray
        Deramped phase data
    ph_ramp : ndarray
        Estimated ramp for each interferogram
    """

    print("Deramping computed on the fly.")

    # ---------------------------------------------------------
    # Determine deramp degree
    # ---------------------------------------------------------
    if degree is None:
        if os.path.isfile("deramp_degree.mat"):
            try:
                deg = loadmat("deramp_degree.mat", squeeze_me=True)
                degree = float(deg["degree"])
                print("Found deramp_degree.mat file will use that value to deramp")
            except Exception:
                degree = 1
        else:
            degree = 1

    # ---------------------------------------------------------
    # Ensure ps.n_ifg consistent with ph_all
    # ---------------------------------------------------------
    n_ps, n_ifg = ph_all.shape
    if ps["n_ifg"] != n_ifg:
        ps["n_ifg"] = n_ifg

    # ---------------------------------------------------------
    # Build design matrix A
    # Coordinates in km (MATLAB: ps.xy(:,2:3) / 1000)
    # ---------------------------------------------------------
    x = ps["xy"][:, 1] / 1000.0
    y = ps["xy"][:, 2] / 1000.0
    ones = np.ones_like(x)

    if degree == 1:
        # z = ax + by + c
        A = np.column_stack([x, y, ones])
        print("**** z = ax + by + c")

    elif degree == 1.5:
        # z = ax + by + cxy + d
        A = np.column_stack([x, y, x * y, ones])
        print("**** z = ax + by + cxy + d")

    elif degree == 2:
        # z = ax² + by² + cxy + d
        A = np.column_stack([x**2, y**2, x * y, ones])
        print("**** z = ax^2 + by^2 + cxy + d")

    elif degree == 3:
        # z = ax³ + by³ + cx²y + dy²x + ex² + fy² + gxy + h
        A = np.column_stack([
            x**3,
            y**3,
            x**2 * y,
            y**2 * x,
            x**2,
            y**2,
            x * y,
            ones
        ])
        print("**** z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h")

    else:
        raise ValueError(f"Unsupported deramp degree: {degree}")

    A = A.astype(np.float64)

    # ---------------------------------------------------------
    # Deramp each interferogram
    # ---------------------------------------------------------
    ph_ramp = np.full_like(ph_all, np.nan, dtype=np.float64)
    ph_all = ph_all.astype(np.float64, copy=True)

    for k in range(ps["n_ifg"]):
        col = ph_all[:, k]
        valid = ~np.isnan(col)

        # MATLAB condition: ps.n_ps - sum(ix) > 5
        if np.sum(valid) > 5:
            coeff, *_ = lstsq(A[valid, :], col[valid])
            ramp = A @ coeff
            ph_ramp[:, k] = ramp
            ph_all[:, k] = col - ramp
        else:
            print(f"Ifg {k + 1} is not deramped")

    return ph_all, ph_ramp


def load_ps_data():
    """
    Load ps*.mat according to psver.

    Returns
    -------
    ps : dict
        Dictionary with keys matching MATLAB ps structure
    """

    # ---------- load psver ----------
    if not os.path.isfile("psver.mat"):
        raise FileNotFoundError("psver.mat not found")

    psver_mat = loadmat("psver.mat", squeeze_me=True)
    if "psver" not in psver_mat:
        raise KeyError("psver not found in psver.mat")

    psver = int(psver_mat["psver"])

    # ---------- load ps file ----------
    psname = f"ps{psver}.mat"
    if not os.path.isfile(psname):
        raise FileNotFoundError(f"{psname} not found")

    ps_raw = loadmat(psname, squeeze_me=True)

    # ---------- strip MATLAB metadata ----------
    ps = {k: v for k, v in ps_raw.items() if not k.startswith("__")}

    # ---------- required fields for u and v ----------
    required = [
        "day",
        "master_day",
        "xy",
        "lonlat",
        "n_ps",
        "n_ifg",
        "ll0",
    ]

    for key in required:
        if key not in ps:
            raise KeyError(f"{key} missing from {psname}")

    # ---------- optional fields ----------
    # Present only in some processing chains
    if "ifgday" in ps:
        ps["ifgday"] = np.atleast_2d(ps["ifgday"])
    else:
        ps["ifgday"] = None  # explicitly mark missing

    # ---------- normalize shapes ----------
    ps["day"] = np.atleast_1d(ps["day"])
    ps["lonlat"] = np.atleast_2d(ps["lonlat"])
    ps["xy"] = np.atleast_2d(ps["xy"])

    ps["n_ps"] = int(ps["n_ps"])
    ps["n_ifg"] = int(ps["n_ifg"])

    # MATLAB-compatible master index
    ps["master_ix"] = int(np.sum(ps["day"] < ps["master_day"]))

    return ps


"""
ps_setref.py
Python translation of MATLAB ps_setref.m (STAMPS)
"""

"""
llh2local.py
Python translation of MATLAB llh2local.m (Cervelli / Murray / Hooper)

Converts longitude/latitude to local Cartesian coordinates (km)
relative to a given origin using WGS-84.
"""


def llh2local(llh, origin):
    """
    Convert longitude/latitude to local XY coordinates.

    Parameters
    ----------
    llh : array_like, shape (2,N) or (3,N)
        [lon; lat; (height)] in decimal degrees.
        Height is ignored.
    origin : array_like, shape (2,) or (3,)
        [lon, lat, (height)] in decimal degrees.

    Returns
    -------
    xy : ndarray, shape (2,N)
        Local coordinates in kilometers.
        x = East, y = North
    """

    # WGS-84 ellipsoid constants
    a = 6378137.0
    e = 0.08209443794970

    llh = np.asarray(llh, dtype=float)
    origin = np.asarray(origin, dtype=float)

    if llh.shape[0] < 2:
        raise ValueError("llh must have at least lon and lat rows")

    # Keep only lon/lat, convert to radians
    lon = np.deg2rad(llh[0, :])
    lat = np.deg2rad(llh[1, :])

    lon0 = np.deg2rad(origin[0])
    lat0 = np.deg2rad(origin[1])

    xy = np.zeros((2, lon.size))

    # Identify latitude != 0
    z = lat != 0.0

    # ---------- main projection ----------
    if np.any(z):
        dlambda = lon[z] - lon0

        M = a * (
            (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat[z]
            - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * lat[z])
            + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat[z])
            - (35 * e**6 / 3072) * np.sin(6 * lat[z])
        )

        M0 = a * (
            (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat0
            - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * lat0)
            + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat0)
            - (35 * e**6 / 3072) * np.sin(6 * lat0)
        )

        N = a / np.sqrt(1 - e**2 * np.sin(lat[z])**2)
        E = dlambda * np.sin(lat[z])

        cot_lat = 1 / np.tan(lat[z])

        xy[0, z] = N * cot_lat * np.sin(E)
        xy[1, z] = M - M0 + N * cot_lat * (1 - np.cos(E))

    # ---------- special case: latitude == 0 ----------
    if np.any(~z):
        dlambda = lon[~z] - lon0
        xy[0, ~z] = a * dlambda
        xy[1, ~z] = -(
            a * (
                (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat0
                - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * lat0)
                + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat0)
                - (35 * e**6 / 3072) * np.sin(6 * lat0)
            )
        )

    # Convert meters → kilometers
    return xy / 1000.0


def ps_setref(ps2=None):
    """
    Find reference PS indices.

    Returns
    -------
    ref_ps : ndarray or int
        Indices of reference PS, or 0 if none
    """

    # ---------- load ps structure ----------
    psver = int(loadmat("psver.mat", squeeze_me=True)["psver"])
    psname = f"ps{psver}.mat"

    if ps2 is None:
        ps = loadmat(psname, squeeze_me=True)
        ps2 = ps
    else:
        # merge ll0 and n_ps from disk PS
        ps_temp = loadmat(psname, squeeze_me=True)
        ps2["ll0"] = ps_temp["ll0"]
        ps2["n_ps"] = ps2["lonlat"].shape[0]

    # ---------- check for ref_x / ref_y (legacy) ----------
    ref_lon, parmname = getparm("ref_x")

    if parmname == "ref_x":
        ref_x = getparm("ref_x")[0]
        ref_y = getparm("ref_y")[0]

        ref_ps = np.where(
            (ps2["xy"][:, 1] > ref_x[0]) &
            (ps2["xy"][:, 1] < ref_x[1]) &
            (ps2["xy"][:, 2] > ref_y[0]) &
            (ps2["xy"][:, 2] < ref_y[1])
        )[0]

    else:
        ref_lon = getparm("ref_lon")[0]
        ref_lat = getparm("ref_lat")[0]
        ref_centre_lonlat = getparm("ref_centre_lonlat")[0]
        ref_radius = getparm("ref_radius")[0]

        if ref_radius == -np.inf:
            return 0

        ref_ps = np.where(
            (ps2["lonlat"][:, 0] > ref_lon[0]) &
            (ps2["lonlat"][:, 0] < ref_lon[1]) &
            (ps2["lonlat"][:, 1] > ref_lat[0]) &
            (ps2["lonlat"][:, 1] < ref_lat[1])
        )[0]

        # circular restriction
        if ref_radius < np.inf and ref_ps.size > 0:
            ref_xy = llh2local(np.array(ref_centre_lonlat).reshape(2, 1),
                               ps2["ll0"]) * 1000
            xy = llh2local(ps2["lonlat"][ref_ps, :].T, ps2["ll0"]) * 1000

            dist_sq = (xy[0, :] - ref_xy[0, 0]) ** 2 + \
                      (xy[1, :] - ref_xy[1, 0]) ** 2

            ref_ps = ref_ps[dist_sq <= ref_radius ** 2]

    # ---------- fallback ----------
    if ref_ps is None or len(ref_ps) == 0:
        if ps2 is not None:
            print("None of your external data points have a reference, all are set as reference.")
            return np.arange(ps2["n_ps"])

    if ps2 is None:
        if ref_ps == 0:
            print("No reference set")
        else:
            print(f"{len(ref_ps)} ref PS selected")

    return ref_ps



def logit(logmsg=0, whereto=0, parent_flag=0):
    """
    LOGIT write message to log file and/or stdout

    logit(LOGMSG, WHERETO, PARENT_FLAG)

    WHERETO:
        0 -> stamps.log + stdout (default)
        1 -> stamps.log only
        2 -> stdout only
        3 -> debug.log only
        string -> filename + stdout

    PARENT_FLAG:
        0 -> current directory
        1 -> parent directory
    """

    # ---------- function name (dbstack equivalent) ----------
    stack = inspect.stack()
    if len(stack) > 1:
        fname = stack[1].function.upper()
    else:
        fname = "Command line"

    # ---------- numeric logmsg mapping ----------
    if isinstance(logmsg, (int, float)):
        if logmsg == 0:
            logmsg = "Starting"
        elif logmsg == 1:
            logmsg = "Finished"
        else:
            logmsg = str(logmsg)

    logmsg = str(logmsg)

    # Remove trailing '\n' if present (MATLAB behavior)
    if logmsg.endswith("\\n"):
        logmsg = logmsg[:-2]

    # ---------- determine logfile ----------
    if not isinstance(whereto, (int, float)):
        logfile = str(whereto)
        whereto = 0
    else:
        logfile = "STAMPS.log"

    debugfile = "DEBUG.log"

    if parent_flag == 1:
        logfile = os.path.join("..", logfile)
        debugfile = os.path.join("..", debugfile)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} {fname:<16} {logmsg}"

    # ---------- write to stamps.log ----------
    if whereto < 2:
        try:
            with open(logfile, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    # ---------- stdout ----------
    if whereto in (0, 2):
        print(f"{fname}: {logmsg}")

    # ---------- debug.log ----------
    if whereto == 3:
        try:
            with open(debugfile, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


def getparm(parmname=None, printflag=0):
    """
    Python equivalent of MATLAB getparm.m

    Parameters
    ----------
    parmname : str or None
        Parameter name (partial match allowed)
    printflag : int
        If non-zero, log the parameter value

    Returns
    -------
    value, parmname
    """

    parmfile = "parms"
    localparmfile = "localparms"

    # ---- Load parms.mat ----
    if os.path.isfile("./parms.mat"):
        parms = loadmat("parms.mat", squeeze_me=True)
    elif os.path.isfile("../parms.mat"):
        parmfile = "../parms"
        parms = loadmat("../parms.mat", squeeze_me=True)
    else:
        raise FileNotFoundError("parms.mat not found")

    # Remove MATLAB metadata
    parms = {
        k: v for k, v in parms.items()
        if not k.startswith("__")
    }

    # ---- Load localparms.mat ----
    if os.path.isfile("localparms.mat"):
        localparms = loadmat("localparms.mat", squeeze_me=True)
        localparms = {
            k: v for k, v in localparms.items()
            if not k.startswith("__")
        }
    else:
        localparms = {"Created": date.today().isoformat()}

    # ---- No arguments: print parms ----
    if parmname is None:
        for key in sorted(parms.keys()):
            print(f"{key}: {parms[key]}")
        if len(localparms) > 1:
            print("\nlocalparms:")
            for key, value in localparms.items():
                print(f"{key}: {value}")
        return None, None

    # ---- Partial-name matching (MATLAB strmatch) ----
    matches = [k for k in parms.keys() if k.startswith(parmname)]

    if len(matches) > 1:
        raise ValueError(f"Parameter {parmname}* is not unique")
    elif len(matches) == 0:
        return None, None

    parmname = matches[0]

    # ---- localparms override ----
    if parmname in localparms:
        value = localparms[parmname]
    else:
        value = parms[parmname]

    # ---- Print/log if requested ----
    if printflag != 0:
        if isinstance(value, np.ndarray):
            value_str = " ".join(str(v) for v in value.flatten())
        elif isinstance(value, (int, float, np.number)):
            value_str = str(value)
        else:
            value_str = f"'{value}'"

        msg = f"{parmname}={value_str}"
        logit(msg)

    return value, parmname


"""
ps_parms_default.py
Python translation of MATLAB ps_parms_default.m (STAMPS)

Andy Hooper original: June 2006+
Python translation: faithful semantics for getparm / ps_plot (u, v)
"""




def _load_parms():
    """Load parms.mat or create default."""
    if os.path.isfile("parms.mat"):
        return "parms.mat", loadmat("parms.mat", squeeze_me=True)
    elif os.path.isfile("../parms.mat"):
        return "../parms.mat", loadmat("../parms.mat", squeeze_me=True)
    else:
        return "parms.mat", {"Created": date.today().isoformat(), "small_baseline_flag": "n"}


def _clean(matdict):
    """Remove MATLAB metadata."""
    return {k: v for k, v in matdict.items() if not k.startswith("__")}


def ps_parms_default():
    parmfile, parms = _load_parms()
    parent_flag = parmfile.startswith("../")
    parms = _clean(parms)

    before_fields = set(parms.keys())
    num_fields = len(before_fields)

    # ---------- helpers ----------
    def ensure(name, value):
        if name not in parms:
            parms[name] = value

    sb = parms.get("small_baseline_flag", "n").lower() == "y"

    # ---------- defaults ----------
    ensure("max_topo_err", 20)
    ensure("quick_est_gamma_flag", "y")
    ensure("select_reest_gamma_flag", "y")
    ensure("filter_grid_size", 50)
    ensure("filter_weighting", "P-square")
    ensure("gamma_change_convergence", 0.005)
    ensure("gamma_max_iterations", 3)
    ensure("slc_osf", 1)
    ensure("clap_win", 32)
    ensure("clap_low_pass_wavelength", 800)
    ensure("clap_alpha", 1)
    ensure("clap_beta", 0.3)
    ensure("select_method", "DENSITY")

    ensure("density_rand", 2 if sb else 20)
    ensure("percent_rand", 1 if sb else 20)

    ensure("gamma_stdev_reject", 0)
    ensure("weed_time_win", 730)
    ensure("weed_max_noise", np.inf)
    ensure("weed_standard_dev", np.inf if sb else 1.0)
    ensure("weed_zero_elevation", "n")
    ensure("weed_neighbours", "n")

    ensure("unwrap_method", "3D_QUICK" if sb else "3D")
    ensure("unwrap_patch_phase", "n")

    # unwrap_ifg_index → drop_ifg_index
    if "unwrap_ifg_index" in parms:
        try:
            ps = loadmat("ps2.mat", squeeze_me=True)
        except Exception:
            try:
                ps = loadmat("ps1.mat", squeeze_me=True)
            except Exception:
                ps = None

        if ps is not None and parms["unwrap_ifg_index"] != "all":
            parms["drop_ifg_index"] = list(
                set(range(1, int(ps["n_ifg"]) + 1))
                - set(parms["unwrap_ifg_index"])
            )
        del parms["unwrap_ifg_index"]
        num_fields = 0

    ensure("drop_ifg_index", [])
    ensure("unwrap_la_error_flag", "y")
    ensure("unwrap_spatial_cost_func_flag", "n")
    ensure("unwrap_prefilter_flag", "y")
    ensure("unwrap_grid_size", 200)
    ensure("unwrap_gold_n_win", 32)
    ensure("unwrap_alpha", 8)
    ensure("unwrap_time_win", 730)
    ensure("unwrap_gold_alpha", 0.8)
    ensure("unwrap_hold_good_values", "n")

    ensure("scla_drop_index", [])
    ensure("scn_wavelength", 100)
    ensure("scn_time_win", 365)
    ensure("scn_deramp_ifg", [])
    ensure("scn_kriging_flag", "n")

    ensure("ref_lon", [-np.inf, np.inf])
    ensure("ref_lat", [-np.inf, np.inf])
    ensure("ref_centre_lonlat", [0, 0])
    ensure("ref_radius", np.inf)
    ensure("ref_velocity", 0)

    ensure("n_cores", 1)

    ensure("plot_dem_posting", 90)
    ensure("plot_scatterer_size", 120)
    ensure("plot_pixels_scatterer", 3)
    ensure("plot_color_scheme", "inflation")

    ensure("shade_rel_angle", [90, 45])
    ensure("lonlat_offset", [0, 0])

    ensure("merge_resample_size", 100 if sb else 0)
    ensure("merge_standard_dev", np.inf)

    ensure("scla_method", "L2")
    ensure("scla_deramp", "n")

    # ---------- wavelength ----------
    if "lambda" not in parms:
        for path in ["lambda.1.in", "../lambda.1.in", "../../lambda.1.in"]:
            if os.path.isfile(path):
                parms["lambda"] = float(np.loadtxt(path))
                break
        else:
            parms["lambda"] = np.nan

    # ---------- heading ----------
    if "heading" not in parms:
        for path in ["heading.1.in", "../heading.1.in", "../../heading.1.in"]:
            if os.path.isfile(path):
                parms["heading"] = float(np.loadtxt(path))
                break
        else:
            parms["heading"] = np.nan

    ensure("sb_scla_drop_index", [])
    ensure("subtr_tropo", "n")
    ensure("tropo_method", "a_l")

    # ---------- processor ----------
    if "insar_processor" not in parms:
        processor = None
        for path in ["processor.txt", "../processor.txt", "../../processor.txt"]:
            if os.path.isfile(path):
                with open(path) as f:
                    processor = f.read().strip()
                break

        parms["insar_processor"] = processor if processor else "doris"
        if processor not in (None, "gamma", "doris"):
            print("WARNING: processor not supported (gamma, doris only)")

    # ---------- save & log ----------
    after_fields = set(parms.keys())
    if len(after_fields) != num_fields:
        try:
            savemat(parmfile, parms)
            for k in after_fields - before_fields:
                v = parms[k]
                if isinstance(v, (int, float, np.number)):
                    logit(f"{k} = {v}", 0, parent_flag)
                else:
                    logit(f"{k} = {v}", 0, parent_flag)
        except Exception:
            print("Warning: missing parameters could not be updated (no write access)")
