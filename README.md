# LiCSAR Extra

Tools for advanced processing of LiCSAR data (starting from geotiffs).
Primarily for unwrapping interferograms (integrated into [LiCSBAS](https://github.com/comet-licsar/licsbas)),
but also used as a repository for experimental functions requiring atypical or heavy python libraries.

Supported by NERC Large Grant project "Looking inside the continents from Space" (NE/K010867/1);
some functionality was developed within activities on ESA Open SAR Library (4000140600/23/I-DT extension),
recognised as the [AlignSAR InSAR Time Series extension](https://github.com/AlignSAR/alignSAR/tree/main/alignsar_extension_InSAR_TS).

## Installation

One single command is enough — all dependencies (including the [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/)
unwrapping binary, which is pulled in automatically as the `snaphu` PyPI package) are installed together:

```bash
pip install git+https://github.com/comet-licsar/licsar_extra.git
```

To update later:

```bash
pip install --upgrade --force-reinstall git+https://github.com/comet-licsar/licsar_extra.git
```

> **Note:** the `snaphu` dependency is provided by [snaphu-py](https://github.com/isce-framework/snaphu-py),
> which bundles the official SNAPHU executable (Linux/macOS wheels; a C compiler is only needed when building from source).
> If you use conda, you can alternatively install it with `conda install -c conda-forge snaphu`.
> GDAL python bindings are supplied as precompiled wheels via the `gdal-wheel` package, so no system GDAL is required.

Optional speedups (used automatically when present): `dask`, `pyinterp`.

### Quick test

```python
from licsar_extra import lics_unwrap
help(lics_unwrap.process_ifg_pair)
```

## Modules

### lics_unwrap.py

The primary tool of `licsar_extra` — improved phase unwrapping, as (partly) described in
[IGARSS 2022](https://ieeexplore.ieee.org/document/9884337) and
[SARWatch 2023](https://www.sciencedirect.com/science/article/pii/S187705092401679X).

Having at least the geotiffs of the wrapped interferogram and coherence, you can unwrap e.g. using:

```python
phatif  = '20xxxxxx_20xxxxxx.unfiltered_pha.geo.tif'
cohtif  = '20xxxxxx_20xxxxxx.coh.geo.tif'
outunw  = '20xxxxxx_20xxxxxx.unw.geo.tif'
unwcube = lics_unwrap.process_ifg_pair(phatif, cohtif,
        ml = 1, fillby = 'nearest', thres = 0.35,
        lowpass = False, goldstein = True, specmag = True,
        outtif = outunw, cascade = False)   # try cascade True/False, it has strong effect
```

Note: if `dask` and `pyinterp` are present, they will be used (optionally) to speed up processing.

### lics_tstools.py

Useful functions to operate with the LiCSBAS datacube.

### licsbas_mintpy_PMM.py

Functionality directly imported from [MintPy](https://github.com/insarlab/MintPy), allowing LiCSBAS to
calculate/correct a plate motion velocity plane — effectively fixing the solution towards the
Eurasian tectonic plate.
