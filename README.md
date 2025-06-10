# LiCSAR Extra
Tools for advanced processing of LiCSAR data (starting from geotiffs).
Primarily for unwrapping interferograms (integrated to [LiCSBAS](https://github.com/comet-licsar/licsbas)),
but further used as repository for experimental functions requiring atypical or heavy python libraries.  

While supported by NERC Large Grant project on "Looking inside the continents from Space" (NE/K010867/1), some functionality was developed within activities on
ESA Open SAR Library 4000140600/23/I-DT extension, recognised as the [AlignSAR InSAR Time Series extension](https://github.com/AlignSAR/alignSAR/tree/main/alignsar_extension_InSAR_TS).

You may install this set of tools e.g. using pip by:  
`
pip install git+https://github.com/comet-licsar/licsar_extra.git
`

Then, to test it, try e.g.:  
```
from licsar_extra import lics_unwrap
help(lics_unwrap.process_ifg_pair)
```

## lics_unwrap.py
The primary tool of the 'licsar_extra' library, improving the phase unwrapping, as was (partly) described in [IGARSS 2022](https://ieeexplore.ieee.org/document/9884337) and [SARWatch 2023](https://www.sciencedirect.com/science/article/pii/S187705092401679X).
As additional prerequisite, you must install [SNAPHU](http://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/).

Then, having at least geotiffs of the wrapped interferogram and coherence, you can unwrap e.g. using:
```
phatif = '20xxxxxx_20xxxxxx.unfiltered_pha.geo.tif'
cohtif = '20xxxxxx_20xxxxxx.coh.geo.tif'
outunw = '20xxxxxx_20xxxxxx.unw.geo.tif'
unwcube = lics_unwrap.process_ifg_pair(phatif, cohtif,
        ml = 1, fillby = 'nearest', thres = 0.35, 
        lowpass = False, goldstein = True, specmag = True,
        outtif = outunw, cascade = False) # try cascade True/False, it has strong effect
```

## lics_tstools.py
Useful functions to operate with the LiCSBAS datacube.


## licsbas_mintpy_PMM.py
Functionality directly imported from [MintPy](https://github.com/insarlab/MintPy) allowing LiCSBAS to calculate/correct plate motion velocity plane,
effectively fixing towards Eurasian tectonic plate.