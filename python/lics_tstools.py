#!/usr/bin/env python3

################################################################################
# LiCSAR Time Series Tools
# by Milan Lazecky, 2024+, University of Leeds
#
# some TS post-processing tools (probably to be later implemented as part of LiCSBAS etc.)
#
################################################################################

from lics_unwrap import *

#import datetime as dt
#t1 = dt.datetime(2016, 7, 1)
#t2 = dt.datetime(2017, 7, 31)

def calculate_defo_period(xrds, t1, t2, defolabel, medwin = 5, inside_period = True):
    '''Extracts deformation from cum layer within given time period <t1,t2>, i.e. including t1,t2, as diff between median of medwin head/tail (see ~inside_period for opposite)
    
    Args:
        xrds (xr.Dataset): as loaded from LiCSBAS2nc.py. Must contain ['cum'] dataarray with time, lat, lon dimensions
        t1, t2 (dt.datetime)
        defolabel (str): e.g. 'July 2016-July 2017'
        medwin (int): window for (1-dimensional in time) median filter
        inside_period (bool): if True, we search (apply the medwin to get avg defo value) in time period (t1,t2), otherwise in (>t2) minus (<t1)
    Returns:
        updated xrds (with dimension defolabel and variable defo having this output)
    '''
    if inside_period:
        ootsel = xrds['cum'].sel(time=slice(t1, t2))
        avgt1 = ootsel.head(time=medwin).median(dim=['time']) #['lon','lat'])
        avgt2 = ootsel.tail(time=medwin).median(dim=['time'])
    else:
        ootsel = xrds['cum'].sel(time=slice(None, t1))
        avgt1 = ootsel.tail(time=medwin).median(dim=['time']) #['lon','lat'])
        ootsel = xrds['cum'].sel(time=slice(t2, None))
        avgt2 = ootsel.head(time=medwin).median(dim=['time'])
    # prepare dataarray and set dimension label
    gdefo = (xrds['vel'] * np.nan).copy()
    gdefo.attrs['unit'] = 'mm'
    gdefo.values = (avgt2 - avgt1).values
    gdefo = gdefo - gdefo.median()  # normalise around 0
    gdefo = gdefo.expand_dims(dim={"defolabel": [defolabel]})
    if not 'defolabel' in xrds.dims:   #assuming that also 'defo' exists!
        xrds['defo'] = gdefo #xrds['defo'].expand_dims(dim={"defolabel": [defolabel]})
    else:
        mergeda = xr.concat([xrds['defo'], gdefo], dim="defolabel")
        xrds = xrds.drop_dims('defolabel')
        xrds['defo'] = mergeda
    return xrds
