#!/usr/bin/env python3

################################################################################
# LiCSAR Time Series Tools
# by Milan Lazecky, 2024+, University of Leeds
#
# some TS post-processing tools (probably to be later implemented as part of LiCSBAS etc.)
#
################################################################################

from lics_unwrap import *
import re
#import datetime as dt
#t1 = dt.datetime(2016, 7, 1)
#t2 = dt.datetime(2017, 7, 31)

def grep1line(arg,filename):
    file = open(filename, "r")
    res=''
    for line in file:
        if re.search(arg, line):
            res=line
            break
    file.close()
    if res:
        res = res.split('\n')[0]
    return res


def load_licsbas_cumh5_as_xrda(cumfile):
    ''' Loads cum.h5 (now only cum layer) as standard xr.DataArray (in lon/lat)'''
    cum = xr.load_dataset(cumfile)
    #
    sizex = len(cum.vel[0])
    sizey = len(cum.vel)
    #
    lon = cum.corner_lon.values + cum.post_lon.values * np.arange(sizex) - 0.5 * float(cum.post_lon)
    lat = cum.corner_lat.values + cum.post_lat.values * np.arange(sizey) + 0.5 * float(cum.post_lat)  # maybe needed? yes! for gridline/AREA that is default in rasterio...
    #
    time = np.array(([dt.datetime.strptime(str(imd), '%Y%m%d') for imd in cum.imdates.values]))
    #
    velxr = xr.DataArray(cum.vel.values.reshape(sizey, sizex), coords=[lat, lon], dims=["lat", "lon"])
    # LiCSBAS uses 0 instead of nans...
    velxr = velxr.where(velxr != 0)
    velxr.attrs['unit'] = 'mm/year'
    # vinterceptxr = xr.DataArray(cum.vintercept.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
    #
    cumxr = xr.DataArray(cum.cum.values, coords=[time, lat, lon], dims=["time", "lat", "lon"])
    cumxr.attrs['unit'] = 'mm'
    refarea = str(cum.refarea.values)
    # x is first...
    refx1 = int(refarea.split('/')[0].split(':')[0])
    refx2 = int(refarea.split('/')[0].split(':')[1])
    refy1 = int(refarea.split('/')[1].split(':')[0])
    refy2 = int(refarea.split('/')[1].split(':')[1])
    refx=int((refx2+refx1)/2)
    refy = int((refy2 + refy1) / 2)
    cumxr.attrs['ref_lon'] = cumxr.lon.values[refx]
    cumxr.attrs['ref_lat'] = cumxr.lat.values[refy]
    return cumxr


def correct_cum_from_tifs(cumhdfile, tifdir = 'GEOC.EPOCHS', ext='geo.iono.code.tif', tif_scale2mm = 1, outputhdf = None, directcorrect = True):
    ''' This will load the cum.h5 and either correct cum layer (if directcorrect==True) or add new data var to the cube (if not directcorrect)

    Args:
        cumhdfile (str): input cum.h5 file
        tifdir (str): where to find corrections. Must be in epoch subdirs, e.g. tifdir/20240216/20240216.ext
        ext (str):    what extension is used for the correction files after the epochdate, e.g. 'geo.iono.code.tif', 'tide.geo.tif'
        tif_scale2mm (float):  for iono [rad]: 55.465/(4*np.pi), for SET [m]: 1000, for sltd [rad] same as iono but opposite sign (real delay)
        outputhdf (str):   if None, will overwrite the input file, otherwise will export to this filename (functional: H5, NC, possibly: zarr)
        directcorrect (bool): if True, it will directly reduce the cum data, otherwise it stores it to the datacube
    
    Returns:
        bool:  True if all went ok
    '''
    if not outputhdf:
        outputhdf = cumhdfile
    print('loading LiCSBAS datacube')
    cumxr = load_licsbas_cumh5_as_xrda(cumhdfile)
    print('loading external corrections')
    cumxr = cumcube_remove_from_tifs(cumxr, tifdir, ext, tif_scale2mm, only_load_ext = not directcorrect)
    if type(cumxr) == type(False):
        print('ERROR - probably the correction did not exist for some epochs. Cancelling')
        return False
    cumh = xr.load_dataset(cumhdfile)
    if directcorrect:
        cumh.cum.values = cumh.cum.values - cumxr.values
    else:
        newcumname = 'external_data'
        codes = ['iono', 'tide', 'icams']
        for c in codes:
            if ext.find(c)>-1:
                newcumname = c
        if newcumname in cumh:
            # rename to extension
            newcumname = ext
        if newcumname in cumh:
            print('WARNING, the layer "'+newcumname+'"already existed in the datacube! Will overwrite it now')
        print('storing as variable: '+newcumname)
        cumh[newcumname]=cumh.cum.copy()
        cumh[newcumname].values=cumxr.values
    print('saving to: '+outputhdf)
    cumh.to_netcdf(outputhdf)
    return True


# def check_complete_set(imdates, epochsdir, ext='geo.iono.code.tif')
def cumcube_remove_from_tifs(cumxr, tifdir = 'GEOC.EPOCHS', ext='geo.iono.code.tif', tif_scale2mm = 1, only_load_ext = False):
    ''' Correct directly from tifs, no need to store in cubes.
    NOTE - you can also just load the exts into the cumcube without removing anything..
    (in any case, values are referred temporally to the first epoch)
    
    Args:
        cumxr (xr.DataArray): only cum
        tifdir:
        ext:
        tif_scale2mm:  for iono [rad]: (0.055465*1000)/(4*np.pi), for SET [m]: 1/1000
        only_load_ext:  would only load the ext files in the cube and return it (no removal!)
        
    Returns:
        xr.DataArray: corrected cum values (only_load_ext=False) or only loaded corrections
    '''
    #if check_complete_set(cumxr.time.values)
    #times = cumxr.time.values
    reflon, reflat = cumxr.attrs['ref_lon'], cumxr.attrs['ref_lat']
    #
    firstepvals = 0
    leneps = len(cumxr)
    for i in range(leneps): # times first coord..
        print('  Running for {0:6}/{1:6}th epoch'.format(i+1, leneps), flush=True, end='\r')
        cumepoch = cumxr[i]
        epoch = str(cumepoch.time.values).split('T')[0].replace('-','')
        extif = os.path.join(tifdir, epoch, epoch+'.'+ext)
        if not os.path.exists(extif):
            print('\n\r WARNING: no correction available for epoch '+epoch+'. Filling with NaNs')
            extepoch = cumepoch.copy() * np.nan
            extepoch.attrs.clear()
        else:
            extepoch = load_tif2xr(extif)
            extepoch = extepoch.where(extepoch != 0) # just in case...
            extepoch = extepoch * tif_scale2mm
            extepoch = extepoch.interp_like(cumepoch, method='linear') # CHECK!
            extepoch = extepoch - extepoch.sel(lon=reflon, lat=reflat, method='nearest') # could be done better though
        if i == 0:
            firstepvals = extepoch.fillna(0).values
        # here we do diff w.r.t. first epoch
        extepoch.values = extepoch.values - firstepvals
        # mask that - not needed (?)
        #extepoch = extepoch.where(~np.isnan(cumxr[i]))
        if only_load_ext:
            cumxr.values[i] = extepoch.values
        else:
            cumxr.values[i] = cumxr.values[i] - extepoch.values
    print('\n\r  done')
    #if only_load_ext:
    #    cumxr = cumxr-cumxr[0] #.cumsum(axis=0)
    #    cumxr = cumxr.cumsum(axis=0)-cumxr[0]
    return cumxr


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
