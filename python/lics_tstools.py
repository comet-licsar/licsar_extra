#!/usr/bin/env python3

################################################################################
# LiCSAR Time Series Tools
# by Milan Lazecky, 2024+, University of Leeds
#
# some TS post-processing tools (probably to be later implemented as part of LiCSBAS etc.)
#
################################################################################

from lics_unwrap import *
import re, os, glob
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



def licsbas_tsdir_remove_gacos(tsgacosdir):
    ''' This will remove gacos correction from the time series and store as new TSXGEOC folder'''
    tsgacosdir = os.path.realpath(tsgacosdir)
    tsg = tsgacosdir.split('/')[-1]
    tsdir = tsg.replace('GACOS','')
    tsdir = os.path.join(os.path.dirname(tsgacosdir), tsdir)
    if os.path.exists(tsdir):
        print('ERROR, dir already exists')
        return False
    os.mkdir(tsdir)
    #os.mkdir(tsdir+'/results')
    rc = os.system('cp -r '+os.path.join(tsgacosdir, 'results')+' '+tsdir+'/.')
    rc = os.system('cp -r ' + os.path.join(tsgacosdir, 'info') + ' ' + tsdir + '/.')
    if not rc == 0:
        print('error copying mask')
        return False
    cumgacosfile = os.path.join(tsgacosdir, 'cum.h5')
    cumfile = os.path.join(tsdir, 'cum.h5')
    rc = os.system('LiCSBAS_out2nc.py -i {0}/cum.h5 -o {1}/todel.nc --alignsar'.format(tsgacosdir, os.path.dirname(tsgacosdir)))
    cumgacos = xr.open_dataset(cumgacosfile)
    alignsar = xr.open_dataset(os.path.join(os.path.dirname(tsgacosdir), 'todel.nc'))
    newcum=cumgacos.cum.values.copy()
    for i in range(newcum.shape[0]):
        newcum[i,:,:] = newcum[i,:,:]-np.flipud(alignsar.atmosphere_external.values[i])
    cumgacos.cum.values = newcum
    #velfile = os.path.join(tsgacosdir, 'results', 'vel')
    #vel=np.fromfile(velfile, dtype=np.float32)
    #vel=vel.reshape(cumgacos.vel.shape)
    cumgacos.to_netcdf(cumfile)
    rc = os.system('LiCSBAS16_filt_ts.py -t '+tsdir+' --n_para 4')


def apply_func_in_volclipdir(volclip, predir = '/work/scratch-pw2/licsar/earmla/batchdir/subsets',
                             func = licsbas_tsdir_remove_gacos):
    vdir=os.path.join(predir,str(volclip))
    import glob
    for frdir in glob.glob(vdir+'/???[A,D]'):
        for tsgacosdir in glob.glob(frdir+'/TS*GACOS'):
            tsdir = tsgacosdir[:-5]
            if not os.path.exists(tsdir):
                if os.path.exists(os.path.join(tsgacosdir, 'cum_filt.h5')):
                    try:
                        func(tsgacosdir)
                    except:
                        print('Some error processing in:')
                        print(tsgacosdir)
                else:
                    print('This is not processed yet:')
                    print(tsgacosdir)
            else:
                print('TSDIR already exists, stopping for this directory:')
                print(tsgacosdir)


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
            extif = os.path.join(tifdir, epoch + '.' + ext)
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


def load_metatif(keystr='U', geocdir='GEOC', frame=None):
    '''set either geocdir or frame ID (if to use LiCSAR data) to load one of: U, E, N, hgt, landmask (as keystr)'''
    M = None
    if geocdir:
        M = glob.glob(geocdir + '/*.geo.' + keystr + '.tif')
    if M:
        M = M[0]
    elif frame:
        metadir = os.path.join(os.environ['LiCSAR_public'], str(int(frame[:3])), frame, 'metadata')
        M = os.path.join(metadir, frame + '.geo.' + keystr + '.tif')
        if not os.path.exists(M):
            M = None
    if M:
        M = load_tif2xr(M)
        M = M.where(M != 0)
        return M
    else:
        print('ERROR: no ' + keystr + ' layer exists')
        return False


def generate_pmm_velocity(frame, plate = 'Eurasia', geocdir = None, outif = None):
    '''This will generate LOS velocity for selected tectonic plate, such as for absolute referencing towards Eurasia..
    uses MintPy functionality that implements velocity calculation using Euler poles rather than plate motion model with plates defined as polygons.

    For all codes, see licsbas_mintpy_PMM
    If geocdir is None, it will search directly on LiCSAR system (if you run this on JASMIN..)
    '''
    import licsbas_mintpy_PMM as pmm
    sampling = 20000  # m --- note, this is only primary sampling, we will then interpolate to fit the frame data

    # getting plate data
    plate = pmm.ITRF2014_PMM[plate]
    pole_obj = pmm.EulerPole(
        wx=plate.omega_x,
        wy=plate.omega_y,
        wz=plate.omega_z,
        unit='mas/yr',
    )
    # pole_obj.print_info()

    # getting the frame data
    E = load_metatif('E', geocdir, frame)
    N = load_metatif('N', geocdir, frame)
    U = load_metatif('U', geocdir, frame)

    # coarsening unit vector U as template for the plate velocity
    resolution = get_resolution(U, in_m=True)  # just mean avg in both lon, lat should be ok
    # how large area is covered
    lonextent = len(U.lon) * resolution
    # so what is the multilook factor?
    mlfactorlon = round(len(U.lon) / (lonextent / sampling))
    latextent = len(U.lat) * resolution
    mlfactorlat = round(len(U.lat) / (latextent / sampling))
    Uml = U.coarsen({'lat': mlfactorlat, 'lon': mlfactorlon}, boundary='trim').mean()

    lats = []
    lons = []
    for i in range(len(Uml.lat.values)):
        for j in range(len(Uml.lon.values)):
            lats.append(Uml.lat.values[i])
            lons.append(Uml.lon.values[j])
    lats = np.array(lats)
    lons = np.array(lons)

    # finally getting the plate velocities from the Euler pole definition over the frame area
    ve, vn, vu = pole_obj.get_velocity_enu(lats, lons, alt=0.0, ellps=True)

    # 1. interpolate the plate vel enus (i know, 3x more work here but still .. reasonable due to nans in ENU etc.)
    print('Interpolating the plate velocity ENU vectors to the original frame resolution')
    Uml.values = ve.reshape(Uml.shape)
    ve = Uml.interp_like(U, method='linear', kwargs={"fill_value": "extrapolate"})
    Uml.values = vn.reshape(Uml.shape)
    vn = Uml.interp_like(U, method='linear', kwargs={"fill_value": "extrapolate"})
    Uml.values = vu.reshape(Uml.shape)
    vu = Uml.interp_like(U, method='linear', kwargs={"fill_value": "extrapolate"})

    # 2.
    print('Calculating the plate motion velocity in LOS (please check the sign here)')
    vlos_plate = ve*E + vn*N + vu*U
    vlos_plate = 1000*vlos_plate # to mm/year
    if outif:
        export_xr2tif(vlos_plate, outif, dogdal = False)
    return vlos_plate



'''from lixcor:
import xarray as xr
import netCDF4
def _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size):
    # For time deltas, we must ensure that we use the same encoding as
    # what was previously stored.
    # We likely need to do this as well for variables that had custom
    # econdings too
    if hasattr(nc_variable, 'calendar'):
        data.encoding = {
            'units': nc_variable.units,
            'calendar': nc_variable.calendar,
        }
    data_encoded = xr.conventions.encode_cf_variable(data) # , name=name)
    left_slices = data.dims.index(expanding_dim)
    right_slices = data.ndim - left_slices - 1
    nc_slice   = (slice(None),) * left_slices + (slice(nc_shape, nc_shape + added_size),) + (slice(None),) * (right_slices)
    nc_variable[nc_slice] = data_encoded.data


def append_to_netcdf(filename, ds_to_append, unlimited_dims):
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]
    if len(unlimited_dims) != 1:
        # TODO: change this so it can support multiple expanding dims
        raise ValueError(
            "We only support one unlimited dim for now, "
            f"got {len(unlimited_dims)}.")
    unlimited_dims = list(set(unlimited_dims))
    expanding_dim = unlimited_dims[0]
    with netCDF4.Dataset(filename, mode='a') as nc:
        nc_dims = set(nc.dimensions.keys())
        nc_coord = nc[expanding_dim]
        nc_shape = len(nc_coord)
        added_size = len(ds_to_append[expanding_dim])
        variables, attrs = xr.conventions.encode_dataset_coordinates(ds_to_append)
        for name, data in variables.items():
            if expanding_dim not in data.dims:
                # Nothing to do, data assumed to the identical
                continue
            nc_variable = nc[name]
            _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size)
'''

"""
def import_tifs2cube(tifspath, ncfile, searchstring='/*/*geo.mli.tif', varname = 'amplitude', thirddim = 'time', apply_func = None):
    '''e.g. for amplitude from mlis, use apply_func = np.sqrt
    Note the unlimited_dims should be already set to thirddim !'''
    import glob
    import pandas as pd

    cube = xr.open_dataset(ncfile)
    tifs=glob.glob(tifspath+searchstring)
    for tif in tifs:
        fname = os.path.basename(tif)
        epoch=fname.split('.')[0]
        if '_' in epoch:  # in case of ifgs, we set this to the later date
            epoch = epoch.split('_')[-1]
        epochdt = pd.Timestamp(epoch)
        try:
            data = rioxarray.open_rasterio(tif)
        except:
            print('ERROR loading tif for epoch '+epoch)
            continue
        data = data.rename({'x': 'lon','y': 'lat','band': thirddim})
        data[thirddim] = [epochdt]
        da = xr.DataArray(data=ds['amplitude'].values, dims=cube.dims)



    append_to_netcdf(ncfile, ds, unlimited_dims=thirddim)
"""

'''
def import_tifs(path, varname = 'soil_moisture', cliparea_geo = '', thirddim = 'time', outnc = 'out.nc'):
    firstpass = True
    if cliparea_geo:
        minclipx, maxclipx, minclipy, maxclipy = cliparea_geo.split('/')
        minclipx, maxclipx, minclipy, maxclipy = float(minclipx), float(maxclipx), float(minclipy), float(maxclipy)
    for tif in os.listdir(path):
        if not 'tif' in tif:
            continue
        if thirddim == 'time':
            third = pd.Timestamp(tif.split('.')[0])
        else:
            third = tif.split('.')[0]
        data = xr.open_rasterio(os.path.join(path,tif))
        data = data.rename({'x': 'lon','y': 'lat','band': thirddim})
        data[thirddim] = [third]
        print('importing {}'.format(third))
        data = data.sortby([thirddim,'lon','lat'])
        coordsys = data.crs.split('=')[1]
        if cliparea_geo:
            data = data.sel(lon=slice(minclipx, maxclipx), lat=slice(minclipy, maxclipy))
        ds = xr.Dataset({varname:data})
        if firstpass:
            #ds = xr.Dataset({varname:data})
            #if cliparea_geo:
            #    #in case the coords do not fit
            #    coordsys = data.crs.split('=')[1]
            #    #from pyproj import Transformer
            #    #transformer = Transformer.from_crs("epsg:4326", coordsys)
            #    #(minclipx, minclipy) = transformer.transform(minclipy, minclipx)
            #    #(maxclipx, maxclipy) = transformer.transform(maxclipy, maxclipx)
            #    data = data.sel(lon=slice(minclipx, maxclipx), lat=slice(minclipy, maxclipy))
            #da = data
            ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            ds.rio.write_crs(coordsys, inplace=True)
            ds.to_netcdf(outnc, mode='w', unlimited_dims=[thirddim])
            #if not coordsys == "epsg:4326":
            #    ds = ds.rio.reproject("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
            #    ds = ds.rename({'x': 'lon','y': 'lat'})
            #    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            #    ds.rio.write_crs("epsg:4326", inplace=True)
            firstpass = False
        else:
            append_to_netcdf(outnc, ds, unlimited_dims=thirddim)
            #da = xr.concat([da, data], dim='time')
    #ds = xr.Dataset({varname:da})
    #ds = ds.sortby([thirddim,'lon','lat'])
    #ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    #ds.rio.write_crs(coordsys, inplace=True)
    #if not coordsys == "epsg:4326":
    #    ds = ds.rio.reproject("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    #    ds = ds.rename({'x': 'lon','y': 'lat'})
    #    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    #    ds.rio.write_crs("epsg:4326", inplace=True)
    #ds = ds.transpose(thirddim,"lat","lon")
    #ds = ds.sortby([thirddim,'lon','lat'])
    ds = xr.open_dataset(outnc)
    return ds


def get_date_matrix(pairs):
    date_matrix = pd.DataFrame(pairs, columns=['pair'])
    date_matrix['date1'] = pd.to_datetime(date_matrix.pair.str[:8], format='%Y%m%d')
    date_matrix['date2'] = pd.to_datetime(date_matrix.pair.str[9:], format='%Y%m%d')
    date_matrix['btemp'] = date_matrix.date2 - date_matrix.date1
    date_matrix = date_matrix.set_index('pair')
    return date_matrix
'''