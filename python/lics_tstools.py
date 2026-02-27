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
from scipy.constants import speed_of_light
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

def loadall2cube(cumfile, column='cum', extracols=['loop_ph_avg_abs']):
    cumdir = os.path.dirname(cumfile)
    cohfile = os.path.join(cumdir,'results/coh_avg')
    rmsfile = os.path.join(cumdir,'results/resid_rms')
    vstdfile = os.path.join(cumdir,'results/vstd')
    stcfile = os.path.join(cumdir,'results/stc')
    maskfile = os.path.join(cumdir,'results/mask')
    metafile = os.path.join(cumdir,'../../metadata.txt')
    #h5datafile = 'cum.h5'
    cum = xr.load_dataset(cumfile)

    # --- NEW: validate that the requested column exists ---
    if column not in cum.data_vars:
        raise ValueError(
            f"Requested column '{column}' not found in dataset. "
            f"Available data variables: {list(cum.data_vars)}"
        )

    sizex = len(cum.vel[0])
    sizey = len(cum.vel)
    
    lon = cum.corner_lon.values+cum.post_lon.values*np.arange(sizex)-0.5*float(cum.post_lon)
    lat = cum.corner_lat.values+cum.post_lat.values*np.arange(sizey)+0.5*float(cum.post_lat)  # maybe needed? yes! for gridline/AREA that is default in rasterio...
    
    time = np.array(([dt.datetime.strptime(str(imd), '%Y%m%d') for imd in cum.imdates.values]))
    
    velxr = xr.DataArray(cum.vel.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
    #LiCSBAS uses 0 instead of nans...
    velxr = velxr.where(velxr!=0)
    velxr.attrs['unit'] = 'mm/year'

    # --- CHANGED: use the selected variable name ---
    cumxr = xr.DataArray(cum[column].values, coords=[time, lat, lon], dims=["time", "lat", "lon"])
    cumxr.attrs['unit'] = 'mm'
    #bperpxr = xr.DataArray(cum.bperp.values, coords=[time], dims=["time"])
    
    cube = xr.Dataset()
    cube[column] = cumxr     # keep the chosen name in the output
    cube['vel'] = velxr
    #cube['vintercept'] = vinterceptxr
    try:
        cube['bperp'] = xr.DataArray(cum.bperp.values, coords=[time], dims=["time"])
        cube['bperp'] = cube.bperp.where(cube.bperp!=0)
        # re-ref it to the first date
        if np.isnan(cube['bperp'][0]):
            firstbperp = 0
        else:
            firstbperp = cube['bperp'][0]
        cube['bperp'] = cube['bperp'] - firstbperp
        cube['bperp'] = cube.bperp.astype(np.float32)
        cube.bperp.attrs['unit'] = 'm'
    except:
        print('some error loading bperp info')
    
    #if 'mask' in cum:
    #    # means this is filtered version, i.e. cum_filt.h5
    cube.attrs['filtered_version'] = 'mask' in cum
    
    #add coh_avg resid_rms vstd
    if os.path.exists(cohfile):
        infile = np.fromfile(cohfile, 'float32')
        cohxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        cube['coh'] = cohxr
        cube.coh.attrs['unit']='unitless'
    else: print('No coh_avg file detected, skipping')
    if os.path.exists(rmsfile):
        infile = np.fromfile(rmsfile, 'float32')
        rmsxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        rmsxr.attrs['unit'] = 'mm'
        cube['rms'] = rmsxr
    else: print('No RMS file detected, skipping')
    try:
        for e in extracols:
            efile=os.path.join(cumdir,'results',e)
            if os.path.exists(efile):
                infile = np.fromfile(efile, 'float32')   # should be always float. but we can check with os.stat('loop_ph_avg_abs').st_size
                exr = xr.DataArray(infile.reshape(sizey, sizex), coords=[lat, lon], dims=["lat", "lon"])
                #rmsxr.attrs['unit'] = 'mm'
                cube[e] = exr
            else:
                print('No '+e+' file detected, skipping')
    except:
        print('debug - extra layers not included')
    if os.path.exists(vstdfile):
        infile = np.fromfile(vstdfile, 'float32')
        vstdxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        vstdxr.attrs['unit'] = 'mm/year'
        cube['vstd'] = vstdxr
    else: print('No vstd file detected, skipping')
    if os.path.exists(stcfile):
        infile = np.fromfile(stcfile, 'float32')
        stcxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        stcxr.attrs['unit'] = 'mm'
        cube['stc'] = stcxr
    else: print('No stc file detected, skipping')
    if os.path.exists(maskfile):
        infile = np.fromfile(maskfile, 'float32')
        #infile = np.nan_to_num(infile,0).astype(int)  # change nans to 0
        infile = np.nan_to_num(infile,0).astype(np.int8)  # change nans to 0
        maskxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        maskxr.attrs['unit'] = 'unitless'
        cube['mask'] = maskxr
    else: print('No mask file detected, skipping')
    # add inc_angle
    if os.path.exists(metafile):
        #a = subp.run(['grep','inc_angle', metafile], stdout=subp.PIPE)
        #inc_angle = float(a.stdout.decode('utf-8').split('=')[1])
        inc_angle = float(grep1line('inc_angle',metafile).split('=')[1])
        cube.attrs['inc_angle'] = inc_angle
    else: print('')#'warning, metadata file not found. using general inc angle value')
        #inc_angle = 39
    
    #cube['bperp'] = bperpxr
    #cube[]
    cube.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    cube.rio.write_crs("EPSG:4326", inplace=True)
    #cube = cube.sortby(['time','lon','lat']) # 2025/03: not really right as lat should be opposite-signed..
    return cube



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
    rc = os.system('cp -r ' + os.path.join(tsgacosdir, 'network') + ' ' + tsdir + '/.')
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
    os.system('rm {0}/todel.nc'.format(os.path.dirname(tsgacosdir)))
    # recalc vel etc
    rc = os.system('LiCSBAS_cum2vel.py -i '+tsdir+'/cum.h5 -o '+tsdir+'/tomove')
    rc = os.system('mv '+tsdir+'/tomove.vel '+tsdir+'/results/vel')
    rc = os.system('mv ' + tsdir + '/tomove.vconst ' + tsdir + '/results/vconst')
    cum = xr.open_dataset(cumfile)
    cum=cum.load()
    b = np.fromfile(tsdir + '/results/vel', dtype=np.float32)
    cum.vel.values=b.reshape(cum.vel.shape)
    b = np.fromfile(tsdir + '/results/vconst', dtype=np.float32)
    cum.vintercept.values = b.reshape(cum.vintercept.shape)
    cum.close()
    cum.to_netcdf(cumfile)
    rc = os.system('LiCSBAS14_vel_std.py -t '+tsdir+' --mem_size 8192')
    rc = os.system('LiCSBAS15_mask_ts.py -t '+tsdir+' -c 0.1 -u 0.5 -s 15 -i 700 -L 0.35 -T 0.5 -v 10 -g 10 --avg_phase_bias 1 -r 10 --n_gap_use_merged')
    rc = os.system('LiCSBAS16_filt_ts.py -t '+tsdir+' --nopngs --interpolate_nans --n_para 2')


''' e.g.
import pandas as pd
vidsfile='chilevolcsokvids.txt'
vids=pd.read_csv(vidsfile, header=None)

for vid in vids[0].values:
    print(vid)
    apply_func_in_volclipdir(vid)


'''
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


def load_licsbas_cumh5_as_xrda(cumfile, dvars = ['cum','vel'] ):
    ''' Loads cum.h5 (now only cum layer) as standard xr.DataArray (in lon/lat)'''
    # print(cumfile)
    cum = xr.load_dataset(cumfile) #, engine='h5netcdf')  # or 'netcdf4'
    #
    sizex = len(cum.vel[0])
    sizey = len(cum.vel)
    #
    lon = cum.corner_lon.values + cum.post_lon.values * np.arange(sizex) - 0.5 * float(cum.post_lon)
    lat = cum.corner_lat.values + cum.post_lat.values * np.arange(sizey) + 0.5 * float(cum.post_lat)  # maybe needed? yes! for gridline/AREA that is default in rasterio...
    #
    time = np.array(([dt.datetime.strptime(str(imd), '%Y%m%d') for imd in cum.imdates.values]))
    #
    out = xr.Dataset()
    for dvar in dvars:
        if len(cum[dvar].shape) == 2:
            varxr = xr.DataArray(cum[dvar].values.reshape(sizey, sizex), coords=[lat, lon], dims=["lat", "lon"])
            # LiCSBAS uses 0 instead of nans...
            varxr = varxr.where(varxr != 0)
            if dvar == 'vel':
                varxr.attrs['unit'] = 'mm/year'
            # vinterceptxr = xr.DataArray(cum.vintercept.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
            out[dvar]=varxr
        elif len(cum[dvar].shape) == 3:
            #
            cumxr = xr.DataArray(cum[dvar].values, coords=[time, lat, lon], dims=["time", "lat", "lon"])
            if dvar == 'cum':
                cumxr.attrs['unit'] = 'mm'
            out[dvar]=cumxr
        else:
            print('trying to add layer '+dvar)
            out[dvar]=cum[dvar]
    refarea = str(cum.refarea.values)
    # x is first...
    refx1 = int(refarea.split('/')[0].split(':')[0])
    refx2 = int(refarea.split('/')[0].split(':')[1])
    refy1 = int(refarea.split('/')[1].split(':')[0])
    refy2 = int(refarea.split('/')[1].split(':')[1])
    refx=int((refx2+refx1)/2)
    refy = int((refy2 + refy1) / 2)
    out.attrs['ref_lon'] = out.lon.values[refx]
    out.attrs['ref_lat'] = out.lat.values[refy]
    return out


def correct_cum_from_tifs(cumhdfile, tifdir = 'GEOC.EPOCHS', ext='geo.iono.code.tif', tif_scale2mm = 1, sbovl=False ,outputhdf = None, directcorrect = True, newcumname = 'external_data'):
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

    ds = load_licsbas_cumh5_as_xrda(cumhdfile)
    cumxr = ds.cum.copy()
    cumxr.attrs = {**ds.attrs, **cumxr.attrs}

    print('loading external corrections')
    if 'STEC' in ext.upper():
        cumxr = cumcube_sbovl_remove_from_tifs(cumxr, tifdir, ext, only_load_ext = not directcorrect)
    else:
        cumxr = cumcube_remove_from_tifs(cumxr, tifdir=tifdir, ext=ext, tif_scale2mm=tif_scale2mm, only_load_ext=not directcorrect)
    if type(cumxr) == type(False):
        print('ERROR - probably the correction did not exist for some epochs. Cancelling')
        return False
    cumh = xr.load_dataset(cumhdfile)
    if directcorrect:
        cumh.cum.values = cumxr.values
    else:
        codes = ['iono', 'tide', 'icams', 'sltd']
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


# looks a bit complex so I will create for sbovl specifically
def cumcube_sbovl_remove_from_tifs(cumxr, tifdir = 'GEOC.EPOCHS', ext='geo.iono.code.sTECA.tif',  only_load_ext = False):
    ''' Correct directly from tifs, no need to store in cubes.
    NOTE - you can also just load the exts into the cumcube without removing anything..
    (in any case, values are referred temporally to the first epoch)
    
    Args:
        cumxr (xr.DataArray): input cum.h5 file (only cum)
        tifdir:
        ext1: sTECA
        ext2: sTECB
        tif_scale2mm:  for iono [rad]: 14000 for sTECA/B
        only_load_ext:  would only load the ext files in the cube and return it (no removal!)
        
    Returns:
        xr.DataArray: corrected cum values (only_load_ext=False) or only loaded corrections
    '''
    
    try:
        reflon, reflat = cumxr.attrs['ref_lon'], cumxr.attrs['ref_lat'] #MN to ML? we skip reflat and reflon actually both LoS and SBOI just doing mean, why is the reason?
    except:
        print('warning, no ref area information')
        reflon, reflat = None, None
    
    print('sbovl-iono correction is being applied')
    firstepvals = 0
    leneps = len(cumxr)
    error_log = []
    for i in range(leneps): # times first coord..
        print('  Running for {0:6}/{1:6}th epoch'.format(i+1, leneps), flush=True, end='\r')
        cumepoch = cumxr[i]
        epoch = str(cumepoch.time.values).split('T')[0].replace('-','')
        if 'STEC' in ext.upper():
            ext1='geo.iono.code.sTECA.tif'
            ext2='geo.iono.code.sTECB.tif'
        extif1 = os.path.join(tifdir, epoch, epoch+'.'+ext1)
        extif2 = os.path.join(tifdir, epoch, epoch+'.'+ext2)
        if not os.path.exists(extif1) or not os.path.exists(extif2):
            extif1 = os.path.join(tifdir, epoch+'.'+ext1)
            extif2 = os.path.join(tifdir, epoch+'.'+ext2)
        
        try:
            if not os.path.exists(extif1) or not os.path.exists(extif2):
                raise FileNotFoundError(f'File not found: {extif1} or {extif2}')
            #backward
            extepoch1 = load_tif2xr(extif1)
            extepoch1 = extepoch1.where(extepoch1 != 0) # just in case...
            extepoch1 = extepoch1.interp_like(cumepoch, method='linear') # CHECK! ##looks redundant so far (maybe not)
            #forward
            extepoch2 = load_tif2xr(extif2)
            extepoch2 = extepoch2.where(extepoch2 != 0)
            extepoch2 = extepoch2.interp_like(cumepoch, method='linear')
            
            ####gradient method Lazecky et al. 2023,GRL #https://github.com/comet-licsar/daz/blob/main/lib/daz_iono.py#L561
            ###parameter for TEC gradient
            azpix=14000
            PRF = 486.486
            k = 40.308193 # m^3 / s^2
            f0 = 5.4050005e9
            c = speed_of_light
        
            ##scaling_tif
            workdir=os.getcwd()
            frame=os.path.basename(workdir)
            metafolder = os.path.join(os.environ['LiCSAR_public'], str(int(frame[:3])), frame, 'metadata')
            # Check if the metadata folder exists
            if os.path.exists(metafolder) and os.path.isdir(metafolder):
                scaling_tif = None  # Initialize variable to track if a file is found
                
                for files in os.listdir(metafolder):  
                    if files.endswith('bovl_scaling.tif'):
                        scaling_tif = os.path.join(metafolder, files)
                     
                # Check if no scaling file was found
                if scaling_tif is None:
                    raise FileNotFoundError("No .geo.sbovl_scaling.tif file found in metadata folder.")
            else:
                raise FileNotFoundError("metadata is not exist in LiCSAR_public")    

            ##scaling2dfdc
            scaling_factor=load_tif2xr(scaling_tif)
            scaling_factor = scaling_factor.interp_like(cumepoch, method='linear')
            dfDC=azpix*PRF/(2*np.pi*scaling_factor)
            # dfDC=4350 #TODO, this is a constant value for now, but should be calculated from scaling_factor #MN
            fH = f0 + dfDC*0.5
            fL = f0 - dfDC*0.5
            tecovl=(extepoch1/fH-extepoch2/fL)
            iono_grad = 2*PRF*k/c/dfDC * tecovl #unitless
            iono_grad_mm=iono_grad*azpix #mm
            
        except Exception as e:
            print(f'\n\r WARNING: failed to load or compute correction for epoch {epoch}: {str(e)}')
            error_log.append(epoch)
            iono_grad_mm = cumepoch.copy() * np.nan
            iono_grad_mm.attrs.clear()
        
        if reflon:
            iono_grad_mm = iono_grad_mm - iono_grad_mm.sel(lon=reflon, lat=reflat, method='nearest')
        else:
            iono_grad_mm = iono_grad_mm - iono_grad_mm.mean(skipna=True)
            
        if i == 0:
            firstepvals = iono_grad_mm.fillna(0).values
        # here we do diff w.r.t. first epoch
        iono_grad_mm.values = iono_grad_mm.values - firstepvals
        
        if only_load_ext:
            cumxr.values[i] = iono_grad_mm.values
        else:
            cumxr.values[i] = cumxr.values[i] - iono_grad_mm.values
    print('\n\r  done')
    
    # Save the list of failed epochs
    if error_log:
        with open(f"failed_{ext}.txt", "w") as f:
            for epoch in error_log:
                f.write(epoch + "\n")
        print(f"\nSaved list of failed epochs to failed_{ext}.txt")
    return cumxr

# def check_complete_set(imdates, epochsdir, ext='geo.iono.code.tif')
def cumcube_remove_from_tifs(cumxr, tifdir = 'GEOC.EPOCHS', ext='geo.iono.code.tif',tif_scale2mm = 1, only_load_ext = False):
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
    try:
        reflon, reflat = cumxr.attrs['ref_lon'], cumxr.attrs['ref_lat']
    except:
        print('warning, no ref area information')
        reflon, reflat = None, None
    #
    firstepvals = 0
    leneps = len(cumxr)
    error_log = [] ##to save/remove/recreate for second iteration
    
    for i in range(leneps): # times first coord..
        print('  Running for {0:6}/{1:6}th epoch'.format(i+1, leneps), flush=True, end='\r')
        cumepoch = cumxr[i]
        epoch = str(cumepoch.time.values).split('T')[0].replace('-','')
        extif = os.path.join(tifdir, epoch, epoch+'.'+ext)
        if not os.path.exists(extif):
            extif = os.path.join(tifdir, epoch + '.' + ext)
        
        try:
            if not os.path.exists(extif):
                raise FileNotFoundError(f'File not found: {extif}')
            extepoch = load_tif2xr(extif)
            extepoch = extepoch.where(extepoch != 0) # just in case...
            extepoch = extepoch * tif_scale2mm
            extepoch = extepoch.interp_like(cumepoch, method='linear') # CHECK!

            # if not sbovl_abs:
                # reflon, reflat = cumxr.attrs['ref_lon'], cumxr.attrs['ref_lat']
            if reflon:
                extepoch = extepoch - extepoch.sel(lon=reflon, lat=reflat, method='nearest') # could be done better though
            else:
                extepoch = extepoch - extepoch.mean(skipna=True)
        except Exception as e:
            print(f'\n\r WARNING: failed to load correction for epoch {epoch}: {str(e)}')
            error_log.append(extif)
            extepoch = cumepoch.copy() * np.nan
            extepoch.attrs.clear()    
        
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
    
    # Save the list of failed paths
    if error_log:
        with open(f"failed_{ext}.txt", "w") as f:
            for path in error_log:
                f.write(path + "\n")
        print(f"\nSaved list of corrupted or missing files to corrupted_ext_tifs.txt")

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
        #M = M.where(M != 0)
        return M
    else:
        print('ERROR: no ' + keystr + ' layer exists')
        return False


def generate_pmm_velocity(frame, plate = 'Eurasia', geocdir = None, outif = None, azi = False):
    '''This will generate LOS velocity for selected tectonic plate, such as for absolute referencing towards Eurasia..
    uses MintPy functionality that implements velocity calculation using Euler poles rather than plate motion model with plates defined as polygons.

    For all codes, see licsbas_mintpy_PMM
    If geocdir is None, it will search directly on LiCSAR system (if you run this on JASMIN..)
    If azi, it will use E[NU].azi tif files..
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
    if azi:
        azistr='.azi'
    else:
        azistr=''
    # getting the frame data
    E = load_metatif('E'+azistr, geocdir, frame)
    N = load_metatif('N'+azistr, geocdir, frame)
    U = load_metatif('U'+azistr, geocdir, frame)

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
    vlos_plate = ve*E + vn*N + vu*U #double check if vu is nan or zero
    vlos_plate = vlos_plate.where(vlos_plate!=0)
    vlos_plate = 1000*vlos_plate # to mm/year
    if outif:
        export_xr2tif(vlos_plate, outif, dogdal = False)
    return vlos_plate

# STAMPS-related operations:
def csv2nc(csv, outncfile = None, resol = 0.0025, extracols = [], compressnc=True):
    """ Converts a csv file into netcdf.
    The CSV must have following columns (order does not matter, but keep the upper case):
    LAT,LON,VEL,COHER,*dates*
    where *dates* are column names in format of yyyy-mm-dd.

    Usage example:
        nc = csv2nc('test.csv', 'test.nc')
        nc.vel.plot(); plt.show()

    Args:
        csv (string): path to the csv
        outncfile (string): path to the output netcdf file (or None to only return the object)
        resol (float): output resolution cell size in degrees (WGS-84)
        extracols (list): list of column names to be also exported (apart from VEL,COHER and dates)
    Returns:
        xr.DataArray
    """
    df = pd.read_csv(csv)
    print('converting to the netcdf file')
    nc = df2nc(df, outncfile = outncfile, resol = resol, extracols = extracols, compressnc=compressnc)
    return nc


def deg2m(val, m2deg=False):
    if m2deg:
        return val/111111
    else:
        return val*111111


def df2nc(df, outncfile=None, resol=0.0025, extracols=[], compressnc=True):
    """ Converts pandas dataframe (loaded csv file) to NetCDF.
    See help of csv2nc for proper formatting.
    Grid would aggregate values in each cell by their median.
    """
    to_bin = lambda x: np.floor(x / resol) * resol + resol/2 # +resol/2 means shifting towards cell centre (hope correct?)
    df["lat"] = to_bin(df['LAT'])
    df["lon"] = to_bin(df['LON'])
    groups = df.groupby(["lat", "lon"])
    medgrid = groups.median()  # .agg(np.nanmedian)
    #
    #lat = medgrid.index.get_level_values(level=0)
    #lon = medgrid.index.get_level_values(level=1)
    dates = df.columns[df.columns.str.match(r"\d{4}-\d{2}-\d{2}")].to_list()
    #
    # cols = ['VEL_U', 'COHER'] #, 'SIGMA VEL_U']
    cols = ['VEL', 'COHER']
    cols = cols + extracols
    nc = medgrid[cols].to_xarray()
    # velcol = 'vel'
    nc = nc.rename({'VEL': 'vel', 'COHER': 'coh'})
    #
    # now convert from dates
    datum = dates[0]
    a = medgrid[datum].to_xarray().assign_coords(
        {'time': dt.datetime.strptime(datum, '%Y-%m-%d')}).expand_dims('time').rename('cum')
    for datum in dates[1:]:
        b = medgrid[datum].to_xarray().assign_coords(
            {'time': dt.datetime.strptime(datum, '%Y-%m-%d')}).expand_dims('time').rename('cum')
        a = xr.concat([a, b], dim='time')
    #
    nc = nc.assign_coords({'time': a.time.values})
    nc['cum'] = a
    # but values can be missing! so...:
    x_min, x_max = float(nc.lon.min()), float(nc.lon.max())
    y_min, y_max = float(nc.lat.min()), float(nc.lat.max())
    #
    x_reg = np.arange(x_min, x_max + resol, resol)
    y_reg = np.arange(y_min, y_max + resol, resol)
    if nc.lat[1]<nc.lat[0]:
        y_reg = y_reg[::-1]
    nc = nc.reindex(lon=x_reg, lat=y_reg, method='nearest') # method=None did not work well (but why??!!)
    # now add CRS:
    nc.attrs['crs'] = 'EPSG:4326'  # Optional global attribute
    #
    # Add a variable to define the CRS following CF conventions
    nc['crs'] = xr.DataArray(0, attrs={
        'grid_mapping_name': 'latitude_longitude',
        'epsg_code': '4326',
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
        'longitude_of_prime_meridian': 0.0,
        'units': 'degrees'
    })
    #
    # Link the CRS to your data variable(s)
    for varn in ['vel', 'coh', 'cum']+extracols:
        nc[varn].attrs['grid_mapping'] = 'crs'
    #
    if compressnc:
        # compress it and store as netcdf
        encode = {'vel': {'zlib': True, 'complevel': 9},
                  'coh': {'zlib': True, 'complevel': 9},
                  'cum': {'zlib': True, 'complevel': 9},
                  'time': {'dtype': 'i4'}
                  }
        for extracol in extracols:
            encode.update({extracol: {'zlib': True, 'complevel': 9}})
        if outncfile:
            nc.to_netcdf(outncfile, encoding=encode)
    else:
        if outncfile:
            nc.to_netcdf(outncfile)
    return nc


''' # the exaggeration did not work, so postponing:
def exaggerate_amplitude(rslcfile):
    # check if scomplex or what
    import LiCSAR_iofunc as LICSARio
    dtype1 = LICSARio.dtype_slc_gamma_par(rslcfile)
    if (dtype1 == 'SCOMPLEX'):
        cpxSLC = LICSARio.read_fast_slc_gamma_scomplex(rslcfile)
        # no need to byteswap!!???
    else:
        cpxSLC = LICSARio.read_fast_slc_gamma_fcomplex(rslcfile)
        cpxSLC = cpxSLC.byteswap()
    mag = np.abs(cpxSLC)**2 # exaggerate
    pha = np.angle(cpxSLC)
    R = np.cos(pha) * mag
    I = np.sin(pha) * mag
    mag, pha = None, None
    out = R + 1j * I
    if (dtype1 == 'SCOMPLEX'):
        # out
        LICSARio.write_fast_slc_gamma_scomplex(out, rslcfile)
    else:
        LICSARio.write_fast_slc_gamma_fcomplex(out, rslcfile)


def exaggerate_amplitude_rslcdir(rslcdir = 'RSLC'):
    if not os.path.exists(rslcdir):
        print('The directory '+rslcdir+' does not exist')
        return False
    for epoch in os.listdir(rslcdir):
        epochrslc = os.path.join(rslcdir, epoch, epoch+'.rslc')
        epochrslcbackup = os.path.join(rslcdir, epoch, epoch + '.rslc.orig')
        if os.path.exists(epochrslc):
            print(epoch)
            if not os.path.exists(epochrslcbackup):
                cmd = 'cp '+epochrslc+' '+epochrslcbackup
                os.system(cmd)
                exaggerate_amplitude(epochrslc)
            else:
                print('ERROR: the backup RSLC exists - probably already done, continuing')
'''

'''
# STEP 2 - convert STAMPS PS outputs to nc:
klstep = 0
if klstep == 2:
    for csv in [
    '022D.csv',
    '044A.csv',
    '095D.csv',
    '146A.csv'
    ]:
        outncfile=csv.replace('.csv','.nc')
        nc = xr.open_dataset(outncfile)
        periodname = 'Q12'
        t1=dt.datetime(2025,1,1)
        t2=dt.datetime(2025,5,31)
        nc = create_period(nc, periodname, t1, t2)
        #
        periodname = 'Q34'
        t1=dt.datetime(2025,6,1)
        t2=dt.datetime(2025,8,31)
        nc = create_period(nc, periodname, t1, t2)
        #
        increment = nc['displacement.Q34']-nc['displacement.Q12']
        increment_std = nc['displacement_std.Q34']+nc['displacement_std.Q12']
        # adding estimated std from temporal coherence:
        nc['sigma_mmyr']=nc['vel'].copy()
        nc['sigma_mmyr'].values=np.sqrt(1/(2*nc.coh.values**2))
        #increment_std = increment_std+nc['sigma_mmyr'] # sigma_mmyr is per year, and our periods are (or..should be...) two half-years
        nc['inc.Q34'] = increment
        nc['inc.Q34.std'] = increment_std
        outncfile=csv.replace('.csv','.v2.nc')
        nc.to_netcdf(outncfile)

## STEP 3: decompose (need LA!)
# for x in *v2.nc; do a=`echo $x | cut -d '.' -f1`; frnc=`ls $a/*.nc | cut -d '/' -f2`; mv $x $frnc; done
framesnc=['022D_03989_131313.nc','044A_03932_131313.nc','095D_03993_131313.nc','146A_04048_131313.nc']

decompose_np_multi(input_data, beta = 0, do_velUN=False)

def create_period(nc, periodname = 'Q12', t1=dt.datetime(2025,1,1), t2=dt.datetime(2025,5,31)):
    # coordname = 'period'
    colname_cumval = 'displacement'+'.'+periodname
    colname_cumstd = 'displacement_std'+'.'+periodname
    cumlyr = nc.cum.sel(time=slice(t1,t2))  # slice includes the end points
    cumval = cumlyr.mean(axis=0)
    cumstd = cumlyr.std(axis=0)
    nc[colname_cumval] = cumval
    nc[colname_cumstd] = cumstd
    return nc

'''

# for colname in [colname_cumval, colname_cumstd]:
#     if not colname in nc.data_vars:
#         nc[colname] = xr.DataArray(dims=['lat', 'lon','period'],coords={'lat': nc.lat, 'lon': nc.lon,'period': []})
#
# cumval = cumval.expand_dims(dim=coordname, axis=-1).assign_coords({coordname: [periodname]})
# cumval = xr.concat([nc[colname_cumval], cumval], dim=coordname)
# cumstd = cumstd.expand_dims(dim=coordname, axis=-1).assign_coords({coordname: [periodname]})
# cumstd = xr.concat([nc[colname_cumstd], cumstd], dim=coordname)
# # ugly trick - xarray will not update the coords! so i first delete it...
# nc = nc.drop_dims(coordname)
# nc[colname_cumval] = cumval
# nc[colname_cumstd] = cumstd

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