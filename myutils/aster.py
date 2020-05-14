#!/usr/bin/env python
# coding: utf-8
import datetime

import numpy as np
import netCDF4
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from collections import OrderedDict

import typhon as ty
from typhon.cloudmask import aster, cloudstatistics


def asterID(filename):
    '''extract YYYY-MM-DD hh:mm:ss from ASTER file name.
    Returns:
        aster ID (str): YYYYMMDD_HHmmSS
    '''
    x = filename.split('/')[-1].split('_')[2]
    
    return str(x[7:11])+str(x[3:7])+'_'+str(x[11:19])

def dt_from_file(file):
    """extract datetime from ASTER file name."""
    x = file.split('/')[-1].split('_')[2]
    dt = str(x[7:11])+str(x[3:7])+str(x[11:19])
    
    return datetime.datetime.strptime(dt, "%Y%m%d%H%M%S")


def sort_aster_filelist(filelist):
    '''Sort ASTER files from a list ouput of glob according to their timestamp.'''
    idx = []
    for file in filelist:
        x = file.split('/')[-1].split('_')[2]
        idx.append(int(str(x[7:11])+str(x[3:7])+str(x[11:19])))
    
    return [filelist[i] for i in np.argsort(idx)]


def is_eurec4a(ai):
    # check instruments are turned on
    if np.all(myutils.aster.check_instrumentmodes(ai)):
        # check EUREC4A time
        if (datetime.datetime(2020, 1, 1) < ai.datetime
            < datetime.datetime(2020, 3, 1)):
            # check EUREC4A region
            if (6 < ai.scenecenter.latitude < 17 and
                -65 < ai.scenecenter.longitude < -40):
                ret = True
            else:
                ret = False
                print(f"ASTER image location with {ai.scenecenter} lies outside"
                      f" of the EUREC4A domain (latitude:7 to 17 degree N, "
                      f"longitude: -60 to -40 degree E)")
        else:
            ret = False
            print(f"ASTER image date {ai.datetime} does not fall into the "
                  f"EUREC4A time from 2020-01-01 to 2020-02-28.")
    else:
        ret = False
        print(f"ASTER instrument modes: {myutils.aster.check_instrumentmodes(ai)}")
    
    return ret



def dt_unique(astfiles):
    '''return list of days (format:YYYYMMDD) of a given list of ASTER images.'''
    return list(OrderedDict.fromkeys([asterID(file)[:8] for file in astfiles]))


def check_VNIRmodeON(file):
    ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    return meta['ASTEROBSERVATIONMODE.1']=='VNIR1, ON'

def check_VNIRBmodeON(file):
    ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    return meta['ASTEROBSERVATIONMODE.2']=='VNIR2, ON'

def check_SWIRmodeON(file):
    ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    return meta['ASTEROBSERVATIONMODE.3']=='SWIR, ON'
    
def check_TIRmodeON(file):
    ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    return meta['ASTEROBSERVATIONMODE.4']=='TIR, ON'

def check_instrumentmodes(ai):
    #ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    vnir1 = meta['ASTEROBSERVATIONMODE.1']=='VNIR1, ON'
    vnir2 = meta['ASTEROBSERVATIONMODE.2']=='VNIR2, ON'
    swir = meta['ASTEROBSERVATIONMODE.3']=='SWIR, ON'
    tir = meta['ASTEROBSERVATIONMODE.4']=='TIR, ON'
    
    return (vnir1, vnir2, swir, tir)


def check_cloudfraction(file, threshold=.8):
    '''Check the cloud fraction in an ASTER image to be smaller than a given
    threshold value, default: 80 %.'''
    ai = aster.ASTERimage(file)
    clmask = ai.retrieve_cloudmask(output_binary=True, include_channel_r5=False)
    
    return cloudstatistics.cloudfraction(clmask) < threshold


def strrnd(x):
    '''round float to 3 digits and return its string.
    '''
    return f'{float(x):.3f}'


def strcornercoords(coordobj):
    '''arrange latitudes of LL, LR, UR, UL corners, followed by the
    longitudes in same order, separated by an underscore. Return it
    as a string.'''
    out = str()
    
    for i in range(2):
        out = (out + '_' + strrnd(coordobj.LOWERLEFT[i])
        + strrnd(coordobj.LOWERRIGHT[i])
        + strrnd(coordobj.UPPERRIGHT[i])
        + strrnd(coordobj.UPPERLEFT[i]))
        
    return out


def aster_mercator_bounds(file):
    ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    
    out = ''
    for i in [float(meta['WESTBOUNDINGCOORDINATE']), float(meta['EASTBOUNDINGCOORDINATE'])]:
        if i < 0:
            out += strrnd(abs(i)) + 'W-'
        else:
            out += strrnd(i) + 'E-'
    
    for i in [float(meta['SOUTHBOUNDINGCOORDINATE']), float(meta['NORTHBOUNDINGCOORDINATE'])]:
        if i < 0:
            out += strrnd(abs(i)) + 'S-'
        else:
            out += strrnd(i) + 'N-'
    
    return out[:-1]


def aster_raw2mercator(file, path2ofile, nth=1):

    ai = aster.ASTERimage(file)
    refl = ai.get_reflectance('3N')
    lats, lons = ai.get_latlon_grid(channel='3N')
    meta = ai.get_metadata()
    extent = [float(meta['WESTBOUNDINGCOORDINATE']),
              float(meta['EASTBOUNDINGCOORDINATE']),
              float(meta['SOUTHBOUNDINGCOORDINATE']),
              float(meta['NORTHBOUNDINGCOORDINATE'])]
    
    projection = ccrs.Mercator()
    data_crs = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(49.8/nth, 42/nth),
                           subplot_kw=dict(projection=projection))
    ax.set_extent(extent, crs=data_crs)
    features = ty.plots.get_cfeatures_at_scale(scale='50m')
    ax.add_feature(features.BORDERS)
    ax.add_feature(features.COASTLINE)

    s = (slice(0, -1, nth), slice(0, -1, nth))
    ax.pcolormesh(lons[s], lats[s], refl[s], transform=data_crs, cmap='Greys_r')

    ax.axis("off")
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    ax.set_position([0, 0, 1, 1])
    ofile = (path2ofile + 'TERRA_ASTER_' + asterID(file)[:-2] + '_'
                + aster_mercator_bounds(file)
                + '_res'+str(15*nth)+'m.png')
    fig.savefig(ofile, transparent=True, dpi=100)#, bbox_inches='standard')
    plt.close()
    
    return ofile


def aster_raw2png(file, path2ofile):
    '''Create PNG images or reflectance at 0.82 micron at full VNIR resolution
    (15m) and at low resolution of TIR (90m) from ASTER raw data.
    
    Parameters:
        file (str): path and original (hdf) ASTER file name.
        path2ofile (str): path to new netcdf file.
        
    Returns:
        str: message confirming writing output file to given path.
    '''
    ai = aster.ASTERimage(file)
    refl = ai.get_reflectance('3N')
    
    img_data = np.zeros((*np.shape(refl), 2))
    img_data[:, :, 0] = refl.clip(min=0, max=1) * 255
    img_data[:, :, 1] = 255
    img_data[np.isnan(refl), 1] = 0
    im = Image.fromarray(img_data.astype(np.uint8), mode='LA')
    
    # save at full image resolution (15m) -> about 6.7 MB
    im.save(path2ofile + '/TERRA_ASTER_' + asterID(file)
            + strcornercoords(ai.cornercoordinates) + '_highres.png')
    # save at reduced image resolution (90m) -> about 0.256 MB
    im.thumbnail((830, 700)) # works in-place
    im.save(path2ofile + '/TERRA_ASTER_' + asterID(file)
            + strcornercoords(ai.cornercoordinates) + '_lowres.png')
    
    return 'Files saved to ' + path2ofile + 'TERRA_ASTER_' + asterID(file) + '_*.png'


def aster_raw2nc(file, path2ofile):
    '''Create netCDF file with EUREC4A relevant data from ASTER raw data.
    
    Parameters:
        file (str): path and original (hdf) ASTER file name.
        path2ofile (str): path to new netcdf file.
        
    Returns:
        str: message confirming writing output file to given path.
    '''
    ai = aster.ASTERimage(file)
    
    # 2d variables
    latitudesVNIR, longitudesVNIR = ai.get_latlon_grid('3N')
    latitudesTIR, longitudesTIR = ai.get_latlon_grid('14')
    cloudmask = ai.retrieve_cloudmask(include_channel_r5=False)
    reflection082 = ai.get_radiance('3N')
    bt11 = ai.get_radiance('14')
    cthIR = ai.get_cloudtopheight()
    # 1d cloud field statistics
    #cmfiltered = cloudstatistics.filter_cloudmask(cloudmask,
    #                                              threshold=4,
    #                                              connectivity=2)
    cf = cloudstatistics.cloudfraction(cloudmask)
    #iorg = cloudstatistics.iorg(cloudmask, connectivity=2)
    #scai = cloudstatistics.scai(cloudmask, connectivity=2)
    
    # create output netcdf file
    ncfile = netCDF4.Dataset(path2ofile + 'TERRA_ASTER_' + asterID(file) + '.nc',
                             mode='w', format='NETCDF4')
    # global attributes
    ncfile.title='ASTER data processed for EUREC4A quicklooks.'
    ncfile.asterfilename = ai.basename
    ncfile.datetime = str(ai.datetime)
    
    # create dimensions
    latlon = ncfile.createDimension('latlon', 2)
    pixel_alongtrack_VNIR = ncfile.createDimension('pixel_alongtrack_VNIR',
                                                   np.shape(reflection082)[0])
    pixel_acrosstrack_VNIR = ncfile.createDimension('pixel_acrosstrack_VNIR',
                                                   np.shape(reflection082)[1])
    pixel_alongtrack_TIR = ncfile.createDimension('pixel_alongtrack_TIR',
                                                   np.shape(bt11)[0])
    pixel_acrosstrack_TIR = ncfile.createDimension('pixel_acrosstrack_TIR',
                                                   np.shape(bt11)[1])
    # write 2d variables
    nlat = ncfile.createVariable('latitudesVNIR', np.float32, 
                                ('pixel_alongtrack_VNIR',
                                 'pixel_acrosstrack_VNIR'))
    nlat.units = 'degrees_north'
    nlat.long_name = 'Latitude'
    nlat.description = 'latitudes of each pixel for visual VNIR radiometer data'
    nlat[:] = latitudesVNIR

    nlon = ncfile.createVariable('longitudesVNIR', np.float32, 
                                ('pixel_alongtrack_VNIR',
                                 'pixel_acrosstrack_VNIR'))
    nlon.units = 'degrees_east'
    nlon.long_name = 'Longitude'
    nlon.description = 'longitudes of each pixel for visual VNIR radiometer data'
    nlon[:] = longitudesVNIR

    nlatTIR = ncfile.createVariable('latitudesTIR', np.float32, 
                                ('pixel_alongtrack_TIR',
                                 'pixel_acrosstrack_TIR'))
    nlatTIR.units = 'degrees_north'
    nlatTIR.long_name = 'Latitude'
    nlatTIR.description = 'latitudes of each pixel for thermal TIR radiometer data'
    nlatTIR[:] = latitudesTIR

    nlonTIR = ncfile.createVariable('longitudesTIR', np.float32, 
                                ('pixel_alongtrack_TIR',
                                 'pixel_acrosstrack_TIR'))
    nlonTIR.units = 'degrees_east'
    nlonTIR.long_name = 'Longitude'
    nlonTIR.description = 'longitudes of each pixel for thermal TIR radiometer data'
    nlonTIR[:] = longitudesTIR
    
    for i in range(3):
        nrefl = ncfile.createVariable(f"reflection082_{i}", np.float32, 
                                    ('pixel_alongtrack_VNIR',
                                     'pixel_acrosstrack_VNIR'))
        nrefl.units = 'unitless'
        nrefl.long_name = 'Relfection at 0.82 micron'
        nrefl.description = 'reflection derived from channel 3N at 0.82 micron'
        nrefl[:] = reflection082
    
    for i in range(5):
        nbt = ncfile.createVariable(f"brightness_temperature11_{i}", np.float32, 
                                    ('pixel_alongtrack_TIR',
                                     'pixel_acrosstrack_TIR'))
        nbt.units = 'Kelvin'
        nbt.long_name = 'Brightness temperature at 11 micron'
        nbt.description = 'brightness temperature derived from channel 14 at 11 micron'
        nbt[:] = bt11

    nclmask = ncfile.createVariable('cloudmask', np.float32, 
                                ('pixel_alongtrack_VNIR',
                                 'pixel_acrosstrack_VNIR'))
    nclmask.units = 'unitless'
    nclmask.long_name = 'ASTER cloud mask'
    nclmask.description = 'ASTER cloud mask according to Werner et al., 2016'
    nclmask[:] = cloudmask

    ncth = ncfile.createVariable('cloudtopheight', np.float32, 
                                ('pixel_alongtrack_TIR',
                                 'pixel_acrosstrack_TIR'))
    ncth.units = 'km'
    ncth.long_name = 'Cloud top height'
    ncth.description = 'IR cloud top height estimate following Baum et al., 2012'
    ncth[:] = cthIR
    
    # create group for cloud statistics
    stats = ncfile.createGroup('cloudstatistics')
    stats.long_name = 'Cloud field statistics'
    
    ncf = stats.createVariable('cloudfraction', np.float32)
    ncf.units = 'unitless'
    ncf.long_name = 'Cloud fraction'
    ncf[:] = cf

    niorg = stats.createVariable('iorg', np.float32)
    niorg.units = 'unitless'
    niorg.long_name = 'Cluster index Iorg'
    niorg.description = ('cloud cluster index I_org following Tompkins and Semie,'
                       +'2017')
    #niorg[:] = iorg

    nscai = stats.createVariable('scai', np.float32)
    nscai.units = 'unitless'
    nscai.long_name = 'Simple convective aggregation index'
    nscai.description = ('cloud cluster index Simple Convective Aggregation Index'
                       +'(SCAI) following Tobin, Bony, and Roca, 2012')
    #nscai[:] = scai
    
    # create group for general ASTER image info (sensor-sun geometry)
    info = ncfile.createGroup('imageinfo')
    info.long_name = 'General ASTER image information'
    
    sun_zenith = info.createVariable('SunZenith', np.float32)
    sun_zenith.long_name = 'Sun zenith angle'
    sun_zenith.description = 'sun zenith angle of central pixel'
    sun_zenith.unit = 'degree'
    sun_zenith[:] = ai.sunzenith

    sun_azimuth = info.createVariable('SunAzimuth', np.float32)
    sun_azimuth.long_name = 'Sun azimuth angle'
    sun_azimuth.description = 'sun azimuth angle of central pixel'
    sun_azimuth.unit = 'degree'
    sun_azimuth[:] = ai.solardirection.azimuth

    scenecenter = info.createVariable('SceneCenter', np.float32, ('latlon'))
    scenecenter.long_name = 'Cetral pixel (latitude, longitude)'
    scenecenter.description = ''
    scenecenter.unit = 'degree'
    scenecenter[:] = ai.scenecenter
    
    ncfile.close()
    
    return 'File saved to ' + path2ofile + 'TERRA_ASTER_' + asterID(file) + '.nc'


def construct_binedges(reflectances):
    '''Get linearly spaced bins corresponding to discrete ASTER reflectance values.'''
    dr = np.unique(reflectances)[1] - np.unique(reflectances)[0]
    
    return np.arange(dr/2, np.nanmax(reflectances), dr)


def edges2mids(binedges):
    return binedges[:-1] + (binedges[1] - binedges[0])/2


def get_hist_aster(hdffile):
    ai = aster.ASTERimage(hdffile)
    refl = ai.get_reflectance(channel='3N')
    refl_scanline = refl[np.shape(refl)[0] // 2]
    refl_scanline = refl_scanline[~np.isnan(refl_scanline)]
    
    binedges = construct_binedges(refl)
    n, _ = np.histogram(refl[~np.isnan(refl)], bins=binedges)
    n = n / len(refl[~np.isnan(refl)])
    n_scanline, _ = np.histogram(refl_scanline[~np.isnan(refl_scanline)],
                                 bins=binedges)
    n_scanline = n_scanline / len(refl_scanline[~np.isnan(refl_scanline)])
    
    return n, n_scanline, binedges


def get_refl_aster(hdffile):
    ai = aster.ASTERimage(hdffile)
    refl = ai.get_reflectance(channel='3N')
    refl_scanline = refl[np.shape(refl)[0] // 2]
    refl_scanline = refl_scanline[~np.isnan(refl_scanline)]
    
    return refl, refl_scanline


def get_hist_libradtran(ncfile, binedges):
    with netCDF4.Dataset(ncfile) as ds:
        refl_lib = ds.variables['reflectance'][:].astype(float)
    n_lib, _ = np.histogram(refl_lib, bins=binedges)
    
    return n_lib / len(refl_lib)


def get_refl_libradtran(ncfile):
    with netCDF4.Dataset(ncfile) as ds:
        refl_lib = ds.variables['reflectance'][:].astype(float)
        
    return refl_lib