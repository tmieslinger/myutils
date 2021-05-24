#!/usr/bin/env python
# coding: utf-8
import datetime
import glob
import time

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

def convert_npdt2asterfilename(dt):
    prefix = "/scratch/uni/u237/data/aster/aster_L1B/eurec4a/AST_L1B_003"
    date = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(dt.astype('O')/1e9),
                                      "%m%d%Y%H%M%S")
    return glob.glob(prefix+date+"*.hdf")[0]

def dt_unique(astfiles):
    '''return list of days (format:YYYYMMDD) of a given list of ASTER images.'''
    return list(OrderedDict.fromkeys([asterID(file)[:8] for file in astfiles]))

def sort_aster_filelist(filelist):
    '''Sort ASTER files from a list ouput of glob according to their timestamp.'''
    idx = []
    for file in filelist:
        x = file.split('/')[-1].split('_')[2]
        idx.append(int(str(x[7:11])+str(x[3:7])+str(x[11:19])))
    
    return [filelist[i] for i in np.argsort(idx)]


def is_eurec4a(ai):
    # check instruments are turned on
    if np.all(check_instrumentmodes(ai)):
        # check EUREC4A time
        if (datetime.datetime(2020, 1, 1) < ai.datetime
            < datetime.datetime(2020, 3, 1)):
            # check EUREC4A region
            if (7 < ai.scenecenter.latitude < 22 and
                -61 < ai.scenecenter.longitude < -41):
                ret = True
            else:
                ret = False
                print(f"ASTER image location {ai.scenecenter} outside.")
        else:
            ret = False
            print(f"ASTER image date {ai.datetime} does not fall into the "
                  f"EUREC4A time from 2020-01-01 to 2020-02-28.")
    else:
        ret = False
        print(f"ASTER instrument modes: {myutils.aster.check_instrumentmodes(ai)}")
    
    return ret

def eurec4a_overpasses(files):
    dt = [dt_from_file(f) for f in files]
    dt_str = [asterID(f)[:8] for f in files]
    dt_num = np.array([int(i) for i in dt_str])
    # create an index array: idx_day
    dt_uniq_str = dt_unique(files)
    dt_uniq = [datetime.datetime.strptime(i, "%Y%m%d") for i in dt_uniq_str]
    # unique days, but at 14 UTC so that it correlates with ASTER overpass in the plot
    dt_uniq = [i + datetime.timedelta(hours=14) for i in dt_uniq]
    idx_day = np.full_like(dt, np.nan)
    for idx, day in enumerate(dt_uniq_str):
        indices = np.where(dt_num==int(day))[0]
        idx_day[indices] = idx
    return dt, idx_day, dt_uniq

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


def aster_raw2nc(file, path2ofile="output/nc"):
    '''Convert ASTER L1B HDF4 files to netCDF.
    ASTER digital counts from bands in the VNIR and TIR range are converted
    to radiances. Additionally, latitude and longitude fields are derived and
    level 2 data such as a cloud mask and cloud top height estimates.
    
    Parameters:
        file (str): path and original (hdf) ASTER file name.
        path2ofile (str): path to new netcdf file.
        
    Returns:
        str: message confirming file conversion.
    '''
    ai = aster.ASTERimage(file)

    # 2d variables
    latitudesVNIR, longitudesVNIR = ai.get_latlon_grid('1')
    latitudesTIR, longitudesTIR = ai.get_latlon_grid('10')
    # additional fields
    cloudmask = ai.retrieve_cloudmask(output_binary=False,
                                      include_channel_r5=False)
    cthIR = ai.get_cloudtopheight()
    # radiance fields
    fields = dict()
    for ch in ai.subsensors['VNIR'] + ai.subsensors['TIR']:
        if ch!='3B':
            fields[ch] = ai.get_radiance(ch)
    # create output netcdf file
    ncfile = netCDF4.Dataset(
        f"{path2ofile}/TERRA_ASTER_processed_{asterID(file)}.nc",
        mode='w', format='NETCDF4')
    # global attributes
    ncfile.title = "ASTER image data and geolocation information."
    ncfile.history = "Created " + time.ctime(time.time())
    ncfile.author = "Theresa Mieslinger: theresa.mieslinger@mpimet.mpg"
    ncfile.description = ("This file contains ASTER L1B radiances, together with "
                          +"geolocation information, a cloud mask, and cloud top "
                          +"height estimates. Only bands from the visual near-"
                          +"infrared (VNIR) and thermal infrared (TIR) radiometers"
                          +" are available since 2007. The data is converted and "
                          +"processed from HDF4 files provided by NASA EOSDIS."
                         )
    ncfile.original_file = ai.basename
    ncfile.datetime = str(ai.datetime)
    # create dimensions
    latlon = ncfile.createDimension('latlon', 2)
    pixel_alongtrack_VNIR = ncfile.createDimension('pixel_alongtrack_VNIR',
                                                   np.shape(fields["1"])[0])
    pixel_acrosstrack_VNIR = ncfile.createDimension('pixel_acrosstrack_VNIR',
                                                   np.shape(fields["1"])[1])
    pixel_alongtrack_TIR = ncfile.createDimension('pixel_alongtrack_TIR',
                                                   np.shape(fields["10"])[0])
    pixel_acrosstrack_TIR = ncfile.createDimension('pixel_acrosstrack_TIR',
                                                   np.shape(fields["10"])[1])
    # write 2d variables
    nlat = ncfile.createVariable("latitudesVNIR",
                                 np.float32,
                                 ("pixel_alongtrack_VNIR","pixel_acrosstrack_VNIR"),
                                 zlib=True)
    nlat.units = "degrees_north"
    nlat.long_name = "Latitude"
    nlat.description = "Latitudes on pixel level for VNIR radiometer data"
    nlat[:] = latitudesVNIR

    nlon = ncfile.createVariable("longitudesVNIR",
                                 np.float32, 
                                 ("pixel_alongtrack_VNIR","pixel_acrosstrack_VNIR"),
                                 zlib=True)
    nlon.units = "degrees_east"
    nlon.long_name = "Longitude"
    nlon.description = "Longitudes on pixel level for VNIR radiometer data"
    nlon[:] = longitudesVNIR

    nlatTIR = ncfile.createVariable("latitudesTIR",
                                    np.float32, 
                                    ("pixel_alongtrack_TIR","pixel_acrosstrack_TIR"),
                                    zlib=True)
    nlatTIR.units = "degrees_north"
    nlatTIR.long_name = "Latitude"
    nlatTIR.description = "Latitudes on pixel level for TIR radiometer data"
    nlatTIR[:] = latitudesTIR

    nlonTIR = ncfile.createVariable("longitudesTIR",
                                    np.float32, 
                                    ("pixel_alongtrack_TIR","pixel_acrosstrack_TIR"),
                                    zlib=True)
    nlonTIR.units = "degrees_east"
    nlonTIR.long_name = "Longitude"
    nlonTIR.description = "Longitudes on pixel level for TIR radiometer data"
    nlonTIR[:] = longitudesTIR
    
    for ch in fields.keys():
        if ch in ai.subsensors['VNIR']:
            nrad = ncfile.createVariable(f"radiance_band{ch}",
                                          np.float32,
                                          ("pixel_alongtrack_VNIR", "pixel_acrosstrack_VNIR"),
                                          zlib=True)
        elif ch in ai.subsensors['TIR']:
            nrad = ncfile.createVariable(f"radiance_band{ch}",
                                          np.float32,
                                          ("pixel_alongtrack_TIR", "pixel_acrosstrack_TIR"),
                                          zlib=True)
        nrad.standard_name = "toa_outgoing_radiance_per_unit_wavelength"
        nrad.units = "W m-2 sr-1 um-1"
        nrad.long_name = ("Spectral radiance values at TOA at {:g} to {:g} µm".format(*ai.wavelength_range[ch]))
        nrad[:] = fields[ch]
        
    ### Level 2
    # cloud mask
    nclmask = ncfile.createVariable("cloudmask",
                                    np.float32,
                                    ("pixel_alongtrack_VNIR", "pixel_acrosstrack_VNIR"),
                                    zlib=True)
    nclmask.units = "unitless"
    nclmask.valid_range = [2, 5]
    nclmask.flag_values = [2, 3, 4, 5]
    nclmask.flag_meanings = "confidently_clear probably _clear probably_cloudy confidently_cloudy"
    nclmask.long_name = "ASTER cloud mask"
    nclmask.description = "ASTER cloud mask according to Werner et al., 2016. "
    nclmask[:] = cloudmask
    # cloud top height
    ncth = ncfile.createVariable("cloudtopheight",
                                 np.float32,
                                 ("pixel_alongtrack_TIR", "pixel_acrosstrack_TIR"),
                                 zlib=True)
    ncth.standard_name = "height_at_effective_cloud_top_defined_by_infrared_radiation"
    ncth.units = "m"
    ncth.long_name = "Cloud top height"
    ncth.description = "IR cloud top height estimates following Baum et al., 2012."
    ncth[:] = cthIR * 1000

    sun_zenith = ncfile.createVariable('SunZenith', np.float32)
    sun_zenith.standard_name = "solar_zenith_angle"
    sun_zenith.long_name = 'Sun zenith angle'
    sun_zenith.description = 'sun zenith angle of central pixel'
    sun_zenith.units = 'degree'
    sun_zenith[:] = ai.sunzenith

    sun_azimuth = ncfile.createVariable('SunAzimuth', np.float32)
    sun_azimuth.standard_name = "solar_azimuth_angle"
    sun_azimuth.long_name = 'Sun azimuth angle'
    sun_azimuth.description = 'sun azimuth angle of central pixel'
    sun_azimuth.units = 'degree'
    sun_azimuth[:] = ai.solardirection.azimuth

    scenecenter = ncfile.createVariable('SceneCenter', np.float32, ('latlon'))
    scenecenter.long_name = 'Central pixel (latitude, longitude)'
    scenecenter.description = ''
    scenecenter.units = 'degree'
    scenecenter[:] = ai.scenecenter
    ncfile.close()
    
    return (f"{ai.basename} converted to netCDF")

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