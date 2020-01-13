#!/usr/bin/env python
# coding: utf-8
import numpy as np
import netCDF4

from PIL import Image
from typhon.cloudmask import aster, cloudstatistics


def asterID(filename):
    '''extract YYYY-MM-DD hh:mm:ss from ASTER file name.
    Returns:
        aster ID (str): YYYYMMDD_HHmmSS
    '''
    x = filename.split('/')[-1].split('_')[2]
    
    return str(x[7:11])+str(x[3:7])+'_'+str(x[11:19])


def strrnd(x):
    '''round float to 3 digits and return its string beginning 
    with an underscore.
    '''
    return '_' + str('{:.3f}'.format(round(x, 3)))


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
    reflection082 = ai.get_reflectance('3N')
    bt11 = ai.get_brightnesstemperature('14')
    cthIR = ai.get_cloudtopheight()
    # 1d cloud field statistics
    cf = cloudstatistics.cloudfraction(cloudmask)
    iorg = cloudstatistics.iorg(cloudmask, connectivity=2)
    scai = cloudstatistics.scai(cloudmask, connectivity=2)
    
    # create output netcdf file
    ncfile = netCDF4.Dataset(path2ofile + '/TERRA_ASTER_' + asterID(file) + '.nc',
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
    
    nrefl = ncfile.createVariable('reflection082', np.float32, 
                                ('pixel_alongtrack_VNIR',
                                 'pixel_acrosstrack_VNIR'))
    nrefl.units = 'unitless'
    nrefl.long_name = 'Relfection at 0.82 micron'
    nrefl.description = 'reflection derived from channel 3N at 0.82 micron'
    nrefl[:] = reflection082

    nbt = ncfile.createVariable('brightness_temperature11', np.float32, 
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
    niorg[:] = iorg

    nscai = stats.createVariable('scai', np.float32)
    nscai.units = 'unitless'
    nscai.long_name = 'Simple convective aggregation index'
    nscai.description = ('cloud cluster index Simple Convective Aggregation Index'
                       +'(SCAI) following Tobin, Bony, and Roca, 2012')
    nscai[:] = scai
    
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

    scenecornerLL = info.createVariable('SceneCornerLL', np.float32, ('latlon'))
    scenecornerLL.long_name = 'Lower left corner (latitude, longitude)'
    scenecornerLL.description = ''
    scenecornerLL.unit = 'degree'
    scenecornerLL[:] = ai.cornercoordinates.LOWERLEFT

    scenecornerLR = info.createVariable('SceneCornerLR', np.float32, ('latlon'))
    scenecornerLR.long_name = 'Lower right corner (latitude, longitude)'
    scenecornerLR.description = ''
    scenecornerLR.unit = 'degree'
    scenecornerLR[:] = ai.cornercoordinates.LOWERRIGHT

    scenecornerUR = info.createVariable('SceneCornerUR', np.float32, ('latlon'))
    scenecornerUR.long_name = 'Upper right corner (latitude, longitude)'
    scenecornerUR.description = ''
    scenecornerUR.unit = 'degree'
    scenecornerUR[:] = ai.cornercoordinates.UPPERRIGHT

    scenecornerUL = info.createVariable('SceneCornerUL', np.float32, ('latlon'))
    scenecornerUL.long_name = 'Upper left corner (latitude, longitude)'
    scenecornerUL.description = ''
    scenecornerUL.unit = 'degree'
    scenecornerUL[:] = ai.cornercoordinates.UPPERLEFT
    
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