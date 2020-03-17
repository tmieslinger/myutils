#!/usr/bin/env python
# coding: utf-8

import os
import glob
import datetime
from datetime import timedelta

import numpy as np
import xarray as xr
import netCDF4
import cdsapi

import typhon as ty
from typhon.cloudmask import aster


def get_uvspec_input_from_aster(filename):
    """Extract sensor sun geometry information from ASTER image.
    """
    ai = aster.ASTERimage(filename)

    ret = dict()
    
    ret['sza'] = np.array([90 - ai.solardirection.elevation])
    ret['phi0'] = np.array([convert_phi_2uvspecformat(ai.solardirection.azimuth)])

    sensor_zenith, sensor_azimuth = ai.sensor_angles()
    # extract scan line in swath mid, excluding NaNs
    senz = sensor_zenith[np.shape(sensor_zenith)[0] // 2]
    ret['theta_sensor'] = senz[~np.isnan(senz)]
    ret['umus'] = np.cos(np.deg2rad(senz[~np.isnan(senz)]))
    sena = sensor_azimuth[np.shape(sensor_zenith)[0] // 2]
    ret['phi_sensor'] = sena[~np.isnan(senz)]
    ret['phis'] = convert_phi_2uvspecformat(sena[~np.isnan(senz)])

    return ret


def uv_distribution(u_avg, v_avg, size):
    '''calculate wind speed distribution. Scale values (std) acc. to Barbados data.'''
    if u_avg > 0:
        u_std = .3
    else:
        u_std = -.13 * u_avg + .3 #-.16
    #v_std = 0.44
    if v_avg > 0:
        v_std = .3
    else:
        v_std = -.21 * v_avg + .3
    #u_std = .3
    #v_std = .3
    u = np.random.normal(loc=u_avg, scale=u_std, size=size)
    v = np.random.normal(loc=v_avg, scale=v_std, size=size)
    
    return (u, v)
    
    
def uvspec_input(file):
    setup = get_uvspec_input_from_aster(file)
    
    u_avg, v_avg = get_era5uv_2aster(file)
    setup['u'], setup['v'] = uv_distribution(u_avg, v_avg, size=np.shape(setup['umus']))
    setup['ws'] = np.sqrt(setup['u']**2 + setup['v']**2)
    #abs(np.random.normal(loc=np.sqrt(u_avg**2 + v_avg**2), scale=1.75, size=len(setup['umus'])))
    
    setup['aod'] = np.array([get_modisAODavg_2aster(file)])
    if np.isnan(setup['aod']):
        setup['aod'] = np.array([.05]) # low default value over ocean
    #setup['aod'] = np.array([.05])
    print('AOD: '+str(np.round(setup['aod'], 2)))
    
    return setup


def convert_phi_2uvspecformat(phi):
    """Convert solar azimuth angle to the uvspec/libradtran convention
        unit: deg,
        0: sun in south, 90: sun in west,
        0: sensor in north, looking south, 90: sensor in east, looking west.
    """
    return (phi + 180) % 360


### wind speed distribution
def get_era5uv_2aster(file):
    '''Check for file existance and download ERA5 if necessary, read wind speed data, calculate average of pixel corresponding to ASTER image.
        
    Parameters:
        file (str): ASTER granule filename including path.
        
    Returns:
        float: average wind speed [ms-1]
    '''
    LL, LR, UL, UR = get_aster_cornercoordinates(file)
    
    era5path = '/scratch/uni/u237/user_data/tmieslinger/era5/'
    erafile = get_era5_file(path=era5path, dt=aster.ASTERimage(file).datetime)

    lon = read_era5_variable(erafile, 'longitude')
    lat = read_era5_variable(erafile, 'latitude')
    lons, lats = np.meshgrid(lon, lat)
    
    mask = np.logical_and(lons>LL[1],
                          np.logical_and(lons<UR[1],
                                         np.logical_and(lats>LR[0], lats<UL[0])))
        
    u = read_era5_variable(erafile, 'u10')[mask]
    v = read_era5_variable(erafile, 'v10')[mask]
    
    return (np.mean(u), np.mean(v))


def get_aster_cornercoordinates(file):
    
    ai = aster.ASTERimage(file)
    meta = ai.get_metadata()
    # LL = (latitude, longitude)
    LL = ai._convert_metastr(meta['LOWERLEFT'], dtype=tuple)
    LR = ai._convert_metastr(meta['LOWERRIGHT'], dtype=tuple)
    UL = ai._convert_metastr(meta['UPPERLEFT'], dtype=tuple)
    UR = ai._convert_metastr(meta['UPPERRIGHT'], dtype=tuple)
    
    return (LL, LR, UL, UR)


def read_era5_variable(file, var):
    ''' Read variable from ERA5 netCDF file.
    
    Parameters:
        file (str): filename including path.
        var (str): ERA5 variable ['longitude', 'latitude', 'time', 'u10', 'v10']
    
    Returns:
        ndarray: value array of variable
    '''
    data = netCDF4.Dataset(file,'r')
    varlist = [i for i in data.variables]
    
    if var not in varlist:
        print('variable '+var+' is not present in the dataset. Choose one of the following: '+str(varlist))  
        pass
    elif var=='longitude':
        # longitudes are defined from 0to360°E. Change to -180 to 180 °E.
        ret = data.variables['longitude'][:]
        ret = -(360 - ret)
    else:
        ret = data.variables[var][:]

    return np.squeeze(ret.data)


def get_era5_file(path, dt):
    '''ERA5 file name.'''
    dt_era = round_to_full_hour(dt)
    
    if check_era5file_existance(path, dt_era):
        pass
    else:
        download_era5(dt_era, path)
    
    return path + 'era5_uv10m_'+dt_era.strftime('%Y%m%d-%H%M')+'.nc'

    
def check_era5file_existance(path, dt):
    '''Check whether an ERA5 file with surface wind speed
    components u, v exists.

    Parameters:
         path (str): Path to ERA5 data directory.
         dt (obj): datetime object.

     Returns:
        bool: ERA5 file corresponding to datetime
            exists (True) / does not exist (False)
    '''
    file = path + 'era5_uv10m_'+dt.strftime('%Y%m%d-%H%M')+'.nc'
    
    return os.path.exists(file)


def download_era5(dt, path_destination):
    '''Download ERA5 file from ECMWF CDS (last checked:2019-11-14)'''
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'format':'netcdf',
            'variable':[
                '10m_u_component_of_wind','10m_v_component_of_wind'
            ],
            'year':dt.strftime('%Y'),
            'month':dt.strftime('%m'),
            'day':dt.strftime('%d'),
            'time':dt.strftime('%H:%M')
        },
        path_destination+'era5_uv10m_'+dt.strftime('%Y%m%d-%H%M')+'.nc')
    
    return


def round_to_full_hour(dt):
    '''Round datetime object to full hour.'''
    if (dt.minute - 30) > 0:
        ret = dt.replace(microsecond=0, second=0 ,minute=0, hour=dt.hour+1)
    else:
        ret = dt.replace(microsecond=0,second=0,minute=0)
    return ret


# ## get MODIS AOD
def round_to_5min(dt):
    '''Round datetime object to 5 minutes.'''
    if (dt.minute % 5 + dt.second / 60) > 2.5: #round up
        ret = dt + timedelta(minutes=(5 - dt.minute % 5))
    else: #round down
        ret = dt - timedelta(minutes=dt.minute % 5)
        
    return ret.replace(microsecond=0,second=0)


def modis_dt(dt):
    '''MODIS datetimes: 22:30 refers to the time period 22:30 - 22:35.'''
    ret = dt - timedelta(minutes=dt.minute % 5)
        
    return ret.replace(microsecond=0,second=0)


def modis_filename(path, dt):
    '''get MODIS file corresponding to a given datetime.'''
    filelist = glob.glob(f'{path}/MOD04_3K.A{modis_dt(dt):%Y%j}.{modis_dt(dt):%H%M}*')
    if filelist:
        return filelist[0]
    else:
        raise FileNotFoundError


def read_modis_var(path, dt, var):
    ''' Read variable from MODIS netCDF file.
    
    Parameters:
        path (str): data directory path.
        dt (datetime): any datetime object. Not necessarily MODIS output times.
        var (str): variable.
        
    Returns:
        ndarray: value array of variable
    '''
    file = modis_filename(path, dt)
    data = netCDF4.Dataset(file,'r')
    data.set_auto_maskandscale(True)
    varlist = [i for i in data.variables]
    
    if var not in varlist:
        print('variable '+var+' is not present in the dataset. Choose one of the following: '+str(varlist))  
        pass
    elif var=='longitude':
        # longitudes are defined from 0to360°E. Change to -180 to 180 °E.
        ret = data['longitude'][:]
        ret -= 180
    else:
        ret = data[var][:]

    return np.squeeze(ret)


def extent(LL, LR, UL, UR):
    '''Get lat/lon extent from corner coordinates of a granule.'''
    latmin = np.min((LL[0], LR[0]))
    latmax = np.max((UL[0], UR[0]))
    lonmin = np.min((LL[1], UL[1]))
    lonmax = np.max((LR[1], UR[1]))

    return latmin, latmax, lonmin, lonmax


def get_aodvalue(aod, lons, lats, latmin, latmax, lonmin, lonmax):
    for i in range(10):
        mask = np.logical_and(lons>=lonmin,
                          np.logical_and(lons<=lonmax,
                                         np.logical_and(lats>=latmin, lats<=latmax)))
        # check how many pixels are no nans
        if np.sum(~np.isnan(aod[mask])) > 10:
            # pass and continue
            print(f'AOD calculation: area was increased by {i * .5} degree. Average of {np.sum(~np.isnan(aod[mask])):.2f} pixels.')
            return (np.nanmean(aod[mask]), (latmin, latmax, lonmin, lonmax))
        else:
            latmin -= .5
            latmax += .5
            lonmin -= .5
            lonmax += .5
    print(f'AOD calculation: no valid pixels in 5 degree radius found.')
    
    return (np.nan, (latmin, latmax, lonmin, lonmax))


def get_modisAODavg_2aster(file, aodvar='Effective_Optical_Depth_Best_Ocean', wl_layer=0):
    '''Calculate average of pixel corresponding to ASTER image.
        
    Parameters:
        file (str): ASTER granule filename including path.
        
    Returns:
        float: average AOD
    '''
    LL, LR, UL, UR = get_aster_cornercoordinates(file)
    
    mod04path = '/scratch/uni/u237/user_data/tmieslinger/modis/MOD04/'
    dt = aster.ASTERimage(file).datetime

    lons = read_modis_var(mod04path, dt=dt , var='Longitude')
    lats = read_modis_var(mod04path, dt=dt , var='Latitude')
    aod = read_modis_var(mod04path, dt=dt , var=aodvar)[wl_layer]
    aod[aod < 0] = np.nan
    
    aod_avg, _ = get_aodvalue(aod, lons, lats, *extent(*get_aster_cornercoordinates(file)))
    
    return aod_avg