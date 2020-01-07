#!/usr/bin/env python
# coding: utf-8
import numpy as np
import netCDF4
from typhon.cloudmask import aster

def ast_id(filename):
    '''extract YYYY-MM-DD hh:mm:ss from ASTER file name.'''
    x = filename.split('/')[-1].split('_')[2]
    
    return str(x[7:11])+str(x[3:7])+'-'+str(x[11:19])


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