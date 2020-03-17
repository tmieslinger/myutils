#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import scipy
from scipy.integrate import trapz
from scipy.interpolate import RectBivariateSpline as RBVS
from functools import lru_cache

'''BRDF according to Cox and Munk. 
All functions taken from Lin et al. 2016 "Radiative transfer simulations of the two-dimensional ocean glint reflectance and determination of the sea surface roughness", except Fresnel equation, which was taken from Stamnes - Radiative Transfer in Atmosphere and Ocean.
'''

def mu(unit_vector):
    '''Cosine of theta, while theta is the angle between unit_vector and z-axis and
    is given by the arccos of the scalar product of the vectors.
    Returns:
        z component of unit vector.
    '''
    return unit_vector[2]


def sfc_slope_variance(ws):
    '''The mean square slope components, crosswind and up/downwind following Cox and
    Munk, 1954.
    
    Parameters:
        ws (ndarray): upwind speed at 10 m height in m/s
        
    Returns:
        tuple: mean square slope (combined, upwind, crosswind) in (m/s)**2.
        
    Reference:
        Cox and Munk, 1954, chapter 6.3
    '''    
    sigma_up2 = 0.00316 * ws
    sigma_cross2 = 0.003 + 0.00192 * ws
    sigma_combined2 = 0.003 + 0.00512 * ws
    
    return sigma_combined2, sigma_up2, sigma_cross2



def mu_sca(sun, view):
    '''Scalar product of sun and view vectors through Einstein sum convention.'''
    return np.einsum('i...,i...->...', sun, view)

def mu_wavefacet(sun, view):
    '''Cosine of the tilt angle between the ocean wave facet normal and the
    vertical direction.
        
    Returns:
        float: tilt angle of wave facet.
    '''
    return (mu(sun) + mu(view)) / np.sqrt(2 * (1 - mu_sca(sun, view)))


def gaussian_surface_slope1D(mu_n, sigma2):
    ''' 1D gaussian surface slope distribution. 
    
    Parameters:
        theta_n (float): tilt angle between ocean wave facet normal and the vertical [rad].
        sigma2 (float): surface slope variance [(m/s)**2].
    
    Returns:
        float: 1D Gaussian surface slope distribution.
        
    Reference:
        Lin et al., 2016: equation (9)
    '''        
    return 1 / (np.pi * sigma2) * np.exp(-(1 - mu_n**2) / mu_n**2 / sigma2)


def unpol_fresnel(sun, view, nt=1, ni=1.333):
    ''' unpolarized Fresnel reflection coefficient.
    
    Parameters:
        theta_i     incidence angle [rad].
        n_t         refractive index of transmitted medium.
        n_i         refractive index of incoming medium.
        
    Returns:
        float: Fresnel reflection coefficient.
    
    Reference:
        Stamnes, K., Thomas, G., & Stamnes, J. (2017). Radiative Transfer in
        the Atmosphere and Ocean. Cambridge: Cambridge University Press.
        doi:10.1017/9781316148549
    '''
    nr = nt / ni
    theta_i = np.arccos(mu_sca(sun, view)) - np.pi
    
    theta_t = np.arcsin(np.sin(theta_i) / nr) #[rad]
    
    mu_i = np.cos(theta_i)
    mu_t = np.cos(theta_t)
    
    term1 = (mu_i - nr * mu_t) / (mu_i + nr * mu_t)
    term2 = (mu_t - nr * mu_i) / (mu_t + nr * mu_i)

    return (term1**2 + term2**2) / 2


def brdf(sun, view, ws):

    mu_n = mu_wavefacet(sun, view)
    forefactor = 1 / (4 * mu(view) * mu(sun) * mu_n**4)
    
    slope_dist1D = gaussian_surface_slope1D(mu_n, sfc_slope_variance(ws)[0])
    
    fresnel = unpol_fresnel(sun, view)
    
    return forefactor * slope_dist1D * fresnel


def reflectance_ground(sun, view, ws, tau):
    '''Reflectance from direct and diffuse radiance reflected at the surface and
    transmitted back to the sensor location.'''
    direct = np.exp(-tau / mu(sun)) * brdf(sun, view, ws)
    diffuse = edown(sun, tau) * hbrdf(view, ws)
    
    return (direct + diffuse) * exp(-tau / mu(view))


def reflectance(sun, view, ws, tau):
    '''Reflectance for given sensor - sun geometry.
    
    Parameters:
        sun (ndarray): unity vector into sun.
        view (ndarray): unity vector into sensor.
        ws (ndarray): wind speed.
        tau (ndarray): aerosol optical thickness.
        
    Returns:
        ndarray: reflectance seen by sensor.
    '''
    return reflectance_ground(sun, view, ws, tau) + reflectance_atm(sun, view, ws, tau)