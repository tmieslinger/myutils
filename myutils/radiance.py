#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import scipy
from scipy.integrate import trapz
from scipy.interpolate import RectBivariateSpline as RBVS
from functools import lru_cache

'''docu...
'''

def theta_phi2vector(theta, phi):
    '''Convert zenith and azimuth angle to unit vector in (x, y, z) coordinate
    system.
    
    Parameters:
        theta (ndarray): zenith angle in rad.
        phi (ndarray): azimuth angle in rad.
    
    Returns:
        ndarray: unit vector (vx, vy, vz).
    '''
    vx = np.sin(theta) * np.sin(phi)
    vy = np.sin(theta) * np.cos(phi)
    vz = np.cos(theta)
    vx, vy, vz = np.broadcast_arrays(vx, vy, vz)
    
    return np.stack([vx, vy, vz], axis=0)


def mu_phi2vector(mu, phi):
    '''Convert zenith and azimuth angle to unit vector in (x, y, z) coordinate
    system.
    
    Parameters:
        mu (ndarray): cosine of zenith angle.
        phi (ndarray): azimuth angle in rad.
    
    Returns:
        ndarray: unit vector (vx, vy, vz).
    '''
    st = np.sqrt(1 - mu * mu)
    
    vx = st * np.sin(phi)
    vy = st * np.cos(phi)
    vz = mu
    # match the shapes
    vx, vy, vz = np.broadcast_arrays(vx, vy, vz)
    
    return np.stack([vx, vy, vz], axis=0)



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

def wavefacet_normal(sun, view):
    '''Cosine of the tilt angle between the ocean wave facet normal and the
    vertical direction.
        
    Returns:
        float: tilt angle of wave facet.
    '''
    #(mu(sun) + mu(view)) / np.sqrt(2 * (1 - mu_sca(sun, view)))
    n = sun + view
    # l2 norm, i.e. abs,  along axis 0
    n /= np.linalg.norm(n, axis=0)
    
    return n


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


def unpol_fresnel(sun, view, nt=1.333, ni=1):
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
    
    #theta_i = np.arccos(mu_sca(sun, view)) - np.pi
    n = wavefacet_normal(sun, view)
    mu_i = mu_sca(sun, n)
    mu_t = np.sqrt(1 - (1 - mu_i * mu_i) / nr**2) #[rad]
    
    term1 = (mu_i - nr * mu_t) / (mu_i + nr * mu_t)
    term2 = (mu_t - nr * mu_i) / (mu_t + nr * mu_i)

    return (term1**2 + term2**2) / 2


def brdf(sun, view, ws):
    '''Bi-directional reflectance function.'''
    mu_n = mu(wavefacet_normal(sun, view))
    forefactor = 1 / (4 * mu(view) * mu(sun) * mu_n**4)
    
    slope_dist1D = gaussian_surface_slope1D(mu_n, sfc_slope_variance(ws)[0])
    
    fresnel = unpol_fresnel(sun, view)
    
    return forefactor * slope_dist1D * fresnel


def hbrdf(view, ws, mu_nodes=5, phi_nodes=7):
    '''Hemispheric bi-directional reflectance function.'''
    # use Gauss quadratur Legendre integration to get high accuracy in the mu
    # space with few nodes.
    leggauss = lru_cache()(np.polynomial.legendre.leggauss)
    x, w = leggauss(mu_nodes)
    # translate nodes x from [-1, 1] to [0, 1]
    w /= 2
    mu_i = (x + 1) / 2
    # azimuth angles for integrating over the half space
    phi_i = np.linspace(0, 2 * np.pi, phi_nodes)
    # dim (3 x N x M)
    sun_i = mu_phi2vector(mu_i[:, np.newaxis], phi_i[np.newaxis, :])
    
    # dim view[] extened by N x M and additional view dimensions
    brdf_i = brdf(sun_i[(Ellipsis,) + (np.newaxis,) * len(view.shape[1:])],
                  view[:, np.newaxis, np.newaxis, ...],
                  ws
                 )
    # integrate over half-space and the brdf_i at all angles (mu_i, phi_i)
    # (the factor mu_i is already included in the edown() Libradtran
    # calculations.)
    hbrdf = trapz(np.einsum('n...,n->...', brdf_i, w),
                  x=phi_i,
                  axis=0
                 )
        
    return hbrdf


def load_LUT_Idiff():
    '''Read LUT generated with LibRadtran for the diffuse radiance reaching a
    sensor situated at the surface and looking up (zenith).'''
    lut = netCDF4.Dataset('/scratch/uni/u237/users/tmieslinger/work/surface_reflectance'
                          +'/output/LUT_radiance_diffuse_transmittance.nc', 'r')
    return RBVS(x=np.cos(np.deg2rad(lut.variables['theta0'][:].astype(float)))[::-1],
                    y=lut.variables['aod'][:].astype(float),
                    z=lut.variables['radiance'][:].data[::-1])


I_diffuse = load_LUT_Idiff()


def edown(sun, tau):
    '''Diffuse radiance reaching the surface.'''
    return np.squeeze(I_diffuse(mu(sun), tau))


def transmittance_ground(sun, view, ws, tau):
    '''Reflectance from direct and diffuse radiance reflected at the surface and
    transmitted back to the sensor location.'''
    direct = np.exp(-tau / mu(sun)) * brdf(sun, view, ws)
    diffuse = edown(sun, tau) * hbrdf(view, ws)
    
    return (direct + diffuse) * np.exp(-tau / mu(view))


def phaseHenyeyGreenstein(sun, view, g=0.83):
    '''Henyey Greenstein phase function.
    
    Parameters:
        theta_sca (float): scattering angle in rad.
        g (float): asymmetry parameter.
    '''
    return 1 / (4 * np.pi) * (1 - g**2) / (1 + g**2 - 2 * g * mu_sca(sun, view))**1.5    
    

def transmittance_atm(sun, view, tau, omega0=.9):
    '''Diffuse radiance with a single (aerosol) scattering event reaching a sensor
    without being reflected at the surface. Similar to the "direct" radiance, but
    without mu_0 as the scattering on a (spherical) aerosol does not need to be
    projected to a surface plane area.'''
    pHG = phaseHenyeyGreenstein(sun, view)
    
    return (pHG * omega0 * mu(sun) * (1 - np.exp(-tau / mu(sun) - tau / mu(view)))
            / (mu(sun) + mu(view)))
    


def transmittance(sun, view, ws, tau):
    '''Transmittance for given sensor - sun geometry.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are (x, y, z).
        ws (ndarray): wind speed. Shape must match `sun` and `view` parameters when
            excluding the first 3 dimensions (x, y, z).
        tau (ndarray): aerosol optical thickness.
        
    Returns:
        ndarray: reflectance seen by sensor.
    '''
    return transmittance_ground(sun, view, ws, tau) + transmittance_atm(sun, view, tau)


def reflectance(sun, view, ws, tau):
    return np.pi / mu(sun) * transmittance(sun, view, ws, tau)


def reflectance_atm(sun, view, tau):
    return np.pi / mu(sun) * transmittance_atm(sun, view, tau)