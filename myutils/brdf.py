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
def delta_angle(alpha, beta):
    '''Difference between two angles.'''
    return abs((alpha - beta + 180) % 360 - 180)


def E0_aster(asterchannel):
    '''mean solar exo-atmospheric irradiances [W m-2 um-1] at TOA according to
    Thome et al., 2001 for certain ASTER channels.
    
    Parameters:
        asterchannel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.
    Returns:
        float: solar exo-atmospheric irradiance.
        
    (alternatively take Tobi's values: https://github.com/d70-t/bier/blob/master/
    bier/data/E0_sun.txt)
    '''
    E0 = [1848, 1549, 1114, 1114,
          225.4, 86.63, 81.85, 74.85, 66.49, 59.85,
          np.nan, np.nan, np.nan, np.nan, np.nan]
    channels = ['1', '2', '3N', '3B',
                '4', '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14']
    return E0[channels.index(asterchannel)]
    

def wswdir2uv(ws, wdir):
    u = -ws * np.sin(np.deg2rad(wdir))
    v = -ws * np.cos(np.deg2rad(wdir))
    
    return (u, v)

def uv2wswdir(u, v):
    wdir = np.rad2deg(np.arctan2(v, u))
    wdir[wdir < 0] += 360
    
    return (np.sqrt(u**2 + v**2), wdir)


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
    np.random.seed(42)
    u = np.random.normal(loc=u_avg, scale=u_std, size=size)
    v = np.random.normal(loc=v_avg, scale=v_std, size=size)
    
    return (u, v)


def u_std(u_avg):
    '''Standard deviation in u wind component according to BCO wind data.'''
    return .14 * abs(u_avg) + .3 # .13 ... .16

def v_std(v_avg):
    '''Standard deviation in u wind component according to BCO wind data.'''
    return  .21 * abs(v_avg) + .3
    

def sfc_slope_variance(u, v=None):
    '''The mean square slope components, crosswind and up/downwind following Cox and Munk, 1954.
    
    Parameters:
        u (float): upwind speed @ 10 m height in m/s
        v (float): crosswind speed @ 10 m height in m/s
        
    Returns:
        tuple: mean square slope (combined, upwind, crosswind) in (m/s)**2.
        
    Reference:
        Cox and Munk, 1954, chapter 6.3
    '''
    ws = np.sqrt(u**2 + v**2)
    
    sigma_up2 = 0.00316 * ws
    sigma_cross2 = 0.003 + 0.00192 * ws
    sigma_combined2 = 0.003 + 0.00512 * ws
    
    return (sigma_combined2, sigma_up2, sigma_cross2)


def gaussian_surface_slope1D(theta_n, sigma):
    ''' 1D gaussian surface slope distribution. 
    
    Parameters:
        theta_n (float): tilt angle between ocean wave facet normal and the vertical [rad].
        sigma (float): surface slope variance [(m/s)**2].
    
    Returns:
        float: 1D Gaussian surface slope distribution.
        
    Reference:
        Lin et al., 2016: equation (9)
    '''        
    return 1 / (np.pi * sigma) * np.exp(-np.tan(theta_n)**2 / sigma)


def gaussian_surface_slope_2D(theta_n, phidiff, sigma_up, sigma_cross):
    '''2D Gaussian surface slope distribution.
    
    Parameters:
        theta_n (float): tilt angle between ocean wave facet normal and the vertical [rad].
        phidiff (float): azimuth angle difference between sun and sfc slope [rad].
        sigma_up (float): surface slope variance in upwind direction in (m/s)**2.
        sigma_cross (float): surface slope variance in crosswind direction in (m/s)**2.
    
    Returns:
        float: 2D Gaussian surface slope distribution.
        
    Reference:
        Lin et al., 2016: equation (10)
        Cox and Munk, 1954: equation (1)
    '''
    
    zx = np.sin(phidiff) * np.tan(theta_n)
    zy = np.cos(phidiff) * np.tan(theta_n)
    e = -((zx**2 / sigma_up) + (zy / sigma_cross)) / 2
    
    return np.exp(e) / (2 * np.pi * sigma_up * sigma_cross)


# angular quantities
def mu_scattering(mu, mup, dphi):
    ''' cosine of scattering angle, equation (7) in Lin et al., 2016.
    
    Parameters;
        mu (float): cosine of the view zenith angle for the incident light.
        mup (float): cosine of the view zenith angle for the reflected light.
        dphi (float): relative azimuth angle in [deg].
    Returns:
        float: cosine of scattering angle.
    '''
    return (-mu * mup + np.sqrt(1 - mu**2) * np.sqrt(1 - mup**2)
            * np.cos(np.deg2rad(dphi)))


def mu_wavefacet(mu, mup, mu_sc):
    '''Cosine of the tilt angle between the ocean wave facet normal and the
    vertical direction.
    
    Parameters:
        mu (float): cosine of the view zenith angle for the incident light.
        mup (float): cosine of the view zenith angle for the reflected light.
        cosTheta (float): cosine of scattering angle.
    
    Returns:
        float: tilt angle of wave facet.
    '''
    return (mu + mup) / np.sqrt(2 * (1 - mu_sc))


def unpol_fresnel(theta_i, nt, ni):
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
    
    theta_t = np.arcsin(np.sin(theta_i) / nr) #[rad]
    
    mu_i = np.cos(theta_i)
    mu_t = np.cos(theta_t)
    
    term1 = (mu_i - nr * mu_t) / (mu_i + nr * mu_t)
    term2 = (mu_t - nr * mu_i) / (mu_t + nr * mu_i)

    return (term1**2 + term2**2) / 2
    

def BRDF(mu_0, phi_0, mu_v, phi_v, u, v=None, nt=1.3, ni=1):
    ''' biderectional reflectance distribution functions over water.
    Shadowing is neglegted.
    
    Parameters:
        mu0 (float): cosine of the view zenith angle for the incident light.
        phi0 (float): sun azimuth angle [deg]
        mup (float): cosine of the view zenith angle for the
                     reflected light.
        phip (float): sensor azimuth angle [deg]
        u (float): upwind speed @ 10 m height [m/s]
        v (float): crosswind speed @ 10 m height [m/s]
        nt (float): refractive index of water
        ni (float): refractive index of air
    
    Returns:
        float: bidirectional reflectance distribution function.
    
    Reference:
        Lin et al., 2016
    '''
    
    mu_sc = mu_scattering(mu_0, mu_v, delta_angle(phi_v, phi_0))
    mu_n = mu_wavefacet(mu_0, mu_v, mu_sc)
      
    forefactor = 1 / (4 * mu_v * mu_0 * mu_n**4)
    
    slope_dist1D = gaussian_surface_slope1D(theta_n=np.arccos(mu_n),
                                            sigma=sfc_slope_variance(u, v)[0]
                                           )
    #slope_dist2D = gaussian_surface_slope_2D(theta_n=np.arccos(mu_n),
    #                                         phidiff=np.deg2rad(delta_angle(phi_0,
    #                                                                        uv2wswdir(u,
    #                                                                                  v)[1]
    #                                                                       )
    #                                                           ),
    #                                         sigma_up=sfc_slope_variance(u, v)[1],
    #                                         sigma_cross=sfc_slope_variance(u, v)[2]
    #                                        )
    
    fresnel = unpol_fresnel(theta_i=np.arccos(mu_sc) - np.pi, # local incidence angle in [rad]
                            nt=nt,
                            ni=ni
                           )
    
    return forefactor * slope_dist1D * fresnel


def rho_diffuse(mu_v, phi_v, u, v=None, nt=1.3, ni=1, mu_nodes=5, phi_nodes=6):
    '''BRDF of diffuse reflected light over ocean surface into the direction of
    a sensor.
    '''
    # use Gauss quadratur Legendre integration to get high accuracy in the mu
    # space with few nodes.
    leggauss = lru_cache()(np.polynomial.legendre.leggauss)
    x, w = leggauss(mu_nodes)
    # translate nodes x from [-1, 1] to [0, 1]
    mu_i = (x + 1) / 2
    # azimuth angles for integrating over the half space
    phi_i = np.linspace(0, 359, phi_nodes)
    
    mu_v = np.asarray(mu_v)
    phi_v = np.asarray(phi_v)
    u = np.asarray(u)
    v = np.asarray(v)
    
    # check whether array shapes are equal. 'set' is a data type similar to list or
    # tuple, that excludes duplicates. If all arrays have the same shape, length of
    # the set-object must be 1.
    if len(set(arr.shape for arr in (mu_v, phi_v, u, v))) == 1: 
        rho = np.zeros(mu_v.shape if mu_v.shape else (1,)) * np.nan
        
        # iterate over multidimensional array:
        for idx in np.ndindex(mu_v.shape):
            brdf = BRDF(*np.meshgrid(mu_i, phi_i),
                        mu_v[idx],
                        phi_v[idx],
                        u=u[idx],
                        v=v[idx],
                        nt=1.3,
                        ni=1,
                        )
            # integrate over half-space and the brdfs at all angles (mu_i, phi_i)
            # the factor mu_i is already included in the I_diff Libradtran calculations.
            rho[idx] = trapz(np.sum(brdf * w, axis=1) / 2,
                             x=np.deg2rad(phi_i),
                             axis=0
                            )
    else:
        raise ValueError('Input arrays mu_v, phi_v, u, and v have differing shapes.')
        
    return rho


def coefficient_beam(tau, mu_0, mu_v):
    return np.pi * np.exp(-(tau / mu_0)) * np.exp(-(tau / mu_v))


#def I_diffuse(tau, mu_0):
#    '''Get corresponding value from the LibRadtran LUT.
#    '''
#    lut = netCDF4.Dataset('/scratch/uni/u237/users/tmieslinger/work/'
#                          +'surface_reflectance/output/LUT_radiance_diffuse_edn.nc', 'r')
#
#    I_diff = lut.variables['radiance'][:].data
#    #I_diff = np.subtract(rad, rad[:,0][:, np.newaxis])
#
#    ind_theta0 = np.argmin(np.abs(lut.variables['theta0'][:].astype(float)
#                                  - np.rad2deg(np.arccos(mu_0))))
#    ind_aod = np.argmin(np.abs(lut.variables['aod'][:].astype(float) - tau))
#   return I_diff[ind_theta0, ind_aod]


def coefficient_diffuse(tau, mu_0, mu_v, I_diff, E_0):
    return I_diff * np.pi / mu_0 / E_0 * np.exp(-tau / mu_v)


def load_LUT_Idiff():
    lut = netCDF4.Dataset('/scratch/uni/u237/users/tmieslinger/work/'
                          +'surface_reflectance/output/LUT_radiance_diffuse_edn.nc', 'r')
    return RBVS(x=np.cos(np.deg2rad(lut.variables['theta0'][:].astype(float)))[::-1],
                    y=lut.variables['aod'][:].astype(float),
                    z=lut.variables['radiance'][:].data[::-1])


I_diffuse = load_LUT_Idiff()


def phaseHenyeyGreenstein(theta_sca, g=0.83):
    '''Henyey Greenstein phase function.
    
    Parameters:
        theta_sca (float): scattering angle in rad.
        g (float): asymmetry parameter.
    '''
    return 1 / (4 * np.pi) * (1 - g**2) / (1 + g**2 - 2 * g * np.cos(theta_sca))**1.5    


def I_diffuse_sglscattered(tau, mu_0, phi_0, mu_v, phi_v, omega0=.9):
    '''Diffuse radiance with a single (aerosol) scattering event reaching a sensor
    without being reflected at the surface. Similar to the "direct" radiance, but
    without mu_0 as the scattering on a (spherical) aerosol does not need to be
    projected to a surface plane area.'''
    theta_sca = np.arccos(mu_scattering(mu_0, mu_v, delta_angle(phi_v, phi_0)))
    pHG = phaseHenyeyGreenstein(theta_sca)
    E_0 = E0_aster('3N')
    
    return (E_0 * pHG * omega0 * mu_0 * (1 - np.exp(-tau / mu_0 - tau / mu_v))
            / (mu_0 + mu_v))


def reflectance_sensor(tau, mu_0, phi_0, mu_v, phi_v, u_avg, v_avg=None, nt=1.3, ni=1):
    u, v = uv_distribution(u_avg, v_avg, np.shape(mu_v))
    rho_beam = BRDF(mu_0=mu_0,
                    phi_0=phi_0,
                    mu_v=mu_v,
                    phi_v=phi_v,
                    u=u,
                    v=v
                   )
    rho_diff = rho_diffuse(mu_v=mu_v,
                           phi_v=phi_v,
                           u=u,
                           v=v
                          )
    c_beam = coefficient_beam(tau=tau,
                              mu_0=mu_0,
                              mu_v=mu_v
                             )
    c_diff = coefficient_diffuse(tau=tau,
                                 mu_0=mu_0,
                                 mu_v=mu_v, 
                                 I_diff=np.squeeze(I_diffuse(mu_0, tau)),
                                 E_0 = E0_aster('3N')
                                )
    
    return c_beam * rho_beam + c_diff * rho_diff