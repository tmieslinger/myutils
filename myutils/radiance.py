#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import scipy
from scipy.integrate import trapz
from scipy.interpolate import RectBivariateSpline as RBVS
from functools import lru_cache

'''This code can calculate an estimate of clear-sky radiance or reflectance
over tropical ocean for a given sensor-sun geometry, atmospheric aerosol
conditions, and surface wind speed.
In short, 3 radiance components are combined: (1) the direct beam traveling
through the atmosphere and being reflected at the surface back to a sensor's
position, (2) the hemispheric diffuse irradince reflected at the surface in the
direction of a sensor, and (3) the diffuse atsmopheric radiance reaching the
sensor after a single scattering event.

The main literature for equations and calculation approaches:
[0] Cox, C., and Munk, W., 1954. Measurement of the Roughness of the Sea Surface from Photographs of the Sun’s Glitter," J. Opt. Soc. Am. 44, 838-850.
[1] Lin, Z., Li, W., Gatebe, C., Poudyal, R., and Stamnes, K. (2016). Radiative transfer simulations of the two-dimensional ocean glint reflectance and determination of the sea surface roughness, Appl. Opt. 55, 1206-1215.
[2] Stamnes, K., Thomas, G., & Stamnes, J. (2017). Radiative Transfer in the Atmosphere and Ocean. Cambridge: Cambridge University Press. doi:10.1017/9781316148549.
[3] Koelling, T., 2015. Characterization, calibration and operation of a hyperspectral sky imager. Master’s thesis.
'''

def theta_phi2vector(theta, phi):
    '''Conversion from angles to vector.
    Convert zenith and azimuth angle [rad] to unit vector in (x, y, z)
    coordinate system.
    
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
    '''Conversion from angles to vector.
    Convert cosine of zenith and azimuth angle to unit vector in (x, y, z)
    coordinate system.
    
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
    '''Cosine of theta.
    Theta is the angle between unit_vector and zenith (z-axis) and is given by
    the arccos of the scalar product of the vectors.
    
    Parameters:
        unit_vector (ndarray): unit vector (vx, vy, vz).
        
    Returns:
        ndarray: cosine of theta, i.e. z component of unit vector.
    '''
    return unit_vector[2]


def phi(unit_vector):
    '''Polar angle phi.
    The angle between unit vector and the North (y-axis).
    
    Parameters:
        unit_vector (ndarray): unit vector (vx, vy, vz).
        
    Returns:
        ndarray: phi angle in rad.
    '''
    return np.arctan2(unit_vector[1], unit_vector[0])


def sfc_slope_variance(ws):
    '''Surface slope variance.
    The mean square slope components, crosswind and up/downwind following Cox
    and Munk, 1954.
    
    Parameters:
        ws (ndarray): upwind speed at 10 m height in m/s
        
    Returns:
        tuple: mean square slope (combined, upwind, crosswind) in (m/s)**2.
        
    Reference:
        Cox, C., and Munk, W., 1954. Measurement of the Roughness of the Sea
        Surface from Photographs of the Sun’s Glitter," J. Opt. Soc. Am. 44,
        838-850. (chapter 6.3)
    '''    
    sigma_up2 = 0.00316 * ws
    sigma_cross2 = 0.003 + 0.00192 * ws
    sigma_combined2 = 0.003 + 0.00512 * ws
    
    return sigma_combined2, sigma_up2, sigma_cross2


def mu_sca(sun, view):
    '''Cosine of scattering angle.
    Calculated as the scalar product of sun and view vectors through Einstein
    sum convention.
        
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).
    
    Returns:
        ndarray: cosine of scattering angle.

    '''
    return np.einsum('i...,i...->...', sun, view)


def wavefacet_normal(sun, view):
    '''Ocean wave facet normal.
        
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).
        
    Returns:
        float: wave facet normal.
    '''
    #(mu(sun) + mu(view)) / np.sqrt(2 * (1 - mu_sca(sun, view)))
    n = sun + view
    # l2 norm, i.e. abs,  along axis 0
    n /= np.linalg.norm(n, axis=0)
    
    return n


def gaussian_surface_slope1D(mu_n, sigma2):
    ''' 1D gaussian surface slope distribution. 
    
    Parameters:
        mu_n (float): cosine of tilt angle between ocean wave facet normal and
            zenith (z-axis).
        sigma2 (float): surface slope variance acc to Cox and Munk in (m/s)**2.
    
    Returns:
        float: 1D Gaussian surface slope distribution.
        
    Reference:
        Lin, Z., Li, W., Gatebe, C., Poudyal, R., and Stamnes, K. (2016).
        Radiative transfer simulations of the two-dimensional ocean glint
        reflectance and determination of the sea surface roughness, Appl. Opt.
        55, 1206-1215. (equation 9)
    '''        
    return 1 / (np.pi * sigma2) * np.exp(-(1 - mu_n**2) / mu_n**2 / sigma2)


def unpol_fresnel(sun, view, nt=1.333, ni=1):
    '''Unpolarized Fresnel reflection coefficient.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        nt (float): refractive index of transmitted medium.
        ni (float): refractive index of incoming medium.
        
    Returns:
        float: Fresnel reflection coefficient.
    
    Reference:
        Stamnes, K., Thomas, G., & Stamnes, J. (2017). Radiative Transfer in
        the Atmosphere and Ocean. Cambridge: Cambridge University Press.
        doi:10.1017/9781316148549
    '''
    nr = nt / ni
    
    n = wavefacet_normal(sun, view)
    # cosine of incidence and transmission angle.
    mu_i = mu_sca(sun, n)
    mu_t = np.sqrt(1 - (1 - mu_i * mu_i) / nr**2)
    
    term1 = (mu_i - nr * mu_t) / (mu_i + nr * mu_t)
    term2 = (mu_t - nr * mu_i) / (mu_t + nr * mu_i)

    return (term1**2 + term2**2) / 2


def brdf(sun, view, ws):
    '''Bi-directional reflectance function.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        ws (ndarray): surface wind speed estimate in m/s. Needs to be
            broadcastable to `sun` and `view` arrays.
        
    Returns:
        float: Bi-directional reflectance function.
    
    Reference:
        Lin, Z., Li, W., Gatebe, C., Poudyal, R., and Stamnes, K. (2016).
        Radiative transfer simulations of the two-dimensional ocean glint
        reflectance and determination of the sea surface roughness, Appl. Opt.
        55, 1206-1215. (equation 5)
    '''
    mu_n = mu(wavefacet_normal(sun, view))
    
    forefactor = 1 / (4 * mu(view) * mu(sun) * mu_n**4)
    
    slope_dist1D = gaussian_surface_slope1D(mu_n, sfc_slope_variance(ws)[0])
    
    fresnel = unpol_fresnel(sun, view)
    
    return forefactor * slope_dist1D * fresnel


def hbrdf(view, ws, mu_nodes=5, phi_nodes=7):
    '''Hemispheric bi-directional reflectance function.
    BRDF integrated over half space.
    
    Parameters:
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        ws (ndarray): surface wind speed estimate in m/s. Needs to be
            broadcastable to `sun` and `view` arrays.
        mu_nodes (int): nodes / points where BRDF is evaluated in cos(theta)
            direction range [0,1].
        phi_nodes (int): nodes of BRDF evaluation in phi direction [0, 2pi].
        
    Returns:
        float: hemispheric BRDF.
    
    Reference:
        Lin, Z., Li, W., Gatebe, C., Poudyal, R., and Stamnes, K. (2016).
        Radiative transfer simulations of the two-dimensional ocean glint
        reflectance and determination of the sea surface roughness, Appl. Opt.
        55, 1206-1215. (equation 4)
    
    '''
    # use Gauss quadratur Legendre integration to get high accuracy in the mu
    # space with few nodes.
    leggauss = lru_cache()(np.polynomial.legendre.leggauss)
    x, w = leggauss(mu_nodes)
    # zenith angles for integrating over half space. Translate nodes x from
    # [-1, 1] to [0, 1]
    w /= 2
    mu_i = (x + 1) / 2
    # azimuth angles for integrating over the half space
    phi_i = np.linspace(0, 2 * np.pi, phi_nodes)
    # dim (3 x N x M)
    sun_i = mu_phi2vector(mu_i[:, np.newaxis], phi_i[np.newaxis, :])
    
    # sun extened by N x M and additional view dimensions
    brdf_i = brdf(sun_i[(Ellipsis,) + (np.newaxis,) * len(view.shape[1:])],
                  view[:, np.newaxis, np.newaxis, ...],
                  ws
                 )
    # integrate over half-space and the brdf_i at all angles (mu_i, phi_i)
    hbrdf = trapz(np.einsum('n...,n->...', brdf_i, w),
                  x=phi_i,
                  axis=0
                 )
        
    return hbrdf


def load_LUT_Idiff():
    '''LUT for diffuse downwelling irradiance.
    Read LUT generated with LibRadtran for diffuse downward irradiance (edn)
    reaching a sensor situated at the surface and looking up (zenith).
    
    Returns:
        `scipy.interpolate.RectBivariateSpline`: diffuse atmospheric irradiance.
            irradiance(sun zenith, AOD)
    '''
    lut = netCDF4.Dataset('/scratch/uni/u237/users/tmieslinger/work/surface_'
                          + 'reflectance/output/LUT_radiance_diffuse_'
                          + 'transmittance.nc', 'r')
    return RBVS(x=np.cos(np.deg2rad(lut.variables['theta0'][:].astype(float))
                        )[::-1],
                    y=lut.variables['aod'][:].astype(float),
                    z=lut.variables['radiance'][:].data[::-1])


I_diffuse = load_LUT_Idiff()


def edown(sun, tau):
    '''Diffuse downwelling irradiance.
    The diffuse atmospheric irradiance is read from a LUT depending on a given
    sun zenith angle (sun vector) and the atmospheric aerosol optical thickness.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        tau (ndarray): aerosol optical thickness.
        
    Returns:
        ndarray: diffuse downwelling irradiance.
    '''
    return np.squeeze(I_diffuse(mu(sun), tau))


def transmittance_ground(sun, view, ws, tau):
    '''Sun light reaching a sensor with ground contact on it's way.
    Radiance reaching a sensor from direct sun beam and diffuse downwelling
    irradiance both being reflected at a surface according to it's BRDF.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        ws (ndarray): surface wind speed estimate in m/s. Needs to be
            broadcastable to `sun` and `view` arrays.
        tau (ndarray): aerosol optical thickness.
        
    Returns:
        ndarray: radiance at sensor with ground contact.

    '''
    direct = np.exp(-tau / mu(sun)) * brdf(sun, view, ws)
    diffuse = edown(sun, tau) * hbrdf(view, ws)
    
    return (direct + diffuse) * np.exp(-tau / mu(view))


def phaseHenyeyGreenstein(sun, view, g=0):
    '''Henyey Greenstein phase function.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        g (float): asymmetry parameter. The mean cosine of the scattering angle
            found by integration over the complete scattering phase function.
            g ~ 0.85 for cloud droplets and g -> 0 for air molecules.
        
    Returns:
        ndarray: Henyey Greenstein phase function.
        
    Reference:
        Henyey, L. G., and J. L. Greenstein 1941. Astrophys. J.. 93. p.70
    '''
    return (1 / (4 * np.pi) * (1 - g**2) / (1 + g**2 - 2 * g
                                           * mu_sca(sun, view))**1.5)
    

def transmittance_atm(sun, view, tau, omega0=.9):
    '''Diffuse atmospheric radiance.
    Radiance reaching a sensor with a single (aerosol) scattering event and no
    contact to the ground. Derivation is similar to the "direct" radiance, but
    without mu_0 as the scattering on a (spherical) aerosol does not need to be
    projected onto a surface plane.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        tau (ndarray): aerosol optical thickness.
        omega0 (float): single scattering albedo. The ratio of the scattering
            coefficient to the extinction coefficient. In the visible range
            omega0 is close to unity.
    
    Returns:
        (ndarray): diffuse atmospheric radiance.
    
    Reference:
        Koelling, T., 2015. Characterization, calibration and operation of a
        hyperspectral sky imager. Master’s thesis.
    '''
    pHG = phaseHenyeyGreenstein(sun, view)
    
    return (pHG * omega0 * mu(sun)
            * (1 - np.exp(-tau / mu(sun) - tau / mu(view)))
            / (mu(sun) + mu(view)))


def transmittance(sun, view, ws, tau):
    '''Transmittance for given sensor - sun geometry.
    Sun light reaching a sensor from 3 different components, (1) the direct sun
    beam traveling through the atmosphere and being reflected at the surface
    back to a sensor's position, (2) the hemispheric diffuse irradince
    reflected at the surface in the direction of a sensor, both combined in
    function `transmittance_ground`, and (3) the diffuse atsmopheric radiance
    reaching the sensor after a single scattering event calculated in
    `transmittance_atm`.
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        ws (ndarray): surface wind speed estimate in m/s. Needs to be
            broadcastable to `sun` and `view` arrays.
        tau (ndarray): aerosol optical thickness.
        
    Returns:
        ndarray: transmittance seen by sensor.
    '''
    return (transmittance_ground(sun, view, ws, tau)
            + transmittance_atm(sun, view, tau))


def reflectance(sun, view, ws, tau):
    '''Reflectance for a given sensor - sun geometry.
    
    See also:
        `transmittance`
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        ws (ndarray): surface wind speed estimate in m/s. Needs to be
            broadcastable to `sun` and `view` arrays.
        tau (ndarray): aerosol optical thickness.
        
    Returns:
        ndarray: reflectance seen by sensor.
    '''
    return np.pi / mu(sun) * transmittance(sun, view, ws, tau)


def reflectance_atm(sun, view, tau, omega0=.9):
    '''Reflectance from diffuse atmospheric sun light.
    
    See also:
        `transmittance_atm`
    
    Parameters:
        sun (ndarray): unity vector into sun. First 3 dimensions are (x, y, z),
            further dimensions for 1d, or 2d field calculations.
        view (ndarray): unity vector into sensor. First 3 dimensions are
            (x, y, z).        
        tau (ndarray): aerosol optical thickness.
        omega0 (float): single scattering albedo.
    
    Returns:
        (ndarray): diffuse atmospheric reflectance.
    '''
    return np.pi / mu(sun) * transmittance_atm(sun, view, tau, omega0)