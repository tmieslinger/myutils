#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

'''BRDF according to Cox and Munk. 
All functions taken from Lin et al. 2016 "Radiative transfer simulations of the two-dimensional ocean glint reflectance and determination of the sea surface roughness", except Fresnel equation, which was taken from Stamnes - Radiative Transfer in Atmosphere and Ocean.
'''
# %% define slope variance

def sigmasq_upwind(w):
    ''' upwind wind slope variance
    
        w  wind speed @ 10m in m/s'''
    return 0.003+0.00192 * w #[(m/s)**2]
    

def sigmasq_crosswind(w):
    ''' cross wind slope variance
    
        w  wind speed @ 10m in m/s'''
    
    return 0.00316 * w #[(m/s)**2]

def sigmasq_wind(u,v=None):
    ''' summed wind slope variance
    
        u  upwind speed @ 10m in m/s
        v  crosswind speed @ 10m in m/s'''

    if v is None:
        v=u
                
    return sigmasq_upwind(u)+sigmasq_crosswind(v) #[(m/s)**2]


# %% gaussian surface slope 

def gaussian_surface_slope1D(theta_n,sigma):
    ''' 1D gaussian surface slope '''    
    
    forefactor=1/(np.pi*sigma**2)
    
    exponent=np.tan(theta_n)/sigma
    
    p=forefactor*np.exp(-(exponent)**2)
    
    return p


# %% angular quantities

def cosineTheta(mu,mup,dphi):
    ''' cosine of scattering angle '''
    
    return -mu*mup+np.sqrt(1-mu**2)*np.sqrt(1-mup**2)*np.cos(dphi)


def normal_cosine(mu,mup,cosTheta):
    
    return (mu+mup)/np.sqrt(2*(1-cosTheta))


def unpol_fresnel_eq(theta_i, n_t, n_i):
    ''' upolarized Fresnel reflection coefficient
        theta_i     incidence angle
        n_t         refractive index of transmitted medium
        n_i         refractive index of incoming medium
    '''
    
   
    n_r=n_t/n_i
    
    theta_t=np.arcsin(np.sin(theta_i)/n_r)
    
    mu_i=np.cos(theta_i)
    mu_t=np.cos(theta_t)
    
    
    term1=(mu_i-n_r*mu_t)/(mu_i+n_r*mu_t)
    term2=(mu_t-n_r*mu_i)/(mu_t+n_r*mu_i)

    rhoF=1./2.*(term1**2+term2**2)
    
    return rhoF
    
# %% BRDF    
def BRDF(theta_sun,theta_sensor, dphi, u, v=None, n_t=1.3, n_i=1):
    ''' biderectional reflectance distribution functions over water 
    
    theta_sun           sun zenith angle [rad] 
    theta_sensor        sensor zenith angle [rad]
    dphi                azimuth difference between sun and sensor
    w                   wind speed [m/s]
    n_t                 refractive index of water
    n_i                 refractive index of air
    '''
    
    
    #angles    
    mu=np.cos(theta_sun)
    mup=np.cos(theta_sensor)
    
    cosTheta=cosineTheta(mu,mup,dphi)
    Theta=np.arccos(cosTheta)
    
    mu_n=normal_cosine(mu,mup,cosTheta)
    theta_n=np.arccos(mu_n)
    
    #slope variance    
    if v is not None:
        sigma=np.sqrt(sigmasq_wind(u,v))    
    else:
        sigma=np.sqrt(sigmasq_wind(u))
    
    
    #additional factors   
    forefactor=1/(4*mup*mu*mu_n**4)
    
    #slope distro
    slope_dist=gaussian_surface_slope1D(theta_n,sigma)
    
    # local incidence angle
    theta_i=(Theta-np.pi)/2      
    
    #fresnel reflection
    fresnel=unpol_fresnel_eq(theta_i, n_t, n_i)
    
    #the BRDF
    rho=forefactor*slope_dist*fresnel
    
    return rho