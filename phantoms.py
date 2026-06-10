# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:48:12 2026

@author: chitnis
"""
import numpy as np
import helpers
from scipy.constants import c, epsilon_0, mu_0

class SSL10_18G:
    filename = 'SSL10-18G'
    fmin = 10e9
    fmax = 18e9

    epsr_inf = 3.65
    sigma0 = 2.15e-6
    depsr_1 = 45.54
    depsr_2 = 4.93
    depsr_3 = 0.00
    depsr_4 = 0.00
    tau_1 = 2.63e-10 
    tau_2 = 2.84e-11
    tau_3 = 0.0
    tau_4 = 0.0
    alpha_1 = 0.12
    alpha_2 = 0.48
    alpha_3 = 0.00
    alpha_4 = 0.00
    
    epsr_Tgrad = 1.00
    sigma_Tgrad = 3.05
    

class SSL18_24G:
    filename = 'SSL18-24G'
    fmin = 18e9
    fmax = 24e9

    epsr_inf = 3.65
    sigma0 = 2.15e-6
    depsr_1 = 45.54
    depsr_2 = 4.93
    depsr_3 = 0.00
    depsr_4 = 0.00
    tau_1 = 2.63e-10 
    tau_2 = 2.84e-11
    tau_3 = 0.0
    tau_4 = 0.0
    alpha_1 = 0.12
    alpha_2 = 0.48
    alpha_3 = 0.00
    alpha_4 = 0.00
    
    epsr_Tgrad = 1.00
    sigma_Tgrad = 3.05
    

class SSL24_30G:
    filename = 'SSL24-30G'
    fmin = 24e9
    fmax = 30e9

    epsr_inf = 3.00
    sigma0 = 0.01
    depsr_1 = 18.75
    depsr_2 = 1.00
    depsr_3 = 0.00
    depsr_4 = 0.00
    tau_1 = 1.25e-10
    tau_2 = 1.90e-12
    tau_3 = 0.0
    tau_4 = 0.0
    alpha_1 = 0.24
    alpha_2 = 0.36
    alpha_3 = 0.00
    alpha_4 = 0.00 
    
    epsr_Tgrad = 0.88
    sigma_Tgrad = 2.37
    
    
class SSL24_30GV2:
    filename = 'SSL24-30GV2'
    fmin = 24e9
    fmax = 30e9

    epsr_inf = 3.00
    sigma0 = 0.00
    depsr_1 = 20.50
    depsr_2 = 3.09
    depsr_3 = 0.00
    depsr_4 = 0.00
    tau_1 = 1.54e-10
    tau_2 = 1.40e-11
    tau_3 = 0.0
    tau_4 = 0.0
    alpha_1 = 0.16
    alpha_2 = 0.51
    alpha_3 = 0.00
    alpha_4 = 0.00 
    
    epsr_Tgrad = 0.88
    sigma_Tgrad = 2.37
    
    
class SSL30_45G:
    filename = 'SSL30-45G'
    fmin = 30e9
    fmax = 45e9

    epsr_inf = 3.13
    sigma0 = 1.81e-6
    depsr_1 = 50.05
    depsr_2 = 4.47
    depsr_3 = 0.00
    depsr_4 = 0.00
    tau_1 = 3.73e-10
    tau_2 = 1.82e-11
    tau_3 = 0.0
    tau_4 = 0.0
    alpha_1 = 0.17
    alpha_2 = 0.62
    alpha_3 = 0.00
    alpha_4 = 0.00
    
    epsr_Tgrad = 0.85
    sigma_Tgrad = 3.13
    
 
class PHA10_18G:
    name = 'PHA10-18G'
    ssl = SSL10_18G()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4,
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.09
    tan_d_foam = 2.68e-3
    sigma_foam = tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 16.3
    tan_d_shell = 0.002
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 1.95e-3
    epoxy_thickness = 0.06e-3
    shell_thickness = 1.63e-3
    
    fmin = ssl.fmin * 1e-9
    fmax = ssl.fmax * 1e-9
    

class PHA18_24G:
    name = 'PHA18-24G'
    ssl = SSL18_24G()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4, 
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.09
    tan_d_foam = 2.68e-3
    sigma_foam = tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 16.3
    tan_d_shell = 0.002
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.04e-3
    epoxy_thickness = 0.06e-3
    shell_thickness = 1.15e-3
    
    fmin = ssl.fmin * 1e-9
    fmax = ssl.fmax * 1e-9
    
class PHA24_30G:
    name = 'PHA24-30G'
    ssl = SSL24_30G()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4, 
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.09
    tan_d_foam = 2.68e-3
    sigma_foam = tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 12.
    tan_d_shell = 0.002
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.0e-3
    epoxy_thickness = 0.
    shell_thickness = 1.08e-3
    
    fmin = ssl.fmin * 1e-9
    fmax = ssl.fmax * 1e-9
    
class PHA24_30G_V2:
    name = 'PHA24-30G-V2'
    ssl = SSL24_30GV2()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4, 
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)
    fmin = ssl.fmin * 1e-9
    fmax = ssl.fmax * 1e-9

    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.09
    tan_d_foam = 2.68e-3
    sigma_foam = tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 12.
    tan_d_shell = 0.002
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.0e-3
    epoxy_thickness = 0.06e-3
    shell_thickness = 1.0e-3
    
    
    
class PHA30_45G:
    name = 'PHA30-45G'
    ssl = SSL30_45G()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4, 
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.09
    tan_d_foam = 2.68e-3
    sigma_foam = tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 12.
    tan_d_shell = 0.002
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.0e-3
    epoxy_thickness = 0.06e-3
    shell_thickness = 0.7e-3
    
    fmin = ssl.fmin * 1e-9
    fmax = ssl.fmax * 1e-9


class PHAmmW45_70G:
    name = 'PHAmmW45_70G'
    '''ssl = SSL30_45G()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4, 
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)'''
    
    freq = np.arange(45e9, 60e9, 0.25e9)
    epsr_ssl = np.ones((len(freq)))
    sigma_ssl = np.zeros((len(freq)))
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.
    tan_d_foam = 2.68e-3
    sigma_foam = 0.#tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 4.
    tan_d_shell = 0.03 #0.01789
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.0e-3
    epoxy_thickness = 0.0
    shell_thickness = 0.75e-3
    
    fmin = np.amin(freq) * 1e-9
    fmax = np.amax(freq) * 1e-9    
'''
60 to 85 GHz
class PHAmmW45_70G:
    name = 'PHAmmW45_70G'
    
    freq = np.arange(45e9, 85e9, 0.25e9)
    epsr_ssl = np.ones((len(freq)))
    sigma_ssl = np.zeros((len(freq)))
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.
    tan_d_foam = 2.68e-3
    sigma_foam = 0.#tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 3.69
    tan_d_shell = 1.14e-2 #0.01789
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.0e-3
    epoxy_thickness = 0.0
    shell_thickness = 0.63e-3
    
    fmin = np.amin(freq) * 1e-9
    fmax = np.amax(freq) * 1e-9
'''
    
class PHAmmW70_110G:
    name = 'PHAmmW70_110G'
    '''ssl = SSL30_45G()
    freq = np.arange(ssl.fmin, ssl.fmax+0.1, 0.25e9)
    epsc_fit = helpers.getColeCole_5term(ssl.epsr_inf, ssl.sigma0, 2*np.pi*freq,
                  ssl.depsr_1, ssl.depsr_2, ssl.depsr_3, ssl.depsr_4, 
                  ssl.tau_1, ssl.tau_2, ssl.tau_3, ssl.tau_4, 
                  ssl.alpha_1, ssl.alpha_2, ssl.alpha_3, ssl.alpha_4)
    epsr_ssl, sigma_ssl = helpers.getEpsrSigma(epsc_fit, 2*np.pi*freq)'''
    
    freq = np.arange(85e9, 110.1e9, 0.25e9)
    epsr_ssl = np.ones((len(freq)))
    sigma_ssl = np.zeros((len(freq)))
    
    delta = helpers.penetration_depth(freq, epsr_ssl, sigma_ssl)
    
    epsr_foam = 1.
    tan_d_foam = 2.68e-3
    sigma_foam = 0.#tan_d_foam * 2 * np.pi * freq * epsilon_0 * epsr_foam
    
    
    epsr_epoxy = 3.2
    tan_d_epoxy = 0.01
    sigma_epoxy = tan_d_epoxy * 2 * np.pi * freq * epsilon_0 * epsr_epoxy
    
    epsr_shell = 2.94
    tan_d_shell = 2.98e-2 #0.01789
    sigma_shell = tan_d_shell * 2 * np.pi * freq * epsilon_0 * epsr_shell
    
    foam_thickness = 2.0e-3
    epoxy_thickness = 0.0
    shell_thickness = 0.54e-3
    
    fmin = np.amin(freq) * 1e-9
    fmax = np.amax(freq) * 1e-9
 