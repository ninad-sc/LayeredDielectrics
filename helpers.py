# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:30:24 2025

@author: chitnis
"""

import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm
import matplotlib
matplotlib.use('Qt5Agg')
plt.rcParams.update({'font.size': 24})


def getEpsrSigma(epscomplex, omega):
    epsr = np.real(epscomplex) / epsilon_0

    sigma = -np.imag(epscomplex) * omega
    return epsr, sigma


def getColeCole(epsr_inf, sigma0, omega, epsr_debye, tau, alpha):

     eps_inf = epsr_inf * epsilon_0
     
     t = get_midTerm(1, epsr_debye - epsr_inf, omega, tau, alpha)

     epscomplex = eps_inf + t - 1j * sigma0 / (omega)

     return epscomplex


def get_midTerm(a, depsr, omega, tau, alpha):
    return (a*depsr*epsilon_0)/(1 + (1j * omega * tau)**(1 - alpha))


def getColeCole_5term(epsr_inf, sigma0, omega,
                      depsr_1, depsr_2, depsr_3, depsr_4, 
                      tau_1, tau_2, tau_3, tau_4, 
                      alpha_1, alpha_2, alpha_3, alpha_4):
    
    eps_inf = epsr_inf * epsilon_0
    
    t1 = get_midTerm(1, depsr_1, omega, tau_1, alpha_1)
    t2 = get_midTerm(1, depsr_2, omega, tau_2, alpha_2)
    t3 = get_midTerm(1, depsr_3, omega, tau_3, alpha_3)
    t4 = get_midTerm(1, depsr_4, omega, tau_4, alpha_4)
    
    epscomplex = eps_inf + t1 + t2 - 1j * sigma0 / (omega) + t3 + t4
    
    return epscomplex


def penetration_depth(f, epsilon_r, sigma):
    w = 2 * np.pi * f
    epsilon_c = (epsilon_r * epsilon_0) - 1j*(sigma/w)
    delta = -1/np.imag(w*np.sqrt(mu_0*epsilon_c)) * 1e3
    return delta


def get_k0(omega, eps_r, sigma):
    return omega * np.sqrt(mu_0 * epsilon_0 * (get_epsr_complex(eps_r, sigma, omega)))


def get_kz(k0, kx):
    kz = np.array([np.sqrt(k0[i]**2 - kx[i,:]**2) for i in range(len(k0))])
    kz_re = np.real(kz)
    kz_im = np.imag(kz)
    kz_re = (-2 * (kz_re < 0).astype(int) + 1) * kz_re
    kz_im = (-2 * (kz_im > 0).astype(int) + 1) * kz_im
    return kz_re + 1j*kz_im


def get_epsr_complex(eps_r, sigma, omega):
    return eps_r - 1j * (sigma/(omega * epsilon_0))


def get_transfer_matrix_TE(k_z1, k_z2, d1):
    t1 = 1 + (k_z1/k_z2)
    t2 = 1 - (k_z1/k_z2)
    T = 0.5*np.array([[t1*np.exp(-1j*k_z1*d1), t2*np.exp(1j*k_z1*d1)],
                 [t2*np.exp(-1j*k_z1*d1), t1*np.exp(1j*k_z1*d1)]])
    return T


def get_transfer_matrix_TM(k_z1, k_z2, d1, eps_c1, eps_c2):
    t1 = np.array([1 + ((eps_c2[i]/eps_c1[i])*(k_z1[i,:]/k_z2[i,:])) for i in range(len(k_z1[:,0]))])
    t2 = np.array([1 - ((eps_c2[i]/eps_c1[i])*(k_z1[i,:]/k_z2[i,:])) for i in range(len(k_z1[:,0]))])
    T = 0.5*np.array([[t1*np.exp(-1j*k_z1*d1), t2*np.exp(1j*k_z1*d1)],
                 [t2*np.exp(-1j*k_z1*d1), t1*np.exp(1j*k_z1*d1)]])
    return T


def get_overall_T(kxn, kz_0, eps_c0, material):
    T_te_10 = get_transfer_matrix_TE(kz_0, material.kz_1, 0.)
    T_te_21 = get_transfer_matrix_TE(material.kz_1, material.kz_2, material.d_1)
    T_te_32 = get_transfer_matrix_TE(material.kz_2, material.kz_3, material.d_2)
    
    T_tm_10 = get_transfer_matrix_TM(kz_0, material.kz_1, 0., eps_c0, material.eps_c1)
    T_tm_21 = get_transfer_matrix_TM(material.kz_1, material.kz_2, material.d_1, material.eps_c1, material.eps_c2)
    T_tm_32 = get_transfer_matrix_TM(material.kz_2, material.kz_3, material.d_2, material.eps_c2, material.eps_c3)

    
    T_te = np.zeros_like(T_te_10)
    T_tm = np.zeros_like(T_te_10)

    for i in range(len(kz_0[:,0])):
        for j in range(len(kz_0[0,:])):
            T_te[:,:,i,j] = T_te_32[:,:,i,j] @ T_te_21[:,:,i,j] @ T_te_10[:,:,i,j]
            T_tm[:,:,i,j] = T_tm_32[:,:,i,j] @ T_tm_21[:,:,i,j] @ T_tm_10[:,:,i,j]
        
    return T_te, T_tm


def get_overall_T_3layer(kxn, kz_0, eps_c0, material):
    T_te_10 = get_transfer_matrix_TE(kz_0, material.kz_1, 0.)
    T_te_21 = get_transfer_matrix_TE(material.kz_1, material.kz_2, material.d_1)
    T_te_32 = get_transfer_matrix_TE(material.kz_2, material.kz_3, material.d_2)
    T_te_43 = get_transfer_matrix_TE(material.kz_3, material.kz_4, material.d_3)
    
    T_tm_10 = get_transfer_matrix_TM(kz_0, material.kz_1, 0., eps_c0, material.eps_c1)
    T_tm_21 = get_transfer_matrix_TM(material.kz_1, material.kz_2, material.d_1, material.eps_c1, material.eps_c2)
    T_tm_32 = get_transfer_matrix_TM(material.kz_2, material.kz_3, material.d_2, material.eps_c2, material.eps_c3)
    T_tm_43 = get_transfer_matrix_TM(material.kz_3, material.kz_4, material.d_3, material.eps_c3, material.eps_c4)

    
    T_te = np.zeros_like(T_te_10)
    T_tm = np.zeros_like(T_te_10)

    for i in range(len(kz_0[:,0])):
        for j in range(len(kz_0[0,:])):
            T_te[:,:,i,j] = T_te_43[:,:,i,j] @ T_te_32[:,:,i,j] @ T_te_21[:,:,i,j] @ T_te_10[:,:,i,j]
            T_tm[:,:,i,j] = T_tm_43[:,:,i,j] @ T_tm_32[:,:,i,j] @ T_tm_21[:,:,i,j] @ T_tm_10[:,:,i,j]
        
    return T_te, T_tm


def get_overall_T_4layer(kxn, kz_0, eps_c0, material):
    T_te_10 = get_transfer_matrix_TE(kz_0, material.kz_1, 0.)
    T_te_21 = get_transfer_matrix_TE(material.kz_1, material.kz_2, material.d_1)
    T_te_32 = get_transfer_matrix_TE(material.kz_2, material.kz_3, material.d_2)
    T_te_43 = get_transfer_matrix_TE(material.kz_3, material.kz_4, material.d_3)
    T_te_54 = get_transfer_matrix_TE(material.kz_4, material.kz_5, material.d_4)
    
    T_tm_10 = get_transfer_matrix_TM(kz_0, material.kz_1, 0., eps_c0, material.eps_c1)
    T_tm_21 = get_transfer_matrix_TM(material.kz_1, material.kz_2, material.d_1, material.eps_c1, material.eps_c2)
    T_tm_32 = get_transfer_matrix_TM(material.kz_2, material.kz_3, material.d_2, material.eps_c2, material.eps_c3)
    T_tm_43 = get_transfer_matrix_TM(material.kz_3, material.kz_4, material.d_3, material.eps_c3, material.eps_c4)
    T_tm_54 = get_transfer_matrix_TM(material.kz_4, material.kz_5, material.d_4, material.eps_c4, material.eps_c5)

    
    T_te = np.zeros_like(T_te_10)
    T_tm = np.zeros_like(T_te_10)

    for i in range(len(kz_0[:,0])):
        for j in range(len(kz_0[0,:])):
            T_te[:,:,i,j] = T_te_54[:,:,i,j] @ T_te_43[:,:,i,j] @ T_te_32[:,:,i,j] @ T_te_21[:,:,i,j] @ T_te_10[:,:,i,j]
            T_tm[:,:,i,j] = T_tm_54[:,:,i,j] @ T_tm_43[:,:,i,j] @ T_tm_32[:,:,i,j] @ T_tm_21[:,:,i,j] @ T_tm_10[:,:,i,j]
        
    return T_te, T_tm


def get_R_T(T):
    R = -T[1,0,:]/T[1,1,:]
    T = ((T[0,0,:]*T[1,1,:])-(T[0,1,:]*T[1,0,:]))/T[1,1,:]
    return R,T


class Skin:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    sigma_1 = 0.,
                    d_1 = 2e-3,
                    d_2 = 20e-6
                    ):
        self.epsr_1 = epsr_1
        self.sigma_1 = sigma_1
        self.eps_c2 = getColeCole(ziskin_model.epsr_inf,
                                                          ziskin_model.sigma_0,
                                                          2*np.pi*f, 
                                                          ziskin_model.epsr_colecole, 
                                                          ziskin_model.tau,
                                                          ziskin_model.alpha
                                                          )
        self.epsr_2, self.sigma_2 = getEpsrSigma(self.eps_c2, 2*np.pi*f)
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole(dermis.epsr_inf,
                                                          dermis.sigma_0,
                                                          2*np.pi*f, 
                                                          dermis.epsr_colecole, 
                                                          dermis.tau,
                                                          dermis.alpha
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        
        
class Skin_1std:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    epsr_2 = 3.,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    d_1 = 2e-3,
                    d_2 = 20e-6
                    ):
        self.epsr_1 = epsr_1
        self.sigma_1 = sigma_1
        self.eps_c2 = getColeCole(ziskin_model.epsr_inf,
                                                          ziskin_model.sigma_0,
                                                          2*np.pi*f, 
                                                          ziskin_model.epsr_colecole, 
                                                          ziskin_model.tau,
                                                          ziskin_model.alpha
                                                          )
        self.epsr_2, self.sigma_2 = getEpsrSigma(self.eps_c2, 2*np.pi*f)
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole(dermis_1std_dev.epsr_inf,
                                                          dermis_1std_dev.sigma_0,
                                                          2*np.pi*f, 
                                                          dermis_1std_dev.epsr_colecole, 
                                                          dermis_1std_dev.tau,
                                                          dermis_1std_dev.alpha
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        
        
class Skin_2std:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    epsr_2 = 3.,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    d_1 = 2e-3,
                    d_2 = 20e-6
                    ):
        self.epsr_1 = epsr_1
        self.sigma_1 = sigma_1
        self.eps_c2 = getColeCole(ziskin_model.epsr_inf,
                                                          ziskin_model.sigma_0,
                                                          2*np.pi*f, 
                                                          ziskin_model.epsr_colecole, 
                                                          ziskin_model.tau,
                                                          ziskin_model.alpha
                                                          )
        self.epsr_2, self.sigma_2 = getEpsrSigma(self.eps_c2, 2*np.pi*f)
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole(dermis_2std_dev.epsr_inf,
                                                          dermis_2std_dev.sigma_0,
                                                          2*np.pi*f, 
                                                          dermis_2std_dev.epsr_colecole, 
                                                          dermis_2std_dev.tau,
                                                          dermis_2std_dev.alpha
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)


class CoatedSilicone:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    epsr_2 = 3.5,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    d_1 = 2e-3,
                    d_2 = 60e-6
                    ):
        self.epsr_1 = epsr_1
        self.epsr_2 = epsr_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole(silicone_typeII.epsr_inf,
                                                          silicone_typeII.sigma_0,
                                                          2*np.pi*f, 
                                                          silicone_typeII.epsr_colecole, 
                                                          silicone_typeII.tau,
                                                          silicone_typeII.alpha
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        
        
class Gabriel_Skin:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    epsr_2 = 3.,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    d_1 = 2e-3,
                    d_2 = 20e-6
                    ):
        self.epsr_1 = epsr_1
        self.epsr_2 = epsr_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole_5term(gabriel_model.epsr_inf,
                                                          gabriel_model.sigma_0,
                                                          2*np.pi*f, 
                                                          gabriel_model.depsr_1, 
                                                          gabriel_model.depsr_2, 
                                                          gabriel_model.depsr_3,
                                                          gabriel_model.depsr_4, 
                                                          gabriel_model.tau_1,
                                                          gabriel_model.tau_2,
                                                          gabriel_model.tau_3,
                                                          gabriel_model.tau_4,
                                                          gabriel_model.alpha_1,
                                                          gabriel_model.alpha_2,
                                                          gabriel_model.alpha_3,
                                                          gabriel_model.alpha_4
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_k0(2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)


class Skin_Thick1:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    sigma_1 = 0.,
                    d_1 = 2e-3,
                    d_2 = 227e-6
                    ):
        self.epsr_1 = epsr_1
        self.sigma_1 = sigma_1
        self.eps_c2 = getColeCole(thick_1.epsr_inf,
                                                          thick_1.sigma_0,
                                                          2*np.pi*f, 
                                                          thick_1.epsr_colecole, 
                                                          thick_1.tau,
                                                          thick_1.alpha
                                                          )
        self.epsr_2, self.sigma_2 = getEpsrSigma(self.eps_c2, 2*np.pi*f)
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole(dermis.epsr_inf,
                                                          dermis.sigma_0,
                                                          2*np.pi*f, 
                                                          dermis.epsr_colecole, 
                                                          dermis.tau,
                                                          dermis.alpha
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)


class Skin_Thick3:
    def __init__(
                    self, 
                    f,
                    epsr_1 = 1.,
                    sigma_1 = 0.,
                    d_1 = 2e-3,
                    d_2 = 295e-6
                    ):
        self.epsr_1 = epsr_1
        self.sigma_1 = sigma_1
        self.eps_c2 = getColeCole(thick_3.epsr_inf,
                                                          thick_3.sigma_0,
                                                          2*np.pi*f, 
                                                          thick_3.epsr_colecole, 
                                                          thick_3.tau,
                                                          thick_3.alpha
                                                          )
        self.epsr_2, self.sigma_2 = getEpsrSigma(self.eps_c2, 2*np.pi*f)
        self.d_1 = d_1
        self.d_2 = d_2
        self.eps_c3 = getColeCole(dermis_2std_dev.epsr_inf,
                                                          dermis_2std_dev.sigma_0,
                                                          2*np.pi*f, 
                                                          dermis_2std_dev.epsr_colecole, 
                                                          dermis_2std_dev.tau,
                                                          dermis_2std_dev.alpha
                                                          )
        self.epsr_3, self.sigma_3 = getEpsrSigma(self.eps_c3, 2*np.pi*f)
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        
        
class Phantom:
    
    def __init__(
                    self,
                    f,
                    epsr_1 = 1.05,
                    epsr_2 = 11.9,
                    epsr_3 = 6.9,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    sigma_3 = 1.95,
                    d_1 = 2e-3,
                    d_2 = 1.08e-3
                    ):
        self.epsr_1 = epsr_1
        self.epsr_2 = epsr_2
        self.epsr_3 = epsr_3
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3
        self.d_1 = d_1
        self.d_2 = d_2
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        

class Phantom_3layer:
    
    def __init__(
                    self,
                    f,
                    epsr_1 = 1.05,
                    epsr_2 = 3.5,
                    epsr_3 = 16.,
                    epsr_4 = 6.,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    sigma_3 = 0.,
                    sigma_4 = 1.95,
                    d_1 = 2e-3,
                    d_2 = 0.4e-3,
                    d_3 = 1.08e-3
                    ):
        self.epsr_1 = epsr_1
        self.epsr_2 = epsr_2
        self.epsr_3 = epsr_3
        self.epsr_4 = epsr_4
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3
        self.sigma_4 = sigma_4
        self.d_1 = d_1
        self.d_2 = d_2
        self.d_3 = d_3
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        self.k0_4 = get_k0(omega, self.epsr_4, self.sigma_4)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        self.eps_c4 = get_epsr_complex(self.epsr_4, self.sigma_4, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        self.kz_4 = get_kz(self.k0_4, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        self.Z4 = np.sqrt(mu_0/self.eps_c4)
        
        
class Phantom_4layer:
    
    def __init__(
                    self,
                    f,
                    epsr_1 = 1.05,
                    epsr_2 = 3.5,
                    epsr_3 = 16.,
                    epsr_4 = 6.,
                    epsr_5 = 6.,
                    sigma_1 = 0.,
                    sigma_2 = 0.,
                    sigma_3 = 0.,
                    sigma_4 = 1.95,
                    sigma_5 = 1.95,
                    d_1 = 2e-3,
                    d_2 = 0.4e-3,
                    d_3 = 1.08e-3,
                    d_4 = 1.08e-3
                    ):
        self.epsr_1 = epsr_1
        self.epsr_2 = epsr_2
        self.epsr_3 = epsr_3
        self.epsr_4 = epsr_4
        self.epsr_5 = epsr_5
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3
        self.sigma_4 = sigma_4
        self.sigma_5 = sigma_5
        self.d_1 = d_1
        self.d_2 = d_2
        self.d_3 = d_3
        self.d_4 = d_4
        self.calc_eps_complex(2*np.pi*f)
        self.calc_Z(2*np.pi*f)
        
    def calc_k0(self, omega):
        self.k0_1 = get_k0(omega, self.epsr_1, self.sigma_1)
        self.k0_2 = get_k0(omega, self.epsr_2, self.sigma_2)
        self.k0_3 = get_k0(omega, self.epsr_3, self.sigma_3)
        self.k0_4 = get_k0(omega, self.epsr_4, self.sigma_4)
        self.k0_5 = get_k0(omega, self.epsr_5, self.sigma_5)
        
    def calc_eps_complex(self, omega):
        self.eps_c1 = get_epsr_complex(self.epsr_1, self.sigma_1, omega) * epsilon_0
        self.eps_c2 = get_epsr_complex(self.epsr_2, self.sigma_2, omega) * epsilon_0
        self.eps_c3 = get_epsr_complex(self.epsr_3, self.sigma_3, omega) * epsilon_0
        self.eps_c4 = get_epsr_complex(self.epsr_4, self.sigma_4, omega) * epsilon_0
        self.eps_c5 = get_epsr_complex(self.epsr_5, self.sigma_5, omega) * epsilon_0
        
    def calc_kz(self, kxn, k0_air):
        kx = np.array([kxn*k for k in k0_air])
        self.kz_1 = get_kz(self.k0_1, kx)
        self.kz_2 = get_kz(self.k0_2, kx)
        self.kz_3 = get_kz(self.k0_3, kx)
        self.kz_4 = get_kz(self.k0_4, kx)
        self.kz_5 = get_kz(self.k0_5, kx)
        
    def calc_Z(self, omega):
        self.calc_eps_complex(omega)
        self.Z1 = np.sqrt(mu_0/self.eps_c1)
        self.Z2 = np.sqrt(mu_0/self.eps_c2)
        self.Z3 = np.sqrt(mu_0/self.eps_c3)
        self.Z4 = np.sqrt(mu_0/self.eps_c4)
        self.Z5 = np.sqrt(mu_0/self.eps_c5)


class silicone_typeII():
    epsr_inf = 12.4
    epsr_colecole = 459
    sigma_0 = 0.256 # [S/m]
    tau = 86.1e-9 # [s]
    alpha = 0.55
    

class dermis():
    epsr_inf = 7.88
    epsr_colecole = 47.0
    sigma_0 = 5.19 # [S/m]
    tau = 8.35e-12 # [s]
    alpha = 0.
    

class dermis_1std_dev():
    epsr_inf = 5.06
    epsr_colecole = 37.0
    sigma_0 = 7.62 # [S/m]
    tau = 8.19e-12 # [s]
    alpha = 0.
    

class dermis_2std_dev():
    epsr_inf = 2.98
    epsr_colecole = 32.8
    sigma_0 = 6.94 # [S/m]
    tau = 8.76e-12 # [s]
    alpha = 0.
    
    
class gabriel_model():
    # Gabriel skin
    #
    epsr_inf = 4.0
    depsr_1 = 32.0
    tau_1 = 7.23e-12
    alpha_1 = 0
    depsr_2 = 1100
    tau_2 = 32.48e-9
    alpha_2 = 0.20
    depsr_3 = 0
    tau_3 = 0
    alpha_3 = 0
    depsr_4 = 0
    tau_4 = 0
    alpha_4 = 0
    sigma_0 = 0.0002
    

class ziskin_model():
    # Debye model from Ziskin paper
    #
    epsr_inf = 2.96
    epsr_colecole = 1.5 + epsr_inf
    sigma_0 = 0
    tau = 6.9e-12
    alpha = 0.
    

class thick_1():
    epsr_inf = 4.01
    epsr_colecole = 9.11
    sigma_0 = 0. # [S/m]
    tau = 1.63e-12 # [s]
    alpha = 0.
    
    
class thick_2():
    epsr_inf = 3.27
    epsr_colecole = 8.22
    sigma_0 = 0. # [S/m]
    tau = 2.04e-12 # [s]
    alpha = 0.
    
    
class thick_3():
    epsr_inf = 2.49
    epsr_colecole = 7.29
    sigma_0 = 0. # [S/m]
    tau = 2.22e-12 # [s]
    alpha = 0.

    
def get_A3_B3(A0, B0, T, kxn):
    
    out = np.zeros((len(B0[:,0]), len(B0[0,:]), 2), dtype=np.complex128)
    for i in range(len(B0[:,0])):
        for j in range(len(B0[0,:])):
            exc = np.array([A0[i], B0[i,j]])
            out[i,j,:] = T[:,:,i,j] @ exc
        
    return out[:,:,0], out[:,:,1]


# Calculate APD for TE mode
def get_APD_vector_te(A, B, kx, kz, Z):
    Ef = np.stack((
                            np.zeros_like(kx),
                            A, 
                            np.zeros_like(kx)
                         ), axis=-1)
    Eb = np.stack((
                            np.zeros_like(kx),
                            B, 
                            np.zeros_like(kx)
                         ), axis=-1)
    
    k_norm_f = np.stack((
                                kx,
                                np.zeros_like(kx),
                                kz
                            )/np.sqrt(kx**2 + kz**2), axis=-1)
    k_norm_b = np.stack((
                                kx,
                                np.zeros_like(kx),
                                -kz
                            )/np.sqrt(kx**2 + kz**2), axis=-1)
    Hf = np.array([(1/Z[i]) * np.cross(k_norm_f[i,:,:], Ef[i,:,:]) for i in range(len(kx[:,0]))])
    Hb = np.array([(1/Z[i]) * np.cross(k_norm_b[i,:,:], Eb[i,:,:]) for i in range(len(kx[:,0]))])
    E = Ef + Eb
    H = Hf + Hb
    return calc_APD(E, H, kx)


# Calculate APD for TM mode
def get_APD_vector_tm(A, B, kx, kz, Z):
    Hf = np.stack((
                            np.zeros_like(kx),
                            A, 
                            np.zeros_like(kx)
                         ), axis=-1)
    Hb = np.stack((
                            np.zeros_like(kx),
                            B, 
                            np.zeros_like(kx)
                         ), axis=-1)
    k_norm_f = np.stack((
                                kx,
                                np.zeros_like(kx),
                                kz
                            )/np.sqrt(kx**2 + kz**2), axis=-1)
    k_norm_b = np.stack((
                                kx,
                                np.zeros_like(kx),
                                -kz
                            )/np.sqrt(kx**2 + kz**2), axis=-1)
    Ef = np.array([Z[i] * np.cross(Hf[i,:,:], k_norm_f[i,:,:]) for i in range(len(kx[:,0]))])
    Eb = np.array([Z[i] * np.cross(Hb[i,:,:], k_norm_b[i,:,:]) for i in range(len(kx[:,0]))])
    E = Ef + Eb
    H = Hf + Hb
    return calc_APD(E, H, kx)


# Vector multiplication E x H*
def calc_APD(E, H, kx):
    P = np.zeros((len(kx[:,0]), len(kx[0,:]), 3), dtype=np.complex128)
    for i in range(len(kx[:,0])):
        for j in range(len(kx[0,:])):
            P[i,j,:] = np.cross(E[i,j,:], np.conjugate(H[i,j,:]))
    return np.real(P[:,:,2])


def normalise_to_IPD(r, APD, Z0):
    E_inc = 1 + r
    IPD = np.abs(E_inc)**2/np.abs(Z0)
    APD_norm = APD/IPD
    return APD_norm


def get_r(k_z1, k_z2, k_z3, d_2):
    r21 = (k_z2-k_z1)/(k_z2+k_z1)
    
    r32 = (k_z3-k_z2)/(k_z3+k_z2)
    
    return get_r_slab(r21, r32, k_z2, d_2) 


def get_r_slab(r1, r2, k1, d1):
    num = r1 + (r2*np.exp(-2*1j*k1*d1))
    denom = 1 + (r1*r2*np.exp(-2*1j*k1*d1))
    
    return num/denom
    
def get_r5(k_z1, k_z2, k_z3, k_z4, d_2, d_3):
    r21 = (k_z2-k_z1)/(k_z2+k_z1)
    
    r32 = (k_z3-k_z2)/(k_z3+k_z2)
    
    r43 = (k_z4-k_z3)/(k_z4+k_z3)
    
    r4 = r43
    r3 = get_r_slab(r32, r4, k_z3, d_3)
    r2 = get_r_slab(r21, r3, k_z2, d_2)
    
    return r2
    

def get_T2(kxn, kz_0, eps_c0, material):
    T_te_10 = get_transfer_matrix_TE(kz_0, material.kz_1, 0.)
    T_te_21 = get_transfer_matrix_TE(material.kz_1, material.kz_2, material.d_1)
    
    T_tm_10 = get_transfer_matrix_TM(kz_0, material.kz_1, 0., eps_c0, material.eps_c1)
    T_tm_21 = get_transfer_matrix_TM(material.kz_1, material.kz_2, material.d_1, material.eps_c1, material.eps_c2)

    T_te_20 = np.zeros_like(T_te_10)
    T_tm_20 = np.zeros_like(T_te_10)


    for i in range(len(kz_0[:,0])):
        for j in range(len(kz_0[0,:])):
            T_te_20[:,:,i,j] = T_te_21[:,:,i,j] @ T_te_10[:,:,i,j]
            T_tm_20[:,:,i,j] = T_tm_21[:,:,i,j] @ T_tm_10[:,:,i,j]
        
    return T_te_20, T_tm_20












