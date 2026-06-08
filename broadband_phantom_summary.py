# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:18:49 2025

@author: chitnis
"""
import numpy as np
import pandas as pd
from scipy.constants import c, epsilon_0, mu_0
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm
import matplotlib
import helpers
import phantoms
matplotlib.use('Qt5Agg')
plt.rcParams.update({'font.size': 24})


# Calculates APD on SC layer
def get_APD_skin(material):
    A0 = np.ones(len(kz_air[:,0]))
    T_te, T_tm = helpers.get_overall_T(kxn, kz_air, epsc_air, material)
    T_te_20, T_tm_20 = helpers.get_T2(kxn, kz_air, epsc_air, material)
    
    r_te, t_te = helpers.get_R_T(T_te)
    A3_te, B3_te = helpers.get_A3_B3(A0, r_te, T_te_20, kxn)
    APD_te = helpers.get_APD_vector_te(A3_te, B3_te, kx, material.kz_2, material.Z2)
    
    r_tm, t_tm = helpers.get_R_T(T_tm)
    A3_tm, B3_tm = helpers.get_A3_B3(A0/Z_air, np.array([r_tm[i,:]/Z_air[i] for i in range(len(r_tm[:,0]))]), T_tm_20, kxn)
    APD_tm = helpers.get_APD_vector_tm(A3_tm, B3_tm, kx, material.kz_2, material.Z2)
    
    return r_te, APD_te, r_tm, APD_tm


# Calculates APD on liquid layer
def get_APD_pha(material):
    A0 = np.ones(len(kz_air[:,0]))
    T_te, T_tm = helpers.get_overall_T_4layer(kxn, kz_air, epsc_air, material)
     
    r_te, t_te = helpers.get_R_T(T_te)
    A3_te, B3_te = helpers.get_A3_B3(A0, r_te, T_te, kxn)
    APD_te = helpers.get_APD_vector_te(A3_te, B3_te, kx, material.kz_5, material.Z5)

    r_tm, t_tm = helpers.get_R_T(T_tm)
    A3_tm, B3_tm = helpers.get_A3_B3(A0/Z_air, np.array([r_tm[i,:]/Z_air[i] for i in range(len(r_tm[:,0]))]), T_tm, kxn)
    APD_tm = helpers.get_APD_vector_tm(A3_tm, B3_tm, kx, material.kz_5, material.Z5)
    
    return r_te, APD_te, r_tm, APD_tm   

#pha = phantoms.PHA10_18G()
#pha = phantoms.PHA18_24G()
pha = phantoms.PHA24_30G_V2()
#pha = phantoms.PHA30_45G()
#pha = phantoms.PHAmmW45_70G()
#pha = phantoms.PHAmmW70_110G()

# %%
fmin_index = np.searchsorted(pha.freq, pha.fmin*1e9)
fmax_index = np.searchsorted(pha.freq, pha.fmax*1e9)

omega = 2 * np.pi * pha.freq

kxn = np.linspace(0, 2, num=200)
k0_air = helpers.get_k0(omega, 1., 0.)
kx = np.zeros((len(k0_air), len(kxn)), dtype=np.complex128)
for ik, k in enumerate(k0_air):
    kx[ik,:] = kxn*k
kz_air = helpers.get_kz(k0_air, kx)

epsr_0 = 1.
sigma_0 = 0.
epsc_air = helpers.get_epsr_complex(epsr_0, sigma_0, omega) * epsilon_0 
Z_air = np.abs(np.sqrt(mu_0/epsc_air))

skin = helpers.Skin_2std(pha.freq)
skin.calc_k0(omega)
skin.calc_kz(kxn, k0_air)
r_te_skin, APD_te_skin, r_tm_skin, APD_tm_skin = get_APD_skin(skin)

phantom_params = [
                    pha.epsr_epoxy, # 0
                    pha.epsr_foam, # 1
                    pha.epsr_epoxy, # 2
                    pha.epsr_shell, # 3
                    pha.epsr_ssl, # 4
                    pha.sigma_epoxy, # 5
                    pha.sigma_foam, # 6
                    pha.sigma_epoxy, # 7
                    pha.sigma_shell, # 8
                    pha.sigma_ssl, # 9
                    pha.epoxy_thickness, # 10
                    pha.foam_thickness, # 11
                    pha.epoxy_thickness, # 12
                    pha.shell_thickness # 13
                    ]

phantom = helpers.Phantom_4layer(pha.freq, 
                      phantom_params[0], 
                      phantom_params[1], 
                      phantom_params[2], 
                      phantom_params[3], 
                      phantom_params[4], 
                      phantom_params[5], 
                      phantom_params[6], 
                      phantom_params[7],
                      phantom_params[8], 
                      phantom_params[9], 
                      phantom_params[10],
                      phantom_params[11],
                      phantom_params[12],
                      phantom_params[13],
                      )
phantom.calc_k0(omega)
phantom.calc_kz(kxn, k0_air)
r_te_pha, APD_te_pha, r_tm_pha, APD_tm_pha = get_APD_pha(phantom)
#_, APD_te_pha_, _, APD_tm_pha_ = get_APD_skin(phantom)

# %%
def plot_2(x1, y1, E1, x2, y2, E2, 
           vmin=-2., vmax=2., num_levels=9, 
           title=None, 
           cbar_label=None, cmap=cm.CMRmap, ncticks=9,
           filename=None):
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(16,9),
                           )
    #fig.suptitle(title, y=0.83, fontsize=22)
    #fig.subplots_adjust(wspace=0.3)
    cs1 = plot_contourf(x1, y1, (E1), ax[0], 'TE', vmin, vmax, num_levels=num_levels, cmap=cmap)
    cs3 = plot_contourf(x2, y2, (E2), ax[1], 'TM', vmin, vmax, num_levels=num_levels, cmap=cmap)

    ax[0].set(xlabel='$k_{xn}$', ylabel ='f /GHz')
    ax[1].set(xlabel='$k_{xn}$')
 
    plt.tight_layout()
    cbar = fig.colorbar(cs3, shrink=0.65, ax=ax)
    cticks = np.linspace(vmin, vmax, ncticks)
    cbar.set_ticks(cticks)
    cbar.set_label(cbar_label, labelpad=15)
    plt.savefig(f'plots/broadband/{filename}.svg', transparent=True)
    plt.savefig(f'plots/broadband/{filename}.pdf', transparent=True)
    
    
def plot_4(x1, y1, E1, x2, y2, E2, x3, y3, E3, x4, y4, E4, vmin=0., vmax=1., title=None, cbar_label=None, cmap=cm.CMRmap, filename=None):
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(16,9),
                           )
    #fig.suptitle(title, y=0.83, fontsize=22)
    #fig.subplots_adjust(wspace=0.3)
    cs1 = plot_contourf(x1, y1, (E1), ax[0][0], 'TE phantom', vmin, vmax, cmap=cmap)
    cs3 = plot_contourf(x2, y2, (E2), ax[0][1], 'TM phantom', vmin, vmax, cmap=cmap)
    cs1 = plot_contourf(x4, y3, (E3), ax[1][0], 'TE skin', vmin, vmax, cmap=cmap)
    cs3 = plot_contourf(x4, y3, (E4), ax[1][1], 'TM skin', vmin, vmax, cmap=cmap)
    
    ax[0][0].set(ylabel ='f /GHz')
    ax[1][0].set(xlabel='$k_{xn}$', ylabel ='f /GHz')
    ax[1][1].set(xlabel='$k_{xn}$')

    plt.tight_layout()
    cbar = fig.colorbar(cs3, shrink=0.65, ax=ax)
    cticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(cticks)
    cbar.set_label(cbar_label, labelpad=15)
    plt.savefig(f'plots/broadband/{filename}.svg', transparent=True)
    
    
def plot_contourf(X, Y, E, ax, title, vmin, vmax, num_levels=51, cmap=cm.CMRmap):
    levels = np.linspace(vmin, vmax, num=num_levels)
    norm = mc.BoundaryNorm(levels, 256)
    ax.set_title(title)
    cs = ax.contourf(X, Y, E,
                    vmin=vmin, vmax=vmax, levels=levels, norm=norm,
                    cmap=cmap, extend='both')
    #ax.set_xlim(0., 2.)
    ax.set_ylim(pha.fmin, pha.fmax)
    #ax.set_aspect('equal')
    return cs


def plot_tricontourf(X, Y, E, ax, title, vmin, vmax):
    levels = np.linspace(vmin, vmax, num=51)
    norm = mc.BoundaryNorm(levels, 256)
    ax.set_title(title)
    cs = ax.tricontourf(X, Y, E,
                    vmin=vmin, vmax=vmax, levels=levels, norm=norm,
                    cmap=cm.CMRmap, extend='both')
    '''ax.set_xlim(-10., 10.)
    ax.set_ylim(-10., 10.)
    ax.set_aspect('equal')'''
    return cs

APD_te_pha_norm = np.array([APD_te_pha[i,:]*Z_air[i] for i in range(len(Z_air))])
APD_tm_pha_norm = np.array([APD_tm_pha[i,:]*Z_air[i] for i in range(len(Z_air))])
APD_te_skin_norm = np.array([APD_te_skin[i,:]*Z_air[i] for i in range(len(Z_air))])
APD_tm_skin_norm = np.array([APD_tm_skin[i,:]*Z_air[i] for i in range(len(Z_air))])

'''
plot_4(kxn, pha.freq*1e-9, APD_te_pha_norm, 
       kxn, pha.freq*1e-9, APD_tm_pha_norm,
       kxn, pha.freq*1e-9, APD_te_skin_norm, 
       kxn, pha.freq*1e-9, APD_tm_skin_norm, 
       cbar_label='APD/IPD',
       filename='APD')

plot_4(kxn, pha.freq*1e-9, np.abs(r_te_pha), 
       kxn, pha.freq*1e-9, np.abs(r_tm_pha),
       kxn, pha.freq*1e-9, np.abs(r_te_skin), 
       kxn, pha.freq*1e-9, np.abs(r_tm_skin), 
       cbar_label='|$\Gamma$|',
       filename='r')
'''

def calc_ml_error(r1, r2):
    return -10*np.log10((1-np.abs(r2)**2)/(1-np.abs(r1)**2))


def calc_ml_loss(r):
    return -10*np.log10(1 - np.abs(r)**2)


def calc_error_rpd(apd1, apd2):
    return -10*np.log10((1-apd2)/(1-apd1))


def calc_error_apd(apd1, apd2):
    return 10*np.log10((apd2)/(apd1))

e_rpd_te = calc_error_rpd(APD_te_skin_norm, APD_te_pha_norm)
e_rpd_tm = calc_error_rpd(APD_tm_skin_norm, APD_tm_pha_norm)

e_apd_te = calc_error_apd(APD_te_skin_norm, APD_te_pha_norm)
e_apd_tm = calc_error_apd(APD_tm_skin_norm, APD_tm_pha_norm)

e_max = 6. 
step = 0.5
num_levels = int((2*e_max)/step) + 1

'''plot_2(kxn, pha.freq*1e-9, e_apd_te, kxn, pha.freq*1e-9, e_apd_tm, 
       cbar_label='$e_{APD}$ |dB', cmap=cm.bwr, 
       vmin=-e_max, vmax=e_max,
       num_levels=num_levels,
       ncticks=7,
       filename='e_APD'
       )'''
plot_2(kxn, pha.freq*1e-9, e_rpd_te, kxn, pha.freq*1e-9, e_rpd_tm, 
       cbar_label='$e_{RPD}$ |dB', cmap=cm.bwr, 
       vmin=-e_max, vmax=e_max,
       num_levels=num_levels,
       ncticks=7,
       filename=f'e_RPD_{pha.name}'
       )
'''
slice_length = 10
N_slices = len(kxn) - slice_length # np.floor(len(kxn) / slice_length).astype(int)

e_windowed_te = np.zeros((len(pha.freq), N_slices))
e_windowed_tm = np.zeros((len(pha.freq), N_slices))

for i in range(len(pha.freq)):
    for j in range(N_slices):
        e_windowed_te[i,j] = np.nanmean(e_rpd_te[i,j:j + slice_length])
        e_windowed_tm[i,j] = np.nanmean(e_rpd_tm[i,j:j + slice_length])

plot_2(kxn[:N_slices], pha.freq*1e-9, e_windowed_te, kxn[:N_slices], pha.freq*1e-9, e_windowed_tm, 
       cbar_label='windowed $e_{RPD}$ |dB', cmap=cm.bwr, 
       vmin=-e_max, vmax=e_max,
       num_levels=num_levels,
       ncticks=7,
       filename='e_win_RPD'
       )
'''













