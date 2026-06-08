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
import helpers_202510 as helpers
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
    plt.savefig(f'plots/broadband/{pha.name}.png', dpi=300, transparent=True)
    
    
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
    plt.savefig(f'plots/broadband/{pha.name}.png', dpi=300, transparent=True)
    
    
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

def calc_ml_error(r1, r2):
    return -10*np.log10((1-np.abs(r2)**2)/(1-np.abs(r1)**2))


def calc_ml_loss(r):
    return -10*np.log10(1 - np.abs(r)**2)


def calc_error_rpd(apd1, apd2):
    return -10*np.log10((1-apd2)/(1-apd1))


def calc_error_apd(apd1, apd2):
    return 10*np.log10((apd2)/(apd1))


def calc_overall_error(skin, freq, phantom_params):
    pha = helpers.Phantom_4layer(freq, 
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
    pha.calc_k0(omega)
    pha.calc_kz(kxn, k0_air)
    r_te_pha, APD_te_pha, r_tm_pha, APD_tm_pha = get_APD_pha(pha)

    #print(f'Skin parameters: eps_r {skin.epsr_3:.2f} and sigma {skin.sigma_3:.2f}')

    APD_te_pha_norm = np.array([APD_te_pha[i,:]*Z_air[i] for i in range(len(Z_air))])
    APD_tm_pha_norm = np.array([APD_tm_pha[i,:]*Z_air[i] for i in range(len(Z_air))])
    APD_te_skin_norm = np.array([APD_te_skin[i,:]*Z_air[i] for i in range(len(Z_air))])
    APD_tm_skin_norm = np.array([APD_tm_skin[i,:]*Z_air[i] for i in range(len(Z_air))])

    e_rpd_te = calc_error_rpd(APD_te_skin_norm, APD_te_pha_norm)
    e_rpd_tm = calc_error_rpd(APD_tm_skin_norm, APD_tm_pha_norm)

    e_apd_te = calc_error_apd(APD_te_skin_norm, APD_te_pha_norm)
    e_apd_tm = calc_error_apd(APD_tm_skin_norm, APD_tm_pha_norm)
    
    return e_rpd_te, e_rpd_tm, APD_te_pha_norm, APD_tm_pha_norm


def get_unc(skin, freq, phantom_params_):
    _, _, apd_te_, apd_tm_ = calc_overall_error(skin, freq, phantom_params_)
    diff_te = apd_te_-apd_te
    diff_tm = apd_tm_-apd_tm
    
    e_max = 6. 
    step = 0.5
    num_levels = int((2*e_max)/step) + 1
    '''
    plot_2(kxn, freq*1e-9, diff_te, kxn, freq*1e-9, diff_tm, 
           cbar_label='$e_{RPD}$ |dB', cmap=cm.bwr, 
           vmin=-e_max, vmax=e_max,
           num_levels=num_levels,
           ncticks=7,
           filename='e_RPD'
           )
    '''
    return np.array([np.amax(np.abs(diff_te)), np.amax(np.abs(diff_tm))])


def get_pn_unc(skin, freq, phantom_params_, dev):
    unc_n = get_unc(skin, freq, [phantom_params_[index] * (1-(dev[index]/100)) for index in range(len(dev))])
    unc_p = get_unc(skin, freq, [phantom_params_[index] * (1+(dev[index]/100)) for index in range(len(dev))])
    return unc_n, unc_p


delta_T = 2 # max temperature change [°C] 

pha = phantoms.PHA24_30G_V2()

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

e_te, e_tm, apd_te, apd_tm = calc_overall_error(skin, pha.freq, phantom_params)
unc_skin_te = np.abs(np.nanmean(e_te)) #np.nanmax(np.abs(np.mean(e_te, 1)))
unc_skin_tm = np.abs(np.nanmean(e_tm)) #np.nanmax(np.abs(np.mean(e_tm, 1)))

contributor = [
                'lamination_lower_thickness',
                'foam_thickness',
                'lamination_upper_thickness',
                'shell_thickness',
                'lamination_lower_epsr',
                'foam_epsr',
                'lamination_upper_epsr',
                'shell_epsr', 
                'ssl_epsr', 
                'lamination_lower_sigma',
                'foam_sigma',
                'lamination_upper_sigma',
                'shell_sigma', 
                'ssl_sigma',
                'ssl_epsr_measurement',
                'ssl_sigma_measurement',
                'ssl_epsr_temperature',
                'ssl_sigma_temperature',
                ]

unc_n = np.zeros((len(contributor), 2))
unc_p = np.zeros((len(contributor), 2))
unc = np.zeros((len(contributor), 2))



for i, c in enumerate(contributor):
    vec_dev = np.zeros((len(phantom_params)))
    if c == 'lamination_lower_epsr':
        vec_dev[0] = 5.
    if c == 'foam_epsr':
        vec_dev[1] = 5.
    if c == 'lamination_upper_epsr':
        vec_dev[2] = 5.
    if c == 'shell_epsr':
        vec_dev[3] = 5.
    if c == 'ssl_epsr':
        vec_dev[4] = 10.
    if c == 'lamination_lower_sigma':
        vec_dev[5] = 5.
    if c == 'foam_sigma':
        vec_dev[6] = 5.
    if c == 'lamination_upper_sigma':
        vec_dev[7] = 5.
    if c == 'shell_sigma':
        vec_dev[8] = 5.
    if c == 'ssl_sigma':
        vec_dev[9] = 10.
    if c == 'lamination_lower_thickness':
        vec_dev[10] = 20.
    if c == 'foam_thickness':
        vec_dev[11] = 5.
    if c == 'lamination_upper_thickness':
        vec_dev[12] = 20.
    if c == 'shell_thickness':
        vec_dev[13] = 5.
    if c == 'ssl_epsr_measurement':
        vec_dev[4] = 3.2
    if c == 'ssl_sigma_measurement':
        vec_dev[9] = 5.2
    if c == 'ssl_epsr_temperature':
        vec_dev[4] = pha.ssl_epsr_Tgrad * delta_T
    if c == 'ssl_sigma_temperature':
        vec_dev[9] = pha.ssl_sigma_Tgrad * delta_T
    unc_n[i,:], unc_p[i,:] = get_pn_unc(skin, pha.freq, phantom_params, vec_dev)

#unc = np.maximum(unc_n, unc_p)
unc = np.vstack((np.maximum(unc_n, unc_p), np.array([unc_skin_te, unc_skin_tm])))

contributor_combined = [
                'ssl_epsr',
                'ssl_sigma',
                'ssl_epsr_measurement',
                'ssl_sigma_measurement',
                'ssl_epsr_temperature',
                'ssl_sigma_temperature',
                'composite_shell_epsr',
                'composite_shell_thickness',
                'skin_emulation',
                'composite_shell_sigma',
                ]
unc_combined = np.zeros((len(contributor_combined), 2))
for i, c in enumerate(contributor_combined):
    if c in contributor:
        index = contributor.index(c)
        unc_combined[i,:] = unc[index,:]
    else:
        if c == 'composite_shell_thickness':
            index1 = contributor.index('lamination_lower_thickness')
            index2 = contributor.index('foam_thickness')
            index3 = contributor.index('lamination_upper_thickness')
            index4 = contributor.index('shell_thickness')
            unc_combined[i,:] = np.sqrt(unc[index1,:]**2 + unc[index2,:]**2 + unc[index3,:]**2 + unc[index4,:]**2)
        if c == 'composite_shell_epsr':
            index1 = contributor.index('lamination_lower_epsr')
            index2 = contributor.index('foam_epsr')
            index3 = contributor.index('lamination_upper_epsr')
            index4 = contributor.index('shell_epsr')
            unc_combined[i,:] = np.sqrt(unc[index1,:]**2 + unc[index2,:]**2 + unc[index3,:]**2 + unc[index4,:]**2)
        if c == 'composite_shell_sigma':
            index1 = contributor.index('lamination_lower_sigma')
            index2 = contributor.index('foam_sigma')
            index3 = contributor.index('lamination_upper_sigma')
            index4 = contributor.index('shell_sigma')
            unc_combined[i,:] = np.sqrt(unc[index1,:]**2 + unc[index2,:]**2 + unc[index3,:]**2 + unc[index4,:]**2)
        if c == 'skin_emulation':
            unc_combined[i,:] = np.array([unc_skin_te, unc_skin_tm])
data = {
        'Contributor': contributor_combined,
        'TE': unc_combined[:,0],
        'TM': unc_combined[:,1],
        'Unc. |dB': np.round(np.max(unc_combined, axis=1), 3),
        }

df = pd.DataFrame(data)














