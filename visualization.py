from utils import get_binfrac_of_Z, get_FeH_from_Z
import astropy.units as u
import matplotlib.pyplot as plt
import legwork.visualisation as vis
from legwork import psd, utils
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.colors as col
from matplotlib.ticker import AutoMinorLocator
import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
from astropy import constants as const
import astropy.coordinates as coords
from astropy.time import Time

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['mathtext.default'] = 'regular'

obs_sec = 4 * u.yr.to('s')
obs_hz = 1 / obs_sec

met_arr = np.logspace(np.log10(1e-4), np.log10(0.03), 15)
met_arr = np.round(met_arr, 8)
met_arr = np.append(0.0, met_arr)

G = const.G.value
c = const.c.value  # speed of light in m s^-1
M_sol = const.M_sun.value  # sun's mass in kg
R_sol = const.R_sun.value  # sun's radius in metres
sec_Myr = u.Myr.to('s')  # seconds in a million years
m_kpc = u.kpc.to('m')  # metres in a kiloparsec
L_sol = const.L_sun.value  # solar luminosity in Watts
Z_sun = 0.02  # solar metallicity
sun = coords.get_sun(Time("2021-04-23T00:00:00", scale='utc'))
sun_g = sun.transform_to(coords.Galactocentric)
sun_yGx = sun_g.galcen_distance.to('kpc').value
sun_zGx = sun_g.z.to('kpc').value
M_astro = 7070  # FIRE star particle mass in solar masses
mag_lim = 23  # chosen bolometric magnitude limit



def plot_FIRE_F_mass(FIRE_path, met_arr, save=False):
    FIRE = pd.read_hdf(FIRE_path+'FIRE.h5')
    fig, ax = plt.subplots()
    plt.grid(lw=0.25, which='both')
    bins = np.append(met_arr[1:-1]/Z_sun, FIRE.met.max())
    bins = np.append(FIRE.met.min(), bins)
    bins = np.log10(bins)
    ax2 = ax.twinx()
    h, bins, _ = ax2.hist(np.log10(FIRE.met), bins=bins, histtype='step', lw=2, 
                          color='xkcd:tomato red', label='Latte m12i')
    ax2.set_yscale('log')
    #plt.xscale('log')
    ax2.legend(loc='lower left', bbox_to_anchor= (0.6, 1.01), ncol=4, borderaxespad=0, frameon=False, 
              fontsize=20)
    ax.scatter(np.log10(met_arr[1:]/Z_sun), get_binfrac_of_Z(met_arr[1:]), color='k', s=15, zorder=2, 
               label='COSMIC Z grid')
    met_plot = np.linspace(FIRE.met.min()*Z_sun, FIRE.met.max()*Z_sun, 10000)
    ax.plot(np.log10(met_plot/Z_sun), get_binfrac_of_Z(met_plot), color='k', label='FZ')
    ax.set_xlim(bins[1]-0.17693008, bins[-2] + 2 * 0.17693008)
    ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False, 
              fontsize=20, markerscale=3)
    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)
    ax.set_xlabel('Log$_{10}$(Z/Z$_\odot$)')
    #ax.set_ylabel('Binary Fraction f$_b$(Z)')
    ax.set_ylabel('Binary Fraction')
    ax2.set_ylabel(r'M$_{\rm{stars}}$ per Z bin (M$_\odot$)')
    #plt.savefig('PaperPlots/FIREfb.png')
    ax2.set_yticks([1e4, 1e5, 1e6, 1e7]);
    ax2.set_yticklabels(['7e7', '7e8', '7e9', '7e10']);
    if save:
        plt.savefig('SFH_vs_fb.png', dpi=250)
    else:
        plt.show(block=False)

    return


def plot_FIRE_F_NSP(FIRE_path, met_arr, save=False):
    FIRE = pd.read_hdf(FIRE_path+'FIRE.h5')
    fig, ax = plt.subplots()
    plt.grid(lw=0.25, which='both')
    bins = np.append(met_arr[1:-1]/Z_sun, FIRE.met.max())
    bins = np.append(FIRE.met.min(), bins)
    bins = np.log10(bins)
    ax2 = ax.twinx()
    h, bins, _ = ax2.hist(np.log10(FIRE.met), bins=bins, histtype='step', lw=2, 
                          color='xkcd:tomato red', label='Latte m12i')
    ax2.set_yscale('log')
    #plt.xscale('log')
    ax2.legend(loc='lower left', bbox_to_anchor= (0.6, 1.01), ncol=4, borderaxespad=0,
               frameon=False, fontsize=20)
    ax.scatter(np.log10(met_arr[1:]/Z_sun), get_binfrac_of_Z(met_arr[1:]), color='k', s=15, 
               zorder=2, label='COSMIC Z grid')
    met_plot = np.linspace(FIRE.met.min()*Z_sun, FIRE.met.max()*Z_sun, 10000)
    ax.plot(np.log10(met_plot/Z_sun), get_binfrac_of_Z(met_plot), color='k', label='FZ')
    ax.set_xlim(bins[1]-0.17693008, bins[-2] + 2 * 0.17693008)
    ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False, 
              fontsize=20, markerscale=3)
    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)
    ax.set_xlabel('Log$_{10}$(Z/Z$_\odot$)')
    ax.set_ylabel('Binary Fraction')
    ax2.set_ylabel(r'N$_{\rm{SP}}$ per Z bin')
    if save:
        plt.savefig('N_form_vs_fb.png', dpi=250)
    else:
        plt.show(block=False)
    return

def plot_FIREpos(FIRE_path, save=False):
    FIRE = pd.read_hdf(FIRE_path+'FIRE.h5')
    X = FIRE.xGx
    Y = FIRE.yGx
    Z = FIRE.zGx
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.hist2d(X, Y, norm=col.LogNorm(), bins=500);
    plt.scatter(0, sun_yGx, edgecolor='xkcd:light pink', facecolor='xkcd:bright pink', s=90, label='Sun', rasterized=True)
    cb = plt.colorbar()
    cb.ax.set_ylabel('LogNormed Density')
    plt.legend(fontsize=20, markerscale=2)
    plt.xlabel('X (kpc)')
    plt.ylabel('Y (kpc)')
    if save:
        plt.savefig('FIRE_pos.png', dpi=300)
    else:
        plt.show(block=False)
    return

def plot_formeff(effHe, effHe05, effCOHe, effCOHe05, effCO, effCO05, effONe, effONe05, save=False):
    from matplotlib.ticker import AutoMinorLocator
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    ax[0].plot(np.log10(met_arr[1:]/Z_sun), effHe*1e3, color='xkcd:tomato red',
                 drawstyle='steps-mid', lw=3, label='FZ', rasterized=True)
    ax[0].plot(np.log10(met_arr[1:]/Z_sun), effHe05*1e3, color='xkcd:tomato red',
               ls='--', drawstyle='steps-mid', lw=3, label='F50', rasterized=True)

    ax[1].plot(np.log10(met_arr[1:]/Z_sun), effCOHe*1e3, color='xkcd:blurple', 
               drawstyle='steps-mid', lw=3, label='FZ', rasterized=True)
    ax[1].plot(np.log10(met_arr[1:]/Z_sun), effCOHe05*1e3, color='xkcd:blurple', 
               ls='--', drawstyle='steps-mid', lw=3, label='F50', rasterized=True)

    ax[2].plot(np.log10(met_arr[1:]/Z_sun), effCO*1e3, color='xkcd:pink', 
               drawstyle='steps-mid', lw=3, label='FZ', rasterized=True)
    ax[2].plot(np.log10(met_arr[1:]/Z_sun), effCO05*1e3, color='xkcd:pink', ls='--', 
               drawstyle='steps-mid', lw=3, label='F50', rasterized=True)

    ax[3].plot(np.log10(met_arr[1:]/Z_sun), effONe*1e3, color='xkcd:light blue', 
               drawstyle='steps-mid', lw=3, label='FZ', rasterized=True)
    ax[3].plot(np.log10(met_arr[1:]/Z_sun), effONe05*1e3, color='xkcd:light blue', ls='--', 
               drawstyle='steps-mid', lw=3, label='F50', rasterized=True)

    ax[0].set_ylabel(r'$\eta_{\rm{form}}$ [10$^{-3}$ M$_\odot^{-1}$]', fontsize=18)


    labels = ['He + He', "CO + He", 'CO + CO', "ONe + X"]
    for i in range(4):
        ax[i].set_xticks([-2, -1.5, -1, -0.5, 0.])
        ax[i].text(0.05, 0.05, labels[i], fontsize=18, transform=ax[i].transAxes)
        ax[i].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, borderaxespad=0, 
                   frameon=False, fontsize=15)
        ax[i].set_xlabel('Log$_{10}$(Z/Z$_\odot$)', fontsize=18)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(labelsize=15)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    #ax[0].set_yticks(np.arange(0.25, 2.75, 0.5))
    #ax[1].set_yticks(np.arange(0.75, 4.0, 0.75))
    #ax[1].set_ylim(top=3.85)
    #ax[2].set_yticks(np.arange(1,7,1.25))
    #ax[3].set_yticks(np.arange(0.1, 0.5, 0.1))
    #ax[3].set_yticklabels(['0.10', '0.20', '0.30', '0.40'])
    if save:
        plt.savefig('form_eff.png', dpi=250)
    else:
        plt.show(block=False)    
        
    return()

def make_numLISAplot(numsFZ, numsF50, FIREmin=0.00015, FIREmax=13.346, Z_sun=0.02, save=False):
    num = 30
    met_bins = np.logspace(np.log10(FIREmin), np.log10(FIREmax), num)*Z_sun

    Henums = numsFZ.He.values
    COHenums = numsFZ.COHe.values
    COnums = numsFZ.CO.values
    ONenums = numsFZ.ONe.values

    Henums05 = numsF50.He.values
    COHenums05 = numsF50.COHe.values
    COnums05 = numsF50.CO.values
    ONenums05 = numsF50.ONe.values

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))

    ax[0].plot(np.log10(met_bins[1:]/Z_sun), Henums/1e5, drawstyle='steps-mid', 
               color='xkcd:tomato red', lw=3, label='FZ', rasterized=True)
    ax[0].plot(np.log10(met_bins[1:]/Z_sun), Henums05/1e5, 
               drawstyle='steps-mid', color='xkcd:tomato red', ls='--', lw=3, label='F50', rasterized=True)
    ax[0].text(0.05, 0.85, 'He + He', fontsize=18, transform=ax[0].transAxes)

    ax[1].plot(np.log10(met_bins[1:]/Z_sun), COHenums/1e5, drawstyle='steps-mid', 
               color='xkcd:blurple', lw=3, label='FZ', rasterized=True)
    ax[1].plot(np.log10(met_bins[1:]/Z_sun), COHenums05/1e5, drawstyle='steps-mid', 
               color='xkcd:blurple', ls='--', lw=3, label='F50', rasterized=True)
    ax[1].text(0.05, 0.85, 'CO + He', fontsize=18, transform=ax[1].transAxes)

    ax[2].plot(np.log10(met_bins[1:]/Z_sun), COnums/1e5, drawstyle='steps-mid', 
               color='xkcd:pink', lw=3, label='FZ', rasterized=True)
    ax[2].plot(np.log10(met_bins[1:]/Z_sun), COnums05/1e5, drawstyle='steps-mid', 
               color='xkcd:pink', ls='--', lw=3, label='F50', rasterized=True)
    ax[2].text(0.05, 0.85, 'CO + CO', fontsize=18, transform=ax[2].transAxes)

    ax[3].plot(np.log10(met_bins[1:]/Z_sun), ONenums/1e5, drawstyle='steps-mid', 
               color='xkcd:light blue', lw=3, label='FZ', rasterized=True)
    ax[3].plot(np.log10(met_bins[1:]/Z_sun), ONenums05/1e5, drawstyle='steps-mid',
               color='xkcd:light blue', ls='--', lw=3, label='F50', rasterized=True)
    ax[3].text(0.05, 0.85, 'ONe + X', fontsize=18, transform=ax[3].transAxes)

    for i in range(4):
        #ax[i].set_yscale('log')
        #ax[i].set_ylim(10, 2.5e6)
        #ax[i].grid(which='both', zorder=0, alpha=0.2)
        ax[i].set_xlabel('Log$_{10}$(Z/Z$_\odot$)', fontsize=18)
        ax[i].set_xticks([-3, -2, -1, 0, 1.])
        ax[i].legend(loc='lower left', bbox_to_anchor= (-0.02, 1.01), ncol=2, 
                     borderaxespad=0, frameon=False, fontsize=15)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(labelsize=15)

    ax[0].set_ylabel(r'N$_{f_{\rm{GW}} \geq 10^{-4} \rm{Hz}}$ (Z) [10$^5$]', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    #ax[0].set_yticks(np.arange(0.0, 2.5, 0.5))
    #ax[0].set_ylim(top=2.05)
    ##ax[0].set_ylim(0.09, 0.505)
    ##ax[0].set_yticklabels(['0.10', '0.20', '0.30', '0.40', '0.50'])
    #ax[1].set_yticks(np.arange(0, 20, 4))
    #ax[1].set_ylim(top=18)
    #ax[1].set_yticklabels(np.arange(0, 20, 4).astype(float).astype(str))
    ##ax[2].set_yticks(np.arange(0.0, 3.5, 0.5))
    ##ax[3].set_yticks(np.arange(1.0, 3.5, 0.5))
    ##ax[3].set_yticklabels(['1.00', '1.50', '2.00', '2.50', '3.00'])
    #ax[3].set_ylim(top=0.81)

    if save:
        plt.savefig('N_LISA_vs_met.png', dpi=250)
    else:
        plt.show(block=False)
    
    return 

def make_Mc_fgw_plot(pathtodat, model):
    resolved_dat = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format(model), key='resolved')
    
    resolved_dat = resolved_dat.loc[resolved_dat.resolved_chirp == 1.0]
    
    Heplot = resolved_dat.loc[(resolved_dat.kstar_1 == 10) & (resolved_dat.kstar_2 == 10)]
    COHeplot = resolved_dat.loc[(resolved_dat.kstar_1 == 11) & (resolved_dat.kstar_2 == 10)]
    COplot = resolved_dat.loc[(resolved_dat.kstar_1 == 11) & (resolved_dat.kstar_2 == 11)]
    ONeplot = resolved_dat.loc[(resolved_dat.kstar_1 == 12) & (resolved_dat.kstar_2.isin([10,11,12]))]
    print(len(Heplot), len(COHeplot), len(COplot), len(ONeplot))

    print(np.histogram(Heplot.met*Z_sun, met_arr))
    print(np.histogram(COHeplot.met*Z_sun, met_arr))
    print(np.histogram(COplot.met*Z_sun, met_arr))
    print(np.histogram(ONeplot.met*Z_sun, met_arr))
    fig, ax = plt.subplots(4, 3, figsize=(20,16))
    levels = [0.01, 0.1, 0.3, 0.6, 0.9]
    colors = ['#80afd6', '#2b5d87', '#4288c2', '#17334a']

    ax[0,0].scatter(y=utils.chirp_mass(Heplot.mass_1.values*u.M_sun, 
                                  Heplot.mass_2.values*u.M_sun),
               x=np.log10(Heplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(Heplot.loc[Heplot.met*Z_sun<=met_arr[1]].mass_1.values*u.M_sun, 
                                  Heplot.loc[Heplot.met*Z_sun<=met_arr[1]].mass_2.values*u.M_sun),
               x=np.log10(Heplot.loc[Heplot.met*Z_sun<=met_arr[1]].f_gw.values), levels=levels,fill=False, 
               ax=ax[0,0], color=colors[0], zorder=3, linewidths=2.5)

    ax[0,1].scatter(y=utils.chirp_mass(Heplot.mass_1.values*u.M_sun, 
                                  Heplot.mass_2.values*u.M_sun),
               x=np.log10(Heplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(Heplot.loc[(Heplot.met*Z_sun>=met_arr[7])&(Heplot.met*Z_sun<=met_arr[8])].mass_1.values*u.M_sun, 
                                  Heplot.loc[(Heplot.met*Z_sun>=met_arr[7])&(Heplot.met*Z_sun<=met_arr[8])].mass_2.values*u.M_sun),
               x=np.log10(Heplot.loc[(Heplot.met*Z_sun>=met_arr[7])&(Heplot.met*Z_sun<=met_arr[8])].f_gw.values), levels=levels,fill=False, 
               ax=ax[0,1], color=colors[1], zorder=3, linewidths=2.5)

    ax[0,2].scatter(y=utils.chirp_mass(Heplot.mass_1.values*u.M_sun, 
                                  Heplot.mass_2.values*u.M_sun),
               x=np.log10(Heplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(Heplot.loc[(Heplot.met*Z_sun>=met_arr[-2])].mass_1.values*u.M_sun, 
                                  Heplot.loc[(Heplot.met*Z_sun>=met_arr[-2])].mass_2.values*u.M_sun),
               x=np.log10(Heplot.loc[(Heplot.met*Z_sun>=met_arr[-2])].f_gw.values), levels=levels,fill=False, 
               ax=ax[0,2], color=colors[3], zorder=3, linewidths=2.5)

    ax[1,0].scatter(y=utils.chirp_mass(COHeplot.mass_1.values*u.M_sun, 
                                  COHeplot.mass_2.values*u.M_sun),
               x=np.log10(COHeplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(COHeplot.loc[COHeplot.met*Z_sun<=met_arr[1]].mass_1.values*u.M_sun, 
                                  COHeplot.loc[COHeplot.met*Z_sun<=met_arr[1]].mass_2.values*u.M_sun),
               x=np.log10(COHeplot.loc[COHeplot.met*Z_sun<=met_arr[1]].f_gw.values), levels=levels,fill=False, 
               ax=ax[1,0], color=colors[0], zorder=3, linewidths=2.5)

    ax[1,1].scatter(y=utils.chirp_mass(COHeplot.mass_1.values*u.M_sun, 
                                  COHeplot.mass_2.values*u.M_sun),
               x=np.log10(COHeplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(COHeplot.loc[(COHeplot.met*Z_sun>=met_arr[7])&(COHeplot.met*Z_sun<=met_arr[8])].mass_1.values*u.M_sun, 
                                  COHeplot.loc[(COHeplot.met*Z_sun>=met_arr[7])&(COHeplot.met*Z_sun<=met_arr[8])].mass_2.values*u.M_sun),
               x=np.log10(COHeplot.loc[(COHeplot.met*Z_sun>=met_arr[7])&(COHeplot.met*Z_sun<=met_arr[8])].f_gw.values), 
               levels=levels,fill=False, ax=ax[1,1], color=colors[1], zorder=3, linewidths=2.5)

    ax[1,2].scatter(y=utils.chirp_mass(COHeplot.mass_1.values*u.M_sun, 
                                  COHeplot.mass_2.values*u.M_sun),
               x=np.log10(COHeplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(COHeplot.loc[(COHeplot.met*Z_sun>=met_arr[-2])].mass_1.values*u.M_sun, 
                                  COHeplot.loc[(COHeplot.met*Z_sun>=met_arr[-2])].mass_2.values*u.M_sun),
               x=np.log10(COHeplot.loc[(COHeplot.met*Z_sun>=met_arr[-2])].f_gw.values), levels=levels,fill=False, 
               ax=ax[1,2], color=colors[3], zorder=3, linewidths=2.5)

    ax[2,0].scatter(y=utils.chirp_mass(COplot.mass_1.values*u.M_sun, 
                                  COplot.mass_2.values*u.M_sun),
               x=np.log10(COplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(COplot.loc[COplot.met*Z_sun<=met_arr[1]].mass_1.values*u.M_sun, 
                                  COplot.loc[COplot.met*Z_sun<=met_arr[1]].mass_2.values*u.M_sun),
               x=np.log10(COplot.loc[COplot.met*Z_sun<=met_arr[1]].f_gw.values), levels=levels,fill=False, 
               ax=ax[2,0], color=colors[0], zorder=3, linewidths=2.5)

    ax[2,1].scatter(y=utils.chirp_mass(COplot.mass_1.values*u.M_sun, 
                                  COplot.mass_2.values*u.M_sun),
               x=np.log10(COplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(COplot.loc[(COplot.met*Z_sun>=met_arr[7])&(COplot.met*Z_sun<=met_arr[8])].mass_1.values*u.M_sun, 
                                  COplot.loc[(COplot.met*Z_sun>=met_arr[7])&(COplot.met*Z_sun<=met_arr[8])].mass_2.values*u.M_sun),
               x=np.log10(COplot.loc[(COplot.met*Z_sun>=met_arr[7])&(COplot.met*Z_sun<=met_arr[8])].f_gw.values), levels=levels,fill=False, 
               ax=ax[2,1], color=colors[1], zorder=3, linewidths=2.5)

    ax[2,2].scatter(y=utils.chirp_mass(COplot.mass_1.values*u.M_sun, 
                                  COplot.mass_2.values*u.M_sun),
               x=np.log10(COplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(COplot.loc[(COplot.met*Z_sun>=met_arr[-2])].mass_1.values*u.M_sun, 
                                  COplot.loc[(COplot.met*Z_sun>=met_arr[-2])].mass_2.values*u.M_sun),
               x=np.log10(COplot.loc[(COplot.met*Z_sun>=met_arr[-2])].f_gw.values), levels=levels,fill=False, 
               ax=ax[2,2], color=colors[3], zorder=3, linewidths=2.5)


    ax[3,0].scatter(y=utils.chirp_mass(ONeplot.mass_1.values*u.M_sun, 
                                  ONeplot.mass_2.values*u.M_sun),
               x=np.log10(ONeplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(ONeplot.loc[ONeplot.met*Z_sun<=met_arr[1]].mass_1.values*u.M_sun, 
                                  ONeplot.loc[ONeplot.met*Z_sun<=met_arr[1]].mass_2.values*u.M_sun),
               x=np.log10(ONeplot.loc[ONeplot.met*Z_sun<=met_arr[1]].f_gw.values), levels=levels,fill=False, 
               ax=ax[3,0], color=colors[0], zorder=3, linewidths=2.5)


    ax[3,1].scatter(y=utils.chirp_mass(ONeplot.mass_1.values*u.M_sun, 
                                  ONeplot.mass_2.values*u.M_sun),
               x=np.log10(ONeplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(ONeplot.loc[(ONeplot.met*Z_sun>=met_arr[7])&(ONeplot.met*Z_sun<=met_arr[8])].mass_1.values*u.M_sun, 
                                  ONeplot.loc[(ONeplot.met*Z_sun>=met_arr[7])&(ONeplot.met*Z_sun<=met_arr[8])].mass_2.values*u.M_sun),
               x=np.log10(ONeplot.loc[(ONeplot.met*Z_sun>=met_arr[7])&(ONeplot.met*Z_sun<=met_arr[8])].f_gw.values), levels=levels,fill=False, 
               ax=ax[3,1], color=colors[1], zorder=3, linewidths=2.5)

    ax[3,2].scatter(y=utils.chirp_mass(ONeplot.mass_1.values*u.M_sun, 
                                  ONeplot.mass_2.values*u.M_sun),
               x=np.log10(ONeplot.f_gw.values), color='xkcd:light grey', zorder=0.)

    sns.kdeplot(y=utils.chirp_mass(ONeplot.loc[(ONeplot.met*Z_sun>=met_arr[-2])].mass_1.values*u.M_sun, 
                                  ONeplot.loc[(ONeplot.met*Z_sun>=met_arr[-2])].mass_2.values*u.M_sun),
               x=np.log10(ONeplot.loc[(ONeplot.met*Z_sun>=met_arr[-2])].f_gw.values), levels=levels,fill=False, 
               ax=ax[3,2], color=colors[3], zorder=3, linewidths=2.5)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:light grey', lw=4),
                    Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[1], lw=4),
                    Line2D([0], [0], color=colors[2], lw=4)]

    ax[0,0].legend([custom_lines[0], custom_lines[1]], ['All Z', 'Z=0.0001'], loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False, 
              fontsize=18)

    ax[0,1].legend([custom_lines[0], custom_lines[3]], ['All Z', 'Z={}'.format(met_arr[8])], loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False, 
              fontsize=18)

    ax[0,2].legend([custom_lines[0], custom_lines[2]], ['All Z', 'Z=03'], loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False, 
              fontsize=18)

    for i in range(4):
        ax[i,0].set_ylabel('Chirp Mass (M$_\odot$)', fontsize=20)
        ax[i,1].set_yticklabels('')
        ax[i,2].set_yticklabels('')

    for i in range(3):
        ax[3,i].set_xlabel(r'Log$_{10}$(f$_{\rm{GW}}$/Hz)', fontsize=20)
        ax[i,0].set_xticklabels('')
        ax[i,1].set_xticklabels('')
        ax[i,2].set_xticklabels('')
        #ax[3,i].set_xticks([-4.25, -3.75, -3.25, -2.75])
        #ax[3,i].set_xticklabels(['-4.25', '-3.75', '-3.25', '-2.75'])
        #ax[0,i].set_ylim(0.175, 0.375)
        #ax[2,i].set_ylim(0.3, 1.05)
        #ax[1,i].set_ylim(0.2, 0.6)
        #ax[3,i].set_ylim(0.3, 1.1)
        ax[0,i].text(0.85, 0.85, 'He + He', fontsize=18, horizontalalignment='center', 
                     transform=ax[0,i].transAxes)
        ax[1,i].text(0.85, 0.85, 'CO + He', fontsize=18, horizontalalignment='center', 
                     transform=ax[1,i].transAxes)
        ax[2,i].text(0.85, 0.85, 'CO + CO', fontsize=18, horizontalalignment='center', 
                     transform=ax[2,i].transAxes)
        ax[3,i].text(0.85, 0.85, 'ONe + X', fontsize=18, horizontalalignment='center', 
                     transform=ax[3,i].transAxes)

    for i in range(4):
        for j in range(3):
            ax[i,j].set_xlim(-3.5, -1.25)
            ax[i,j].axvline(-4.0, color='xkcd:grey', ls='--', lw=2., zorder=1.)

    plt.subplots_adjust(hspace=0.06, wspace=0.03)

    plt.show(block=False)
    
    return

def make_Mc_dist_plot_total(pathtodat, save=False):
    resolved_dat_FZ = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('FZ'), key='resolved')
    resolved_dat_FZ = resolved_dat_FZ.loc[resolved_dat_FZ.resolved_chirp == 1.0]
    
    Heplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 10) & (resolved_dat_FZ.kstar_2 == 10)]
    COHeplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 11) & (resolved_dat_FZ.kstar_2 == 10)]
    COplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 11) & (resolved_dat_FZ.kstar_2 == 11)]
    ONeplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 12) & (resolved_dat_FZ.kstar_2.isin([10,11,12]))]
        
    resolved_dat_F50 = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('F50'), key='resolved')
    resolved_dat_F50 = resolved_dat_F50.loc[resolved_dat_F50.resolved_chirp == 1.0]
    
    Heplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 10) & (resolved_dat_F50.kstar_2 == 10)]
    COHeplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 11) & (resolved_dat_F50.kstar_2 == 10)]
    COplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 11) & (resolved_dat_F50.kstar_2 == 11)]
    ONeplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 12) & (resolved_dat_F50.kstar_2.isin([10,11,12]))]
    
    dists = [x.dist_sun for x in [Heplot, COHeplot, COplot, ONeplot]]
    dists_F50 = [x.dist_sun for x in [Heplot_F50, COHeplot_F50, COplot_F50, ONeplot_F50]]
    
    M_c = [utils.chirp_mass(x.mass_1.values*u.M_sun, x.mass_2.values*u.M_sun) for x in [Heplot, COHeplot, COplot, ONeplot]]
    M_c_F50 = [utils.chirp_mass(x.mass_1.values*u.M_sun, x.mass_2.values*u.M_sun) for x in [Heplot_F50, COHeplot_F50, COplot_F50, ONeplot_F50]]
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    label_y = [0.35, 0.49, 0.95, 1.6]
    colors = ['#add0ed', '#2b5d87', '#4288c2', '#17334a']
    labels = ['He + He', 'CO + He', 'CO + CO', 'ONe + X']
    
    for dist, Mc, dist_F50, Mc_F50, ii in zip(dists, M_c, dists_F50, M_c_F50, range(len(dists))):
        sns.kdeplot(
            x=dist.values, y=Mc, fill=False, ax=ax[ii], color=colors[0], 
            zorder=3, linewidths=2.5, label='FZ', levels=levels
        )
        sns.kdeplot(
            x=dist_F50.values, y=Mc_F50, fill=False, ax=ax[ii], color=colors[1], 
            zorder=3, linewidths=2.5, linestyles='--', label='F50', levels=levels
        )
        ax[ii].legend(loc=(0, 1.01), prop={'size':15}, ncol=2, frameon=False)
    
    ax[0].set_ylabel('Chirp Mass [M$_\odot$]', fontsize=18)
    for i, name in zip(range(4), labels):
        ax[i].set_xlabel(r'Distance [kpc]', fontsize=18)
        ax[i].text(1.8, label_y[i], name, fontsize=18, horizontalalignment='left')
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(labelsize=15)

        
    #ax[3].set_ylim(0, 1.85)
    for j in range(4):
        ax[j].set_xlim(0, 30)
    
    plt.tight_layout()
    if save:
        plt.savefig('Mc_vs_dist.png', dpi=250)
    else:
        plt.show(block=False)        
    return

def make_Mc_f_gw_plot_total(pathtodat, save=False):
    resolved_dat_FZ = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('FZ'), key='resolved')
    resolved_dat_FZ = resolved_dat_FZ.loc[resolved_dat_FZ.resolved_chirp == 1.0]
    
    Heplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 10) & (resolved_dat_FZ.kstar_2 == 10)]
    COHeplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 11) & (resolved_dat_FZ.kstar_2 == 10)]
    COplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 11) & (resolved_dat_FZ.kstar_2 == 11)]
    ONeplot = resolved_dat_FZ.loc[(resolved_dat_FZ.kstar_1 == 12) & (resolved_dat_FZ.kstar_2.isin([10,11,12]))]
        
    resolved_dat_F50 = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('F50'), key='resolved')
    resolved_dat_F50 = resolved_dat_F50.loc[resolved_dat_F50.resolved_chirp == 1.0]
    
    Heplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 10) & (resolved_dat_F50.kstar_2 == 10)]
    COHeplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 11) & (resolved_dat_F50.kstar_2 == 10)]
    COplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 11) & (resolved_dat_F50.kstar_2 == 11)]
    ONeplot_F50 = resolved_dat_F50.loc[(resolved_dat_F50.kstar_1 == 12) & (resolved_dat_F50.kstar_2.isin([10,11,12]))]
    
    f_gws = [x.f_gw for x in [Heplot, COHeplot, COplot, ONeplot]]
    f_gws_F50 = [x.f_gw for x in [Heplot_F50, COHeplot_F50, COplot_F50, ONeplot_F50]]
    
    M_c = [utils.chirp_mass(x.mass_1.values*u.M_sun, x.mass_2.values*u.M_sun) for x in [Heplot, COHeplot, COplot, ONeplot]]
    M_c_F50 = [utils.chirp_mass(x.mass_1.values*u.M_sun, x.mass_2.values*u.M_sun) for x in [Heplot_F50, COHeplot_F50, COplot_F50, ONeplot_F50]]
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    label_y = [0.34, 0.48, 0.935, 1.53]
    colors = ['#add0ed', '#2b5d87', '#4288c2', '#17334a']
    labels = ['He + He', 'CO + He', 'CO + CO', 'ONe + X']
    
    for f_gw, Mc, f_gw_F50, Mc_F50, ii in zip(f_gws, M_c, f_gws_F50, M_c_F50, range(len(f_gws))):
        sns.kdeplot(
            x=np.log10(f_gw), y=Mc, fill=False, ax=ax[ii], color=colors[0], 
            zorder=3, linewidths=2.5, label='FZ', levels=levels
        )
        sns.kdeplot(
            x=np.log10(f_gw_F50), y=Mc_F50, fill=False, ax=ax[ii], color=colors[1], 
            zorder=3, linewidths=2.5, linestyles='--', label='F50', levels=levels
        )
        ax[ii].legend(loc=(0, 1.01), prop={'size':15}, ncol=2, frameon=False)
    
    ax[0].set_ylabel('Chirp Mass [M$_\odot$]', fontsize=18)
    for i, name in zip(range(4), labels):
        ax[i].set_xlabel(r'Log$_{10}$(f$_{\rm{GW}}$/Hz)', fontsize=18)
        ax[i].text(-3.0, label_y[i], name, fontsize=18, horizontalalignment='left')
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(labelsize=15)

        
    ax[3].set_ylim(0, 1.85)
    for j in range(4):
        ax[j].set_xlim(-3.3, -1.25)
    plt.tight_layout()
    if save:
        plt.savefig('Mc_vs_fGW.png', dpi=250)
    else:
        plt.show(block=False)        
    return

def plot_intersep(Heinter, COHeinter, COinter, ONeinter, whichsep, FIREmin=0.000149, FIREmax=13.3456, save=False):
    '''
    whichsep must be either "CEsep" or "RLOFsep"
    '''
    num = 30
    met_bins = np.logspace(np.log10(FIREmin), np.log10(FIREmax), num)#*Z_sun
    met_mids = (met_bins[1:] + met_bins[:-1]) / 2


    Heavgs = []
    Hecovs = []
    COHeavgs = []
    COHecovs = []
    COavgs = []
    COcovs = []
    ONeavgs = []
    ONecovs = []
    for i in range(num-1):
        meti = met_bins[i]
        metf = met_bins[i+1]

        Hebin = Heinter.loc[(Heinter.met>=meti)&(Heinter.met<=metf)]
        if len(Hebin) != 0:
            Heavgs.append(np.mean(Hebin[whichsep].values))
            Hecovs.append(np.std(Hebin[whichsep].values))
        else:
            Heavgs.append(0.)
            Hecovs.append(0.)

        COHebin = COHeinter.loc[(COHeinter.met>=meti)&(COHeinter.met<=metf)]
        if len(COHebin) != 0:
            COHeavgs.append(np.mean(COHebin[whichsep].values))
            COHecovs.append(np.std(COHebin[whichsep].values))
        else:
            COHeavgs.append(0.)
            COHecovs.append(0.)

        CObin = COinter.loc[(COinter.met>=meti)&(COinter.met<=metf)]
        if len(CObin) != 0:
            COavgs.append(np.mean(CObin[whichsep].values))
            COcovs.append(np.std(CObin[whichsep].values))
        else:
            COavgs.append(0.)
            COcovs.append(0.)

        ONebin = ONeinter.loc[(ONeinter.met>=meti)&(ONeinter.met<=metf)]
        if len(ONebin) != 0:
            ONeavgs.append(np.mean(ONebin[whichsep].values))
            ONecovs.append(np.std(ONebin[whichsep].values))
        else:
            ONeavgs.append(0.)
            ONecovs.append(0.)
            
    Heavgs = np.array(Heavgs)
    Hecovs = np.array(Hecovs)
    COHeavgs = np.array(COHeavgs)
    COHecovs = np.array(COHecovs)
    COavgs = np.array(COavgs)
    COcovs = np.array(COcovs)
    ONeavgs = np.array(ONeavgs)
    ONecovs = np.array(ONecovs)
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    ax[0].plot(np.log10(met_mids[Heavgs>0]), Heavgs[Heavgs>0]/1e3, color='xkcd:tomato red', lw=3, ls='-', label='He + He', 
             drawstyle='steps-mid', rasterized=True)
    ax[0].fill_between(np.log10(met_mids[Heavgs>0]), (Heavgs[Heavgs>0]-Hecovs[Heavgs>0])/1e3, 
                     (Heavgs[Heavgs>0]+Hecovs[Heavgs>0])/1e3, alpha=0.3, color='xkcd:tomato red',
                       zorder=0, step='mid', label='$1\sigma$', rasterized=True)

    ax[2].plot(np.log10(met_mids[COavgs>0]), COavgs[COavgs>0]/1e3, color='xkcd:pink', lw=3, ls='-', 
             label='CO + CO', drawstyle='steps-mid', rasterized=True)
    ax[2].fill_between(np.log10(met_mids[COavgs>0]), (COavgs[COavgs>0]-COcovs[COavgs>0])/1e3, 
                     (COavgs[COavgs>0]+COcovs[COavgs>0])/1e3, alpha=0.3, color='xkcd:pink', 
                       zorder=0, step='mid', label='$1\sigma$', rasterized=True)

    ax[1].plot(np.log10(met_mids), COHeavgs/1e3, color='xkcd:blurple', lw=3, ls='-', label='CO + He', 
             drawstyle='steps-mid', rasterized=True)
    ax[1].fill_between(np.log10(met_mids[COHeavgs>0]), (COHeavgs[COHeavgs>0]-COHecovs[COHeavgs>0])/1e3, 
                     (COHeavgs[COHeavgs>0]+COHecovs[COHeavgs>0])/1e3, alpha=0.3, color='xkcd:blurple',
                       zorder=0, step='mid', label='$1\sigma$', rasterized=True)

    ax[3].plot(np.log10(met_mids[ONeavgs>0]), ONeavgs[ONeavgs>0]/1e3, color='xkcd:light blue', lw=3, 
             label='ONe + X', drawstyle='steps-mid', rasterized=True)
    ax[3].fill_between(np.log10(met_mids[ONeavgs>0]), (ONeavgs[ONeavgs>0]-ONecovs[ONeavgs>0])/1e3,
                     (ONeavgs[ONeavgs>0]+ONecovs[ONeavgs>0])/1e3, alpha=0.3, color='xkcd:light blue', 
                       zorder=0, step='mid', label='$1\sigma$', rasterized=True)

    for i in range(4):
        #ax[i].set_xscale('log')
        ax[i].set_xticks([-3., -2., -1., 0., 1.])
        ax[i].tick_params(labelsize=15)
        ax[i].set_xlim(np.log10(met_mids[0]), np.log10(met_mids[-1]))
        ax[i].set_xlabel('Log$_{10}$(Z/Z$_\odot$)', fontsize=18)
        ax[i].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False, 
              fontsize=15, markerscale=0.5)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    #ax[0].set_ylabel('Avg. Interaction\nSeparation (10$^3$ R$_\odot$)', fontsize=18)
    ax[0].set_ylabel(r'$\overline{a}_{\rm{CE}}$  [10$^3$ R$_\odot$]', fontsize=16)

    #ax[0].set_yticks(np.arange(0.1, 0.6, 0.1))
    #ax[0].set_ylim(0.09, 0.505)
    #ax[0].set_yticklabels(['0.10', '0.20', '0.30', '0.40', '0.50'])
    #ax[1].set_yticks(np.arange(0.25, 1.5, 0.25))
    ##ax[1].set_yticklabels(['0.20', '0.40', '0.60', '0.80', '1.00', '1.20'])
    #ax[2].set_yticks(np.arange(0.25, 2.75,0.5))
    #ax[3].set_yticks(np.arange(1.0, 3.5, 0.5))
    #ax[3].set_yticklabels(['1.00', '1.50', '2.00', '2.50', '3.00'])
    #ax[3].set_ylim(top=3.05)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    save=True
    if save:
        plt.savefig('a_{}.png'.format(whichsep), dpi=250)
    else:
        plt.show(block=False) 
    
    return

def plot_LISAcurves(pathtodat, model, save=False):
    from legwork.visualisation import plot_sensitivity_curve
    
    def func(x, a, b, c, d, e):
        return a + b*x + c*x**2 + d*x**3 + e*x**4
    
    def cosmic_confusion(f, L, t_obs=4 * u.yr, approximate_R=True, include_confusion_noise=False):
        lisa_psd_no_conf = psd.power_spectral_density(f, include_confusion_noise=False, t_obs=4 * u.yr)
        conf = 10**func(x=np.log10(f.value), 
                        a=popt[0], b=popt[1], 
                        c=popt[2], d=popt[3], e=popt[4]) * t_obs.to(u.s)
    
        psd_plus_conf = conf + lisa_psd_no_conf
        return psd_plus_conf.to(u.Hz**(-1))
    
    resolved = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format(model), key='resolved')
    popt = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format(model), key='conf_fit')
    popt = popt.values.flatten()
    
    resolved_HeHe = resolved.loc[(resolved.kstar_1 == 10) & (resolved.kstar_2 == 10)]
    resolved_COHe = resolved.loc[(resolved.kstar_1 == 11) & (resolved.kstar_2 == 10)]
    resolved_COCO = resolved.loc[(resolved.kstar_1 == 11) & (resolved.kstar_2 == 11)]
    resolved_ONeX = resolved.loc[(resolved.kstar_1 == 12) & (resolved.kstar_2.isin([10,11,12]))]
    
    t_obs = 4 * u.yr
    
    
    psd_conf = psd.power_spectral_density(f=np.linspace(1e-4, 1e-1, 1000000) * u.Hz, 
                                         instrument="custom", 
                                         custom_function=cosmic_confusion, 
                                         t_obs=t_obs, 
                                         L=None, 
                                         approximate_R=True, 
                                         include_confusion_noise=False)
    
    
    Heasd = ((1/4 * t_obs)**(1/2) * resolved_HeHe.h_0.values).to(u.Hz**(-1/2))
    COasd = ((1/4 * t_obs)**(1/2) * resolved_COCO.h_0.values).to(u.Hz**(-1/2))
    COHeasd = ((1/4 * t_obs)**(1/2) * resolved_COHe.h_0.values).to(u.Hz**(-1/2))
    ONeasd = ((1/4 * t_obs)**(1/2) * resolved_ONeX.h_0.values).to(u.Hz**(-1/2))

    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    ax[0].plot(np.linspace(1e-4, 1e-1, 1000000), psd_conf**0.5, c='black', rasterized=True)
    ax[0].scatter(resolved_COHe.f_gw, COHeasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[0].scatter(resolved_COCO.f_gw, COasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[0].scatter(resolved_ONeX.f_gw, ONeasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[0].scatter(resolved_HeHe.f_gw, Heasd, zorder=10, color='xkcd:tomato red', label='He + He', rasterized=True)
    ax[0].legend(loc='lower left', ncol=4, borderaxespad=0, frameon=False, 
                 fontsize=22, markerscale=2.5, handletextpad=0.15)
    ax[0].text(0.1, 3e-17, model+', SNR > 7: {}'.format(len(Heasd)), fontsize=22, 
           horizontalalignment='right')
    

    ax[2].plot(np.linspace(1e-4, 1e-1, 1000000), psd_conf**0.5, c='black', rasterized=True)
    ax[2].scatter(resolved_COHe.f_gw, COHeasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[2].scatter(resolved_ONeX.f_gw, ONeasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[2].scatter(resolved_HeHe.f_gw, Heasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[2].scatter(resolved_COCO.f_gw, COasd, zorder=10, color='xkcd:pink', label='CO + CO', rasterized=True)
    ax[2].legend(loc='lower left', ncol=4, borderaxespad=0, frameon=False, 
                 fontsize=22, markerscale=2.5, handletextpad=0.15)
    ax[2].text(0.1, 3e-17, model+', SNR > 7: {}'.format(len(COasd)), fontsize=22, 
           horizontalalignment='right')
    
    ax[1].plot(np.linspace(1e-4, 1e-1, 1000000), psd_conf**0.5, c='black', rasterized=True)
    ax[1].scatter(resolved_ONeX.f_gw, ONeasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[1].scatter(resolved_HeHe.f_gw, Heasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[1].scatter(resolved_COCO.f_gw, COasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[1].scatter(resolved_COHe.f_gw, COHeasd, zorder=10, color='xkcd:blurple', label='CO + He', rasterized=True)
    ax[1].legend(loc='lower left', ncol=4, borderaxespad=0, frameon=False, 
                 fontsize=22, markerscale=2.5, handletextpad=0.15)
    ax[1].text(0.1, 3e-17, model+', SNR > 7: {}'.format(len(COHeasd)), fontsize=22, 
           horizontalalignment='right')
    
    

    ax[3].plot(np.linspace(1e-4, 1e-1, 1000000), psd_conf**0.5, c='black', rasterized=True)
    ax[3].scatter(resolved_HeHe.f_gw, Heasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[3].scatter(resolved_COCO.f_gw, COasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[3].scatter(resolved_COHe.f_gw, COHeasd, zorder=10, color='xkcd:light grey', rasterized=True)
    ax[3].scatter(resolved_ONeX.f_gw, ONeasd, zorder=10, color='xkcd:light blue', label='ONe + X')
    ax[3].legend(loc='lower left', ncol=4, borderaxespad=0, frameon=False, 
                 fontsize=22, markerscale=2.5, handletextpad=0.15)
    ax[3].text(0.1, 3e-17, model+', SNR > 7: {}'.format(len(ONeasd)), fontsize=22, 
               horizontalalignment='right')
    
    for i in range(4):
        #ax[i].set_xlim(1e-4, 1e-1)
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
        ax[i].tick_params(labelsize=22)
        ax[i].set_xlabel(r'$f_{\rm{GW}}$ [Hz]', size=24)
    ax[0].set_ylabel(r'ASD [Hz$^{-1/2}$]', size=24)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    if save:
        plt.savefig('LISA_SNR_{}.png'.format(model), dpi=250)
    else:
        plt.show()
        
    return

def plot_foreground(pathtodat, save=False):
    def func(x, a, b, c, d, e):
        return a + b*x + c*x**2 + d*x**3 + e*x**4
    
    colors = ['#add0ed', '#2b5d87', '#4288c2', '#17334a']
    Tobs = 4 * u.yr

    power_dat_F50 = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('F50'), key='total_power')
    popt_F50 = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('F50'), key='conf_fit')
    popt_F50 = popt_F50.values.flatten()
    
    power_dat_FZ = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('FZ'), key='total_power')
    popt_FZ = pd.read_hdf(pathtodat+'resolved_DWDs_{}.hdf'.format('FZ'), key='conf_fit')
    popt_FZ = popt_FZ.values.flatten()
    
    conf_fit_FZ = 10**func(
        x=np.log10(np.linspace(1e-4, 1e-1, 100000)), 
        a=popt_FZ[0], b=popt_FZ[1], c=popt_FZ[2], d=popt_FZ[3], e=popt_FZ[4]
    )* Tobs.to(u.s).value
    
    conf_fit_F50 = 10**func(
        x=np.log10(np.linspace(1e-4, 1e-1, 100000)), 
        a=popt_F50[0], b=popt_F50[1], c=popt_F50[2], d=popt_F50[3], e=popt_F50[4]
    )* Tobs.to(u.s).value

    
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4.2))
    plt.plot(power_dat_F50.f_gw[::10], power_dat_F50.strain_2[::10] * Tobs.to(u.s).value, c=colors[1], lw=1, alpha=1, rasterized=False)#, ls='none', marker=',')
    
    plt.plot(power_dat_FZ.f_gw[::10], power_dat_FZ.strain_2[::10] * Tobs.to(u.s).value, c=colors[0], lw=1, alpha=0.8, rasterized=False)#, ls='none', marker=',')
    
    plt.plot(np.linspace(1e-4, 1e-1, 100000), conf_fit_F50, c=colors[3], ls='--', lw=2, label=r'F50')
    plt.plot(np.linspace(1e-4, 1e-1, 100000), conf_fit_FZ, c=colors[2], ls='--', lw=2, label=r'FZ')
    plt.xscale('log')
    plt.yscale('log')
    
    ax1.set_ylabel(r'PSD [Hz$^{-1}$]', size=15)
    ax1.set_xlabel(r'f$_{\rm{GW}}$ [Hz]', size=15)
    ax1.tick_params(labelsize=12)
    ax1.set_yticks([1e-38, 1e-37, 1e-36, 1e-35, 1e-34])
    plt.xlim(1e-4, 3e-2)
    plt.ylim(1e-38, 5e-34)
    plt.legend(prop={'size': 12}, ncol=2, frameon=False, loc=(0, 1))
    plt.tight_layout()
    if save:
        plt.savefig('PSD.png', facecolor='white', dpi=250)
    else:
        plt.show()
    return
    
def plot_model_var(pathtodat, save=False):
    models = ['log_uniform', 'qcflag_4', 'alpha_0.25', 'alpha_5']
    model_names = ['fiducial', r'q3', r'$\alpha25$', r'$\alpha5$']
    colors = sns.color_palette("mako", n_colors=len(models))
    
    Tobs = 4 * u.yr
    
    def func(x, a, b, c, d, e):
        return a + b*x + c*x**2 + d*x**3 + e*x**4
    
    mosaic = """
    AA
    AA
    BB
    """
    
    fig = plt.figure(figsize=(6, 8))
    ax_dict = fig.subplot_mosaic(mosaic)
    
    
    lisa_ratio = []
    n_lisa_F50_list = []
    
    popt_F50_list = []
    popt_FZ_list = []
    
    for m in models:
        path = pathtodat+m+'/plot_data/'
        n_lisa_F50 = pd.read_hdf(path+'numLISA_30bins_F50.hdf', key='data')
        n_lisa_FZ = pd.read_hdf(path+'numLISA_30bins_FZ.hdf', key='data')
        n_lisa_F50 = np.sum(n_lisa_F50.values.flatten())
        n_lisa_FZ = np.sum(n_lisa_FZ.values.flatten())
        
        lisa_ratio.append(n_lisa_FZ/n_lisa_F50)
        n_lisa_F50_list.append(n_lisa_F50)
    
        popt_F50 = pd.read_hdf(path+'resolved_DWDs_{}.hdf'.format('F50'), key='conf_fit')
        popt_F50 = popt_F50.values.flatten()
        popt_FZ = pd.read_hdf(path+'resolved_DWDs_{}.hdf'.format('FZ'), key='conf_fit')
        popt_FZ = popt_FZ.values.flatten()
                
        popt_F50_list.append(popt_F50)
        popt_FZ_list.append(popt_FZ)
    
            
    for popt_F50, popt_FZ, ii in zip(popt_F50_list, popt_FZ_list, range(len(popt_FZ_list))):
        conf_fit_FZ = 10**func(
            x=np.log10(np.linspace(1e-4, 1e-1, 100000)), 
            a=popt_FZ[0], b=popt_FZ[1], c=popt_FZ[2], d=popt_FZ[3], e=popt_FZ[4]
        )* Tobs.to(u.s).value
        
        conf_fit_F50 = 10**func(
            x=np.log10(np.linspace(1e-4, 1e-1, 100000)), 
            a=popt_F50[0], b=popt_F50[1], c=popt_F50[2], d=popt_F50[3], e=popt_F50[4]
        )* Tobs.to(u.s).value
        
        ax_dict['A'].plot(
            np.linspace(1e-4, 1e-1, 100000), conf_fit_F50, color=colors[ii], ls='--', lw=2.5, zorder=10-ii
        )
        ax_dict['A'].plot(
            np.linspace(1e-4, 1e-1, 100000), conf_fit_FZ, color=colors[ii], ls='-', lw=2.5, label=model_names[ii]
        )
    
    
    
    ax_dict['A'].set_xscale('log')
    ax_dict['A'].set_yscale('log')
    
    ax_dict['A'].set_ylabel(r'confusion fit [Hz$^{-1}$]', size=16)
    ax_dict['A'].set_xlabel(r'f$_{\rm{GW}}$ [Hz]', size=16)
    ax_dict['A'].set_xlim(1e-4, 3.5e-3)
    ax_dict['A'].set_ylim(1e-38, 7e-35)
    
    
    for ii in range(len(lisa_ratio)):
        ax_dict['B'].scatter(n_lisa_F50_list[ii], lisa_ratio[ii], color=colors[ii], marker='s', s=45, label=model_names[ii])
    ax_dict['A'].legend(prop={'size' : 12}, frameon=False, loc='upper right')
    ax_dict['B'].set_xscale('log')
    ax_dict['B'].axhline(0.5, ls='--', color='silver', lw=2, zorder=0)
    ax_dict['B'].set_ylim(0.2, 0.8) 
    ax_dict['B'].set_xlim(3e5, 1e8)
    ax_dict['B'].set_ylabel(r'N$_{\rm{LISA, FZ}}$/N$_{\rm{LISA, F50}}$', size=16)
    ax_dict['B'].set_xlabel(r'N$_{\rm{LISA, F50}}$', size=16)
    ax_dict['A'].tick_params(labelsize=12)
    ax_dict['B'].tick_params(labelsize=12)
    plt.tight_layout()
    if save:
        plt.savefig('model_comp.png', dpi=250, facecolor='white')
    else:
        plt.show()        
    return
    
            
        
def plot_model_var_2(pathtodat, save=False):
    models = ['log_uniform', 'qcflag_4', 'alpha_0.25']#, 'alpha_5']
    model_names = [r'fiducial, N$_{\rm{LISA}}$=1.24e7', r'q3, N$_{\rm{LISA}}$=4.28e7', r'$\alpha25$, N$_{\rm{LISA}}$=5.58e5']#, r'Standard, $\alpha5$']
    colors = sns.color_palette("mako", n_colors=len(models))
    
    Tobs = 4 * u.yr
    
    def func(x, a, b, c, d, e):
        return a + b*x + c*x**2 + d*x**3 + e*x**4
    
    mosaic = """
    AA
    AA
    BB
    """
    
    fig = plt.figure(figsize=(6, 8))
    ax_dict = fig.subplot_mosaic(mosaic)
    
    
    lisa_ratio = []
    n_lisa_F50_list = []
    
    popt_F50_list = []
    popt_FZ_list = []
    
    for m in models:
        path = pathtodat+m+'/plot_data/'
        n_lisa_F50 = pd.read_hdf(path+'numLISA_30bins_F50.hdf', key='data')
        n_lisa_FZ = pd.read_hdf(path+'numLISA_30bins_FZ.hdf', key='data')
        n_lisa_F50 = np.sum(n_lisa_F50.values.flatten())
        n_lisa_FZ = np.sum(n_lisa_FZ.values.flatten())
        
        lisa_ratio.append(n_lisa_FZ/n_lisa_F50)
        n_lisa_F50_list.append(n_lisa_F50)
    
        popt_F50 = pd.read_hdf(path+'resolved_DWDs_{}.hdf'.format('F50'), key='conf_fit')
        popt_F50 = popt_F50.values.flatten()
        popt_FZ = pd.read_hdf(path+'resolved_DWDs_{}.hdf'.format('FZ'), key='conf_fit')
        popt_FZ = popt_FZ.values.flatten()
                
        popt_F50_list.append(popt_F50)
        popt_FZ_list.append(popt_FZ)
    
            
    for popt_F50, popt_FZ, ii in zip(popt_F50_list, popt_FZ_list, range(len(popt_FZ_list))):
        conf_fit_FZ = 10**func(
            x=np.log10(np.linspace(1e-4, 1e-1, 100000)), 
            a=popt_FZ[0], b=popt_FZ[1], c=popt_FZ[2], d=popt_FZ[3], e=popt_FZ[4]
        )* Tobs.to(u.s).value
        
        conf_fit_F50 = 10**func(
            x=np.log10(np.linspace(1e-4, 1e-1, 100000)), 
            a=popt_F50[0], b=popt_F50[1], c=popt_F50[2], d=popt_F50[3], e=popt_F50[4]
        )* Tobs.to(u.s).value
        
        ax_dict['A'].plot(
            np.linspace(1e-4, 1e-1, 100000), conf_fit_F50, color=colors[ii], ls='--', lw=2, label=model_names[ii]
        )
        ax_dict['A'].plot(
            np.linspace(1e-4, 1e-1, 100000), conf_fit_FZ, color=colors[ii], ls='-', lw=2
        )
        
        ax_dict['B'].plot(
            np.linspace(1e-4, 1e-1, 100000), np.abs(conf_fit_FZ - conf_fit_F50)/(conf_fit_F50), color=colors[ii], ls='--', lw=2, label=model_names[ii]
        )
    
    
    ax_dict['A'].set_xscale('log')
    ax_dict['A'].set_yscale('log')
    
    ax_dict['A'].set_ylabel(r'confusion fit [Hz$^{-1}$]', size=16)
    ax_dict['A'].set_xlim(1e-4, 5e-3)
    ax_dict['A'].set_ylim(1e-38, 7e-35)
    ax_dict['A'].set_xticklabels([])
    
        
    ax_dict['A'].legend(prop={'size' : 12}, frameon=False, loc='upper right')
    ax_dict['B'].set_xscale('log')
    ax_dict['B'].set_xlim(1e-4, 5e-3)
    ax_dict['B'].set_ylabel(r'conf$_{\rm{FZ}}$/conf$_{\rm{F50}}$', size=16)
    ax_dict['B'].set_xlabel(r'$f_{\rm{GW}}$ [Hz]', size=16)
    ax_dict['A'].tick_params(labelsize=12)
    ax_dict['B'].tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('conf_ratio.png', facecolor='white', dpi=200)