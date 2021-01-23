#!/usr/bin/env python

"""plot_lib.py: plot_lib is dedicated to plotting utility functions."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
plt.style.use("config/alt.mplstyle")
import pandas as pd
import matplotlib.ticker as mticker

import utils

COLOR = {"ELECTRON":"darkred", "ION":"darkgreen", "N_ION":"darkblue", "C_ION":"gold",
            "AH-SN":"red", "AH-AV":"blue", "AH-ME":"green", "SW-ME":"darkblue",
            "ME":"r", "AV":"green", "SN":"blue"}

def _exponent_(x,d):
    return r"%.1f$\times$ $10^{%d}$"%(x*10**d, d*(-1))

def plot_rio_locations():
    """ Plot all riometers used for the study """
    fig = plt.figure(figsize=(6,6),dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.coastlines("110m")
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor="black")
    ax.set_global()
    g = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
    g.top_labels = False
    g.right_labels = False
    g.ylocator = mticker.FixedLocator(range(30,80,10))
    ax.plot([-135.3, -135.3], [0, 90], color="red", linewidth=0.7, linestyle="--", transform=ccrs.PlateCarree())
    ax.set_extent([-150, -60, 35, 75], ccrs.PlateCarree())
    R = pd.read_csv("config/riometers.csv")
    for _, x in R.iterrows():
        ax.scatter(x["lon"], x["lat"], s=5, marker="o", color="k", zorder=2, transform=ccrs.PlateCarree())
        ax.scatter(x["lon"], x["lat"], s=30, marker="o", color="darkgreen", zorder=2, transform=ccrs.PlateCarree(), alpha=0.5)
        ax.text(x["lon"]-2., x["lat"]-2., x["rio"].upper(), fontdict={"color":"r","size":9}, transform=ccrs.PlateCarree())
    fig.savefig("figs/fovs.png",bbox_inches="tight")
    return

def plot_parameters(h, lam, b, B, gm, ad, adc, ai):
    """ Plot parameters of the simulation """
    matplotlib.rcParams["xtick.labelsize"] = 15
    matplotlib.rcParams["ytick.labelsize"] = 15
    matplotlib.rcParams["mathtext.default"] = "default"
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 15}
    fonttext = {"family": "serif", "color":  "r", "weight": "normal", "size": 15}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=150)
    ax.semilogx(lam, h, "r", label=r"$\lambda$", ls="--", lw=0.8)
    ax.semilogx(b, h, "k", label=r"$\beta$ ($sec^{-1}$)", ls="--", lw=0.8)
    ax.semilogx(gm, h, "darkgreen", label=r"$\gamma$ ($sec^{-1}$)", ls="--", lw=0.8)
    ax.semilogx(B, h, "b", label=r"B ($sec^{-1}$)", ls="--", lw=0.8)
    ax.set_ylim(50,120)
    ax.set_xlim(1e-4,1e2)
    ax.legend(loc=2, scatterpoints=3, ncol=2, fontsize=12, frameon=True)
    ax.text(.8, 0.7, r"$\alpha_d\sim$ %s $m^3sec^{-1}$"%_exponent_(ad,13) + "\n" + \
            r"$\alpha_d^c\sim$ %s $m^3sec^{-1}$"%_exponent_(adc,12) + "\n" + \
            r"$\alpha_i\sim$ %s $m^3sec^{-1}$"%_exponent_(ai,13),
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fonttext)
    ax.set_ylabel("Height, km", fontdict=font)
    ax.set_xlabel("Rate Coefficients / Charge Ratios", fontdict=font)
    fig.savefig("figs/params.png",bbox_inches="tight")
    return

def model_outputs(pg, _ind_):
    """ Comp. model outputs before and after flare """
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["mathtext.default"] = "default"
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}
    fig, axes = plt.subplots(figsize=(4, 6), sharey="row", nrows=3, ncols=2, dpi=150)
    fig.subplots_adjust(hspace=.5)
    for _i_, z in zip(range(2), _ind_):
        ax = axes[0, _i_]
        ax.semilogx(utils.extp(pg.alts, pg.ne[z, :], 64), pg.alts, COLOR["ELECTRON"], label=r"$n_e$", ls="--", lw=0.8)
        ax.semilogx(utils.extp(pg.alts, pg.ni[z, :], 64), pg.alts, COLOR["ION"], label=r"$n^+$", ls="--", lw=0.8)
        ax.semilogx(utils.extp(pg.alts, pg.ni_e[z, :], 64), pg.alts, COLOR["N_ION"], label=r"$n^-$", ls="--", lw=0.8)
        ax.semilogx(utils.extp(pg.alts, pg.ni_x[z, :], 64), pg.alts, COLOR["C_ION"], label=r"$n_x^{+}$", ls="--", lw=0.8)
        ax.set_ylim(60,120)
        ax.set_xlim(1e6,1e12)
        if _i_ == 0: ax.set_ylabel("Height, km", fontdict=font)
        ax.set_xlabel(r"Density, $m^{-3}$", fontdict=font)
        if _i_ == 1: ax.legend(bbox_to_anchor=(1.07, 0.8), scatterpoints=3, ncol=1, fontsize=8, frameon=True)
        ax.text(0.5, 1.05, "%s UT"%pg.dn[z].strftime("%Y-%m-%d %H:%M"), horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontdict=fonttext)
        
        ax = axes[1, _i_]
        ax.semilogx(utils.extp(pg.alts, pg._col_.nu_SN["total"][z,:], 68), pg.alts, COLOR["SN"], label=r"$\nu_{sn}$", ls="--", lw=0.8)
        ax.semilogx(utils.extp(pg.alts, pg._col_.nu_av_CC[z,:], 62), pg.alts, COLOR["AV"], label=r"$\nu_{av}^{cc}$", ls="--", lw=0.8)
        ax.semilogx(utils.extp(pg.alts, pg._col_.nu_av_MB[z,:], 62), pg.alts, COLOR["AV"], label=r"$\nu_{av}^{mb}$", ls="-.", lw=0.8)
        ax.semilogx(utils.extp(pg.alts, pg._col_.nu_FT[z,:], 62), pg.alts, COLOR["ME"], label=r"$\nu_{me}$", ls="--", lw=0.8)
        ax.set_ylim(60,120)
        ax.set_xlim(1e2,1e8)
        if _i_ == 0: ax.set_ylabel("Height, km", fontdict=font)
        ax.set_xlabel(r"$\nu$, $s^{-1}$", fontdict=font)
        if _i_ == 1: ax.legend(bbox_to_anchor=(1.07, 0.8), scatterpoints=3, ncol=1, fontsize=8, frameon=True)

        ax = axes[2, _i_]
        ax.plot(utils.extp(pg.alts, pg._abs_.AH["SN"]["O"][z,:], 68),
                pg.alts, COLOR["AH-SN"], label=r"$\beta^h_{ah}(\nu_{sn})$", ls="--", lw=0.8)
        ax.plot(utils.extp(pg.alts, pg._abs_.AH["AV_CC"]["O"][z,:], 64), pg.alts,
                COLOR["AH-AV"], label=r"$\beta^h_{ah}(\nu_{av}^{cc})$", ls="--", lw=0.8)
        ax.plot(utils.extp(pg.alts, pg._abs_.SW["FT"]["O"][z,:], 64), pg.alts,
                COLOR["SW-ME"], label=r"$\beta^h_{sw}(\nu_{me})$", ls="--", lw=0.8)
        ax.set_xlim(0,.3)
        ax.legend(loc=4, scatterpoints=3, ncol=1, fontsize=6, frameon=True)
        if _i_ == 1: ax.legend(bbox_to_anchor=(1.07, 0.9), scatterpoints=3, ncol=1, fontsize=8, frameon=True)
        ax.set_ylim(60,120)
        if _i_ == 0: ax.set_ylabel("Height, km", fontdict=font)
        ax.set_xlabel(r"$\beta^h$, dB/km", fontdict=font)
        
    fig.savefig("figs/irradiance.png",bbox_inches="tight")
    return

def event_study(ev, stn, pg, stime, etime, fname="figs/event.png"):
    """ Plot event for study """
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}
    legend_font = {"family": "serif", "weight": "normal", "size": 10}
    matplotlib.rcParams["xtick.color"] = "k"
    matplotlib.rcParams["ytick.color"] = "k"
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["mathtext.default"] = "default"

    fmt = matplotlib.dates.DateFormatter("%H:%M")
    fig, axes = plt.subplots(figsize=(7,5), sharey="row", nrows=2, ncols=1, dpi=100)

    xray = utils.read_goes(ev)
    ax = axes[0]
    ax.xaxis.set_major_formatter(fmt)
    ax.semilogy(xray.date, xray.B_AVG, color="r", label=r"SXR ($\lambda$, .1-.8 nm)", linewidth=0.8)
    ax.semilogy(xray.date, xray.A_AVG, color="b", label=r"HXR ($\lambda$, .05-.4 nm)", linewidth=0.8)
    ax.set_xlim(stime, etime)
    ax.set_yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3])
    ax.set_ylim(1e-8,1e-3)
    ax.set_xlabel("Time (UT)", fontdict=font)
    ax.set_ylabel("Solar Flux, "+r"$Wm^{-2}$", fontdict=font)
    ax.legend(loc=4, scatterpoints=2, ncol=1, prop=legend_font, frameon=True)

    time = pg.dn
    ax = axes[1]
    ax.xaxis.set_major_formatter(fmt)
    _abs_ = utils.read_riometer(ev, stn)
    if len(_abs_)>0: 
        info, warn = _abs_[_abs_.flag==1], _abs_[_abs_.flag!=1]
        ax.plot(info.date, info.absorp, "ko", alpha=0.4, markersize=0.3,label=r"$\beta_{R}$", lw=.8)
        ax.plot(warn.date, warn.absorp, "ro", alpha=0.4, markersize=0.1,label=r"$\beta_{R}$", lw=.4)
    ax.plot(time, pg.drap, "k", label=r"$\beta_{DRAP2}$", ls="--", lw=0.8)
    ax.plot(time, utils.int_absorption(pg._abs_.AH["SN"]["O"], pg.alts, extpoint=68),
            COLOR["AH-SN"], label=r"$\beta_{ah}(\nu_{sn})$", ls="--", lw=0.8)
    ax.plot(time, utils.int_absorption(pg._abs_.AH["AV_CC"]["O"], pg.alts, extpoint=64),
            COLOR["AH-AV"], label=r"$\beta_{ah}(\nu_{av}^{cc})$", ls="--", lw=0.8)
    ax.plot(time, utils.int_absorption(pg._abs_.AH["AV_MB"]["O"], pg.alts, extpoint=64),
            COLOR["AH-AV"], label=r"$\beta_{ah}(\nu_{av}^{mb})$", ls="-.", lw=0.8)
    ax.plot(time, utils.int_absorption(pg._abs_.SW["FT"]["O"], pg.alts, extpoint=64),
            COLOR["SW-ME"], label=r"$\beta_{sw}(\nu_{me})$", ls="--", lw=0.8)
    #ax.legend(loc=1, scatterpoints=2, ncol=2, prop=legend_font, frameon=True)
    ax.set_xlim(stime, etime)
    ax.set_ylim(-0.1,6)
    ax.set_xlabel("Time, UT", fontdict=font)
    ax.set_ylabel(r"Absorption ($\beta$), dB", fontdict=font)

    fig.autofmt_xdate()
    fig.savefig(fname,bbox_inches="tight")
    return
