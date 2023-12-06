#!/usr/bin/env python

"""plotlib.py: Plots generators."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
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
plt.style.use("seaborn-bright")
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import datetime as dt
import aacgmv2
from netCDF4 import Dataset, num2date
import matplotlib.dates as mdates

import os
import sys
sys.path.append("models/")

import utils
from fetch_data import Simulation, Riometer

np.set_printoptions(formatter={"float_kind": lambda x:"{:.2f}".format(x)})

def plot_rio_locations():
    """ Plot all riometers used for the study """
    fig = plt.figure(figsize=(5,9),dpi=150)
    ax = plt.axes(projection=ccrs.NearsidePerspective(central_longitude=-110, central_latitude=60))
    kwargs = {}
    kwargs["edgecolor"] = "black"
    kwargs["facecolor"] = "none"
    kwargs["alpha"] = 0.4
    kwargs["linewidth"] = 0.3
    resolution="110m"
    ax.add_feature( cartopy.feature.COASTLINE, **kwargs )
    ax.set_global()
    g = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3, alpha=0.4)
    g.top_labels = False
    g.right_labels = False
    g.ylocator = mticker.FixedLocator(range(0,90,10))
    g.xlocator = mticker.FixedLocator(range(-180,180,60))
    ax.plot(np.ones(90)*-135.3, np.arange(90), color="r", linewidth=0.7, linestyle="--", transform=ccrs.PlateCarree())
    ax.text(-135, 40., "GOES-15", fontdict={"color":"darkblue","size":5}, transform=ccrs.PlateCarree())
    R = pd.read_csv("config/riometers.csv")
    for _, x in R.iterrows():
        if x["rio"] != "sps":
            ax.scatter(x["lon"], x["lat"], s=0.2, marker="o", color="k", zorder=2, transform=ccrs.PlateCarree())
            ax.scatter(x["lon"], x["lat"], s=15, marker="o", color="darkgreen", zorder=2, transform=ccrs.PlateCarree(), alpha=0.5)
            ax.text(x["lon"]+0.5, x["lat"]+0.5, x["rio"].upper(), fontdict={"color":"darkblue","size":5}, transform=ccrs.PlateCarree())
    ax.set_extent([-145, -55, 40, 90], ccrs.PlateCarree())
    fig.savefig("figs/Figure01.png",bbox_inches="tight")
    dtime = dt.datetime(2021, 1, 1)
    R["lonx"] = np.mod( (R["lon"] + 180), 360 ) - 180
    R["mlat"], R["mlon"], _ = aacgmv2.get_aacgm_coord_arr(R.lat.tolist(), R.lonx.tolist(), [300]*len(R), dtime)
    R["rio"] = [r.upper() for r in R.rio]
    R = R.round(1)
    print(R[["rio","lat","lon","mlat","mlon"]].to_latex(index=False, label="tab:01", caption="List of riometers used in this study.",
         column_format="ccccc"))  
    return

def analysis_plot():
    def fit_mod(y, x=np.arange(10,500), lim=55):
        yf = utils.extrap1d(x[lim:], np.log10(y[lim:]), kind="linear")
        return 10**yf(x)
    rio = "ott"
    date = dt.datetime(2015,3,11,16,20)
    sim = Simulation(date, rio)
    sim.create_remote_local_dir()
    sim.get_bgc_file()
    sim.get_flare_file()
    _ncb = Dataset(sim._dir_ + "bgc.nc")
    _ncf = Dataset(sim._dir_ + "flare.nc")
    alt = np.arange(10,500)
    fig, axes = plt.subplots(figsize=(5,8),dpi=120,nrows=3,ncols=2,sharey="row")
    ax = axes[0,0]
    ax.set_ylabel("Height, km")
    ax.set_xlabel(r"Density, $m^{-3}$")
    ax.semilogx(fit_mod(_ncf.variables["ne"][10,:]), alt, color="r", ls="--", lw=1.)
    ax.semilogx(fit_mod(_ncf.variables["ni"][10,:]), np.arange(10,500), color="g", ls="--", lw=1.)
    ax.semilogx(fit_mod(_ncf.variables["ni-"][10,:]), np.arange(10,500), color="b", ls="--", lw=1.)
    ax.semilogx(fit_mod(_ncf.variables["nix"][10,:]), np.arange(10,500), color="gold", ls="--", lw=1.)
    ax.text(0.5,1.05,"2015-03-11 16:02 UT", ha="center", va="center", fontdict={"color":"b"}, transform=ax.transAxes)
    ax.text(0.7,.9,"(a.1)", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylim(60,120)
    ax.set_xlim(1e6,1e12)
    ax = axes[0,1]
    ax.set_xlabel(r"Density, $m^{-3}$")
    ax.semilogx(fit_mod(_ncf.variables["ne"][20,:], lim=60), alt, color="r", ls="--", lw=1., label=r"$n_e$")
    ax.semilogx(fit_mod(_ncf.variables["ni"][20,:], lim=60), np.arange(10,500), color="g", ls="--", lw=1., label=r"$n^+$")
    ax.semilogx(fit_mod(_ncf.variables["ni-"][20,:], lim=60), np.arange(10,500), color="b", ls="--", lw=1., label=r"$n^-$")
    ax.semilogx(fit_mod(_ncf.variables["nix"][20,:], lim=60), np.arange(10,500), color="gold", ls="--", lw=1., label=r"$n_x^+$")
    ax.set_xlim(1e6,1e12)
    ax.set_ylim(60,120)
    ax.text(0.5,1.05,"2015-03-11 16:22 UT", ha="center", va="center", fontdict={"color":"b"}, transform=ax.transAxes)
    ax.text(0.7,.9,"(a.2)", ha="center", va="center", transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(1.05, 0.8))
    
    ax = axes[1,0]
    ax.set_ylabel("Height, km")
    ax.set_xlabel(r"Collision Frequency $(\nu)$, $s^{-1}$")
    ax.semilogx(fit_mod(_ncb.variables["col.av.sn"][10,:]), alt, color="b", ls="--", lw=1.)
    ax.semilogx(_ncb.variables["col.ft"][10,:], alt, color="r", ls="--", lw=1.)
    ax.semilogx(_ncb.variables["col.av.cc"][10,:], alt, color="g", ls="--", lw=1.)
    ax.semilogx(_ncb.variables["col.av.mb"][10,:], alt, color="g", ls="-.", lw=1.)
    ax.text(0.7,.9,"(b.1)", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylim(60,120)
    ax.set_xlim(1e2,1e8)
    ax = axes[1,1]
    ax.set_xlabel(r"Collision Frequency $(\nu)$, $s^{-1}$")
    ax.semilogx(fit_mod(_ncb.variables["col.av.sn"][10,:]), alt, color="b", ls="--", lw=1., label=r"$\nu_{sn}$")
    ax.semilogx(_ncb.variables["col.ft"][20,:], alt, color="r", ls="--", lw=1., label=r"$\nu_{me}$")
    ax.semilogx(_ncb.variables["col.av.cc"][20,:], alt, color="g", ls="--", lw=1., label=r"$\nu^{av}_{cc}$")
    ax.semilogx(_ncb.variables["col.av.mb"][20,:], alt, color="g", ls="-.", lw=1., label=r"$\nu^{av}_{mb}$")
    ax.set_ylim(60,120)
    ax.set_xlim(1e2,1e8)
    ax.text(0.7,.9,"(b.2)", ha="center", va="center", transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(1.05, 0.8))
    
    ax = axes[2,0]
    ax.set_ylabel("Height, km")
    ax.set_xlabel(r"Absorption $(\beta^h)$, $db/km$")
    ax.plot(fit_mod(_ncf.variables["abs.ah.sn.o"][10,:], lim=75), alt, color="r", ls="--", lw=1.)
    ax.plot(fit_mod(_ncf.variables["abs.ah.av.cc.o"][10,:], lim=75), alt, color="g", ls="--", lw=1.)
    ax.plot(fit_mod(_ncf.variables["abs.ah.av.mb.o"][10,:], lim=75), alt, color="b", ls="--", lw=1.)
    ax.plot(fit_mod(_ncf.variables["abs.sw.ft.o"][10,:], lim=75), alt, color="k", ls="-.", lw=1.)
    ax.set_ylim(60,120)
    ax.set_xlim(0,0.015)
    ax.text(0.7,.9,"(c.1)", ha="center", va="center", transform=ax.transAxes)
    ax = axes[2,1]
    ax.set_xlabel(r"Absorption $(\beta^h)$, $db/km$")
    ax.plot(fit_mod(_ncf.variables["abs.ah.sn.o"][22,:]), alt, color="r", ls="--", lw=1., label=r"$\beta_{ah}^h(\nu_{sn})$")
    ax.plot(fit_mod(_ncf.variables["abs.ah.av.cc.o"][22,:]), alt, color="g", ls="--", lw=1., label=r"$\beta_{ah}^h(\nu^{av}_{cc})$")
    ax.plot(fit_mod(_ncf.variables["abs.ah.av.mb.o"][22,:]), alt, color="b", ls="--", lw=1., label=r"$\beta_{ah}^h(\nu^{av}_{mb})$")
    ax.plot(fit_mod(_ncf.variables["abs.sw.ft.o"][22,:]), alt, color="k", ls="-.", lw=1., label=r"$\beta_{sw}^h(\nu_{me})$")
    ax.set_ylim(60,120)
    ax.set_xlim(0,0.15)
    ax.legend(bbox_to_anchor=(1.75, 0.8))
    ax.text(0.7,.9,"(c.2)", ha="center", va="center", transform=ax.transAxes)
    
    sim.clear_local_folders()
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig("figs/Figure03.png",bbox_inches="tight")
    return

def example_event():
    fontT = {"family": "serif", "color":  "k", "weight": "normal", "size": 8}
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    def coloring_axes(ax, atype="left", col="red"):
        ax.spines[atype].set_color(col)
        ax.tick_params(axis="y", which="both", colors=col)
        ax.yaxis.label.set_color(col)
        fmt = matplotlib.dates.DateFormatter("%H%M")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        return ax

    def coloring_twaxes(ax, atype="left", col="red", twcol="k"):
        ax.spines[atype].set_color(col)
        ax.tick_params(axis="y", which="both", colors=twcol)
        ax.yaxis.label.set_color(twcol)
        fmt = matplotlib.dates.DateFormatter("%H%M")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        return ax
    
    rio = "ott"
    alts = np.arange(10,500)
    date = dt.datetime(2015,3,11,16,20)
    sim = Simulation(date, rio)
    sim.create_remote_local_dir()
    sim.get_flare_file()
    file_name = Riometer().get_riometer_file(None, date, rio)
    _ncf = Dataset(sim._dir_ + "flare.nc")
    times = num2date(_ncf.variables["time"][:], _ncf.variables["time"].units, _ncf.variables["time"].calendar,
        only_use_cftime_datetimes=False)
    times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
    times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
    riom = utils.read_riometer(date, "ott")    
    
    fig, ax = plt.subplots(figsize=(5,5),dpi=120,nrows=1,ncols=1)
    sTime,eTime = dt.datetime(2015,3,11,16,10), dt.datetime(2015,3,11,17)  
    ax = coloring_axes(ax, col="gray")
    ax.plot(riom.date, riom.absorp,color="gray",marker="o", markersize=1,ls="None")
    ax.set_ylim(1e-6,1e-3)
    font["color"] = "gray"
    ax.set_ylabel(r"Observation, dB",fontdict=font)
    font["color"] = "k"
    ax.set_xlabel("Time (UT)",fontdict=font)
    ax.grid(False, axis="y")
    ax.set_xlim(sTime,eTime)
    ax.set_ylim(-.1, 3.)
    
    ax = coloring_twaxes(ax.twinx(), col="gray")
    ax.plot(times, _ncf.variables["drap"][:], "darkred", label=r"$\beta_{DRAP2}$", ls="--", lw=0.8)
    ax.plot(times, utils.int_absorption(_ncf.variables["abs.ah.sn.o"][:], alts, extpoint=68),
            "r", label=r"$\beta_{ah}(\nu_{sn})$", ls="-", lw=1.2)
    ax.plot(times, utils.int_absorption(_ncf.variables["abs.ah.av.cc.o"][:], alts, extpoint=64),
            "g", label=r"$\beta_{ah}(\nu_{cc}^{av})$", ls="-", lw=0.8)
    ax.plot(times, utils.int_absorption(_ncf.variables["abs.ah.av.mb.o"][:], alts, extpoint=64),
            "b", label=r"$\beta_{ah}(\nu_{mb}^{av})$", ls="-", lw=1.2)
    ax.plot(times, utils.int_absorption(_ncf.variables["abs.sw.ft.o"][:], alts, extpoint=64),
                    "k", label=r"$\beta_{sw}(\nu_{me})$", ls="-", lw=1.2)
    ax.legend(loc=1, scatterpoints=2, ncol=1, fontsize=8, frameon=True)
    ax.set_ylim(-.1, 3.)
    font["color"] = "k"
    ax.set_ylabel("Modeled HF Absorption, dB",fontdict=font)
    font["color"] = "darkgreen"
    ax.text(0.5,1.05,"Station - OTT, 11 March 2015",horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes,fontdict=font)
    font["color"] = "k"
    ax.set_xlim(sTime,eTime)
    fig.autofmt_xdate(rotation=70,ha="center")
    sim.clear_local_folders()
    os.remove(file_name)
    fig.savefig("figs/Figure04.png",bbox_inches="tight")
    return

if __name__ == "__main__":
    ###############################################
    # Run one time plots for final use
    ###############################################
    #plot_rio_locations()
    analysis_plot()
    #example_event()
    pass