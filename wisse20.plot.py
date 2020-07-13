#!/usr/bin/env python

"""wisse20.plot.py: WISSE 2020 plots"""

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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.style.use("config/alt.mplstyle")

import sys
sys.path.append("models/")
import os
import numpy as np
import argparse
import pandas as pd
import datetime as dt
from netCDF4 import Dataset, num2date
from dateutil import parser as dparser
import glob
import utils


fontT = {"family": "serif", "color":  "k", "weight": "normal", "size": 8}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["mathtext.default"] = "default"

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

ev = dt.datetime(2014,6,10,11,42)
rio = utils.read_riometer(ev, "ott")
gos = utils.read_goes(ev)
fig, ax = plt.subplots(figsize=(3,3),nrows=1,ncols=1,dpi=120)
sTime,eTime = dt.datetime(2014,6,10,11,35), dt.datetime(2014,6,10,12,5)
col = "red"
ax = coloring_axes(ax)
font["color"] = col
ax.semilogy(gos.date,gos.B_AVG,col,linewidth=0.75)
ax.set_ylim(1e-6,1e-3)
ax.set_ylabel(r"Solar Flux, $Wm^{-2}$",fontdict=font)
font["color"] = "k"
ax.set_xlabel("Time (UT)",fontdict=font)
ax = coloring_twaxes(ax.twinx())
ax.plot(rio.date, rio.hf_abs,"ko", markersize=1)
ax.grid(False)
ax.set_xlim(sTime,eTime)
ax.set_ylim(-.1, 3.)
ax.set_ylabel("HF Absorption, dB",fontdict=font)
font["color"] = "darkgreen"
ax.text(0.5,1.05,"Station - OTT, 10 June 2014",horizontalalignment="center",
        verticalalignment="center", transform=ax.transAxes,fontdict=font)
font["color"] = "k"
fig.autofmt_xdate(rotation=70,ha="center")
fig.savefig("docs-pub/wisse20/fig1.png",bbox_inches="tight")


f = "data/sim/2014.06.10.11.42/flare.ott.nc.gz"
os.system("gzip -d " + f)
_x_ = Dataset(f.replace(".gz", ""))
os.system("gzip " + f.replace(".gz", ""))
times = num2date(_x_.variables["time"][:], _x_.variables["time"].units, _x_.variables["time"].calendar,
        only_use_cftime_datetimes=False)
times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
i = times.index(ev)
j = times.index(times[10])
alts = _x_.variables["alts"][:]
ne  = _x_.variables["ne"][:]
fig, axes = plt.subplots(figsize=(3, 6), sharey="row", nrows=2, ncols=1, dpi=100)
fig.subplots_adjust(hspace=.3)
ax = axes[0]
ax.semilogx(utils.extp(alts, ne[j, :], 64), alts, "darkred", ls="-", label=r"PF, $n_e$", lw=0.8)
ax.semilogx(utils.extp(alts, ne[i, :], 64), alts, "darkred", ls="--", label=r"F, $n_e$", lw=0.8)
ax.set_ylabel("Height, km", fontdict=font)
ax.set_xlabel(r"Density, $m^{-3}$", fontdict=font)
ax.legend(loc=2, scatterpoints=3, ncol=1, fontsize=8, frameon=True)
ax.set_ylim(70,120)
ax.set_xlim(1e7,1e12)
ax = axes[1]
ax.set_xlabel(r"$\beta^h$, dB/km", fontdict=font)
ax.set_ylabel("Height, km", fontdict=font)
ax.set_ylim(70,120)
ax.plot(utils.extp(alts, _x_.variables["abs.ah.sn.o"][j,:], 68)*10, alts, "b", 
        label=r"PF, $\beta_{ah}(\nu_{sn})\times10$", ls="-", lw=0.8)
ax.plot(utils.extp(alts, _x_.variables["abs.ah.av.cc.o"][j,:], 64)*10, alts, "g", 
        label=r"PF, $\beta_{ah}(\nu_{av}^{cc})\times10$", ls="-", lw=0.8)
#ax.plot(utils.extp(alts, _x_.variables["abs.ah.av.mb.o"][j,:], 64)*10, alts, "k", 
#        label=r"PF, $\beta_{ah}(\nu_{av}^{mb})\times10$", ls="-", lw=0.8)
ax.plot(utils.extp(alts, _x_.variables["abs.ah.sn.o"][i,:], 68), alts, "b", 
        label=r"F, $\beta_{ah}(\nu_{sn})$", ls="--", lw=0.8)
ax.plot(utils.extp(alts, _x_.variables["abs.ah.av.cc.o"][i,:], 64), alts, "g", 
        label=r"F, $\beta_{ah}(\nu_{av}^{cc})$", ls="--", lw=0.8)
#ax.plot(utils.extp(alts, _x_.variables["abs.ah.av.mb.o"][i,:], 64), alts, "k", 
#        label=r"F, $\beta_{ah}(\nu_{av}^{mb})$", ls="--", lw=0.8)
ax.set_xlim(0,.15)
ax.legend(loc=1, scatterpoints=2, ncol=1, fontsize=8, frameon=True)
fig.savefig("docs-pub/wisse20/fig3.png",bbox_inches="tight")


ev = dt.datetime(2014,6,10,11,42)
rio = utils.read_riometer(ev, "ott")
fig, ax = plt.subplots(figsize=(3,3),nrows=1,ncols=1,dpi=120)
sTime,eTime = dt.datetime(2014,6,10,11,35), dt.datetime(2014,6,10,12,5)
ax = coloring_axes(ax, col="gray")
ax.plot(rio.date, rio.hf_abs,color="gray",marker="o", markersize=1,ls="None")
ax.set_ylim(1e-6,1e-3)
font["color"] = "gray"
ax.set_ylabel(r"Observation, dB",fontdict=font)
font["color"] = "k"
ax.set_xlabel("Time (UT)",fontdict=font)
ax.grid(False, axis="y")
ax.set_xlim(sTime,eTime)
ax.set_ylim(-.1, 2.)
ax = coloring_twaxes(ax.twinx(), col="gray")
ax.plot(times, _x_.variables["drap"][:], "darkred", label=r"$\beta_{DRAP2}$", ls="--", lw=0.8)
ax.plot(times, _x_.variables["sato"][:], "red", label=r"$\beta_{sato}$", ls="--", lw=0.8)
ax.plot(times, utils.int_absorption(_x_.variables["abs.ah.sn.o"][:], alts, extpoint=68),
        "b", label=r"$\beta_{ah}(\nu_{sn})$", ls="-", lw=1.2)
ax.plot(times, utils.int_absorption(_x_.variables["abs.ah.av.cc.o"][:], alts, extpoint=64),
        "g", label=r"$\beta_{ah}(\nu_{av}^{cc})$", ls="-", lw=0.8)
#ax.plot(times, utils.int_absorption(_x_.variables["abs.ah.av.mb.o"][:], alts, extpoint=64),
#        "k", label=r"$\beta_{ah}(\nu_{av}^{mb})$", ls="-", lw=1.2)
ax.legend(loc=1, scatterpoints=2, ncol=1, fontsize=8, frameon=True)
ax.set_ylim(-.1, 2.)
font["color"] = "k"
ax.set_ylabel("Modeled HF Absorption, dB",fontdict=font)
font["color"] = "darkgreen"
ax.text(0.5,1.05,"Station - OTT, 10 June 2014",horizontalalignment="center",
                verticalalignment="center", transform=ax.transAxes,fontdict=font)
font["color"] = "k"
fig.autofmt_xdate(rotation=70,ha="center")
fig.savefig("docs-pub/wisse20/fig4.png",bbox_inches="tight")



evs = [dt.datetime(2013,5,13,16,5), dt.datetime(2014,3,29,17,48), dt.datetime(2014,6,10,11,42), dt.datetime(2015,3,11,16,22)]
starts = [dt.datetime(2013,5,13,15,50), dt.datetime(2014,3,29,17,40), dt.datetime(2014,6,10,11,35), dt.datetime(2015,3,11,16,10)]
ends = [dt.datetime(2013,5,13,16,40), dt.datetime(2014,3,29,18,10), dt.datetime(2014,6,10,12,5), dt.datetime(2015,3,11,17)]
fig, axes = plt.subplots(figsize=(6,6),nrows=2,ncols=2,dpi=120)
fig.subplots_adjust(hspace=0.5)
for i, ev, start, end in zip(range(4), evs, starts, ends):
    print(ev)
    f = "data/sim/{dn}/flare.ott.nc.gz".format(dn=ev.strftime("%Y.%m.%d.%H.%M"))
    ax = axes[int(i/2),np.mod(i,2)]
    os.system("gzip -d " + f)
    _x_ = Dataset(f.replace(".gz", ""))
    os.system("gzip " + f.replace(".gz", ""))
    times = num2date(_x_.variables["time"][:], _x_.variables["time"].units, _x_.variables["time"].calendar,
                    only_use_cftime_datetimes=False)
    times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
    times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
    alts = _x_.variables["alts"][:]
    rio = utils.read_riometer(ev, "ott")
    ax = coloring_axes(ax, col="gray")
    ax.plot(rio.date, rio.hf_abs,color="gray",marker="o", markersize=1,ls="None")
    ax.set_ylim(1e-6,1e-3)
    font["color"] = "gray"
    if np.mod(i,2) == 0: ax.set_ylabel(r"Observation, dB",fontdict=font)
    if np.mod(i,2) == 1: ax.set_yticklabels([])
    font["color"] = "k"
    ax.set_xlabel("Time (UT)",fontdict=font)
    ax.grid(False, axis="y")
    ax.set_xlim(start,end)
    ax.set_ylim(-.1, 4.)
    ax.xaxis.set_tick_params(rotation=20)

    ax = coloring_twaxes(ax.twinx(), col="gray")
    if np.mod(i,2) == 0: ax.set_yticklabels([])
    ax.plot(times, _x_.variables["drap"][:], "darkred", label=r"$\beta_{DRAP2}$", ls="--", lw=0.8)
    ax.plot(times, _x_.variables["sato"][:], "red", label=r"$\beta_{sato}$", ls="--", lw=0.8)
    ax.plot(times, utils.int_absorption(_x_.variables["abs.ah.sn.o"][:], alts, extpoint=68),
                    "b", label=r"$\beta_{ah}(\nu_{sn})$", ls="-", lw=1.2)
    ax.plot(times, utils.int_absorption(_x_.variables["abs.ah.av.cc.o"][:], alts, extpoint=64),
                    "g", label=r"$\beta_{ah}(\nu_{av}^{cc})$", ls="-", lw=0.8)
    #ax.plot(times, utils.int_absorption(_x_.variables["abs.ah.av.mb.o"][:], alts, extpoint=64),
    #                "k", label=r"$\beta_{ah}(\nu_{av}^{mb})$", ls="-", lw=1.2)
    ax.legend(loc=1, scatterpoints=2, ncol=1, fontsize=8, frameon=True)
    ax.set_ylim(-.1, 4.)
    ax.xaxis.set_tick_params(rotation=20)
    font["color"] = "k"
    if np.mod(i,2) == 1: ax.set_ylabel("Modeled HF Absorption, dB",fontdict=font)
    font["color"] = "darkgreen"
    ax.text(0.5,1.05,"Stn. - OTT, " + ev.strftime("%Y-%m-%d"),horizontalalignment="center",
            verticalalignment="center", transform=ax.transAxes,fontdict=font)
    font["color"] = "k"
    o = {
            "A": utils.int_absorption(_x_.variables["abs.ah.sn.o"][:], alts, extpoint=68),
            "B": utils.int_absorption(_x_.variables["abs.ah.av.cc.o"][:], alts, extpoint=64),
            "C": _x_.variables["drap"][:],
            "D": _x_.variables["sato"][:]
        }
    utils.Performance("ott", ev, times, o, start, end)
fig.savefig("docs-pub/wisse20/fig5.png",bbox_inches="tight")


if os.path.exists("models/__pycache__"): os.system("rm -rf models/__pycache__")
if os.path.exists("models/experiments/__pycache__"): os.system("rm -rf models/experiments/__pycache__")
