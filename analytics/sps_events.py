#!/usr/bin/env python

"""sps_events.py: Fetch data from the remote locations and run sps events"""

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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import datetime as dt
from fetch_data import Goes, Simulation
import pandas as pd
from netCDF4 import Dataset, num2date
import numpy as np

import sys
sys.path.append("models/")
import utils
from constant import *

fontT = {"family": "serif", "color":  "k", "weight": "normal", "size": 8}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["mathtext.default"] = "default"

def coloring_axes(ax, atype="left", col="red", fmtr="%H", ivl=60):
    ax.spines[atype].set_color(col)
    ax.tick_params(axis="y", which="both", colors=col)
    ax.yaxis.label.set_color(col)
    fmt = mdates.DateFormatter(fmtr)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=ivl))
    return ax

def coloring_twaxes(ax, atype="left", col="red", twcol="k", fmtr="%H", ivl=60):
    ax.spines[atype].set_color(col)
    ax.tick_params(axis="y", which="both", colors=twcol)
    ax.yaxis.label.set_color(twcol)
    fmt = mdates.DateFormatter(fmtr)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=ivl))
    return ax

if __name__ == "__main__":
    case = "plot"
    ################################
    # Fetch GOES bulk request
    ################################
    events, rs, starts, ends = [dt.datetime(2017,9,5), dt.datetime(2017,9,6), dt.datetime(2017,9,7)],\
                ["sps","sps","sps"],\
                [dt.datetime(2017,9,5,17), dt.datetime(2017,9,6,11), dt.datetime(2017,9,7,13,30)],\
                [dt.datetime(2017,9,5,19,30), dt.datetime(2017,9,6,17), dt.datetime(2017,9,7,19)]
    if case=="goes": Goes.run_goes_downloads_dates(events)
    ################################
    # Run background and flare model
    ################################
    for ev, r, start, end in zip(events, rs, starts, ends):
        if case == "bgc":
            cmd = "python bgc.py -ev {ev} -s {s} -e {e} -r {r} -v -fr {fr}".format(r=r,
                    ev=ev.strftime("%Y-%m-%dT%H:%M"), s=start.strftime("%Y-%m-%dT%H:%M"),
                    e=end.strftime("%Y-%m-%dT%H:%M"), fr=5.24)
            print(" "+ cmd)
            os.system(cmd)
        elif case == "flare":
            cmd = "python simulate.py -p flare -r {r} -ev {ev} -s {s} -e {e} -v -fr {fr}".format(r=r,
                    ev=ev.strftime("%Y-%m-%dT%H:%M"), s=start.strftime("%Y-%m-%dT%H:%M"),
                    e=end.strftime("%Y-%m-%dT%H:%M"), fr=5.24)
            print(" "+ cmd)
            os.system(cmd)
    if case == "plot":
        fmt = mdates.DateFormatter("%H")
        fig, axes = plt.subplots(figsize=(9, 5), nrows=2, ncols=3, dpi=150)
        fig.subplots_adjust(hspace=.1, wspace=.3)
        i = 0
        CC = ["M2.3", "X9.3", "X1.7"]
        for ev, start, end in zip(events, starts, ends):
            Goes().get_goes_file(None, ev)
            _X_ = pd.read_csv("../config/dat/case1.ev{t}.csv".format(t=i))
            _X_["dt"] = [ev + dt.timedelta(hours=h) for h in _X_.dt]
            _X_ = _X_.sort_values(by=["dt"])
            ax = axes[0,i]
            gos = utils.read_goes(ev)
            col = "r"
            ax = coloring_axes(ax, col="k")
            ax.semilogy(gos.date,gos.B_AVG,col,linewidth=0.75, label="SXR (.1-.8 nm)")
            ax.semilogy(gos.date,gos.A_AVG,"b",linewidth=0.75, label="HXR (.05-.4 nm)")
            if i==2: ax.legend(bbox_to_anchor=(1.4, 0.5), scatterpoints=3, ncol=1, fontsize=8, frameon=True)
            ax.set_ylim(1e-8,1e-3)
            ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4,1e-3])
            ax.set_xlim(start,end)
            if i==0: ax.set_ylabel(r"Solar Flux, $Wm^{-2}$",fontdict=font)
            font["color"] = "k"
            font["color"] = "darkgreen"
            ax.text(0.5,1.05,"%s, %s"%(ev.strftime("%d %b %Y"), CC[i]),horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes,fontdict=font)
            font["color"] = "k"

            ax = axes[1,i]
            ax = coloring_axes(ax, col="k")
            if i==0:
                ax.set_ylabel("Observations \n HF Absorption, db",fontdict=font)
            ax.scatter(_X_[_X_.model=="N"].dt, _X_[_X_.model=="N"].db, s=3., color="gray", alpha=0.8, marker="D", label="Ionosonde")
            
            sim = Simulation(ev, "sps")
            sim.create_remote_local_dir()
            sim.get_flare_file()
            fname = sim._dir_ + "flare.nc"
            nc = Dataset(fname)
            times = num2date(nc.variables["time"][:], nc.variables["time"].units, nc.variables["time"].calendar)
            times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
            times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
            ax.plot(times, 2*nc.variables["drap"][:], "darkred", ls="--", linewidth=0.8, label="DRAP2")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.ah.sn.o"][:], model["alts"], extpoint=68), "r",
                                        linewidth=1.2, label=r"$\beta_{ah}(\nu_{sn})$")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.ah.av.cc.o"][:], model["alts"], extpoint=64), "g",
                    linewidth=0.8, label=r"$\beta_{ah}(\nu_{cc}^{av})$")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.ah.av.mb.o"][:], model["alts"], extpoint=64), "b",
                    linewidth=1.2, label=r"$\beta_{ah}(\nu_{mb}^{av})$")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.sw.ft.o"][:], model["alts"], extpoint=64), "k",
                    linewidth=0.8, label=r"$\beta_{sw}(\nu_{me})$")
            ax.set_xlim(start, end)
            ax.scatter(_X_[_X_.model=="Y"].dt, _X_[_X_.model=="Y"].db, s=1.2, color="darkred", 
                    alpha=0.8, label="Levine et al. (2019)")
            if i==2: ax.legend(bbox_to_anchor=(1.12, 0.9), scatterpoints=3, fontsize=8, frameon=True)
            i += 1
            sim.clear_local_folders()
            Goes().clean_local_file(ev)
            
        axes[1,0].set_xlabel("Time (UT)", fontdict=font)
        axes[1,1].set_xlabel("Time (UT)", fontdict=font)
        axes[1,2].set_xlabel("Time (UT)", fontdict=font)
        fig.autofmt_xdate(rotation=25,ha="center")
        fig.savefig("figs/Figure05.png", bbox_inches="tight")