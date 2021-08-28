#!/usr/bin/env python

"""eTemp_analsys.py: Temperature Analysis"""

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
fontT = {"family": "serif", "color":  "k", "weight": "normal", "size": 8}
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}

plt.style.use("seaborn-bright")

#from matplotlib import font_manager
#ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
#matplotlib.rcParams["xtick.color"] = "k"
#matplotlib.rcParams["ytick.color"] = "k"
#matplotlib.rcParams["xtick.labelsize"] = 10
#matplotlib.rcParams["ytick.labelsize"] = 10
#matplotlib.rcParams["mathtext.default"] = "default"

import os
import argparse
import datetime as dt
from dateutil import parser as dparser
import numpy as np
import pandas as pd

import sys
sys.path.append("models/")
from netCDF4 import Dataset, num2date
from model import Model
from fetch_data import Simulation, Riometer
from constant import *
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prog", default="plot", help="Program code [bgc/flare] (default bgc)")
    parser.add_argument("-r", "--rio", default="ott", help="Riometer code (default ott)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,22), help="Start date (default 2015-3-11T16:22)",
            type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,16), help="Start date (default 2015-3-11T15:30)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,17), help="End date (default 2015-3-11T17:30)",
            type=dparser.isoparse)
    parser.add_argument("-g", "--save_goes", action="store_false", help="Save goes data (default True)")
    parser.add_argument("-sat", "--sat", type=int, default=15, help="Satellite number (default 15)")
    parser.add_argument("-sps", "--species", type=int, default=0, help="Species Type (default 0)")
    parser.add_argument("-rm", "--save_riom", action="store_false", help="Save riometer data (default True)")
    parser.add_argument("-ps", "--plot_summary", action="store_true", help="Plot summary report (default False)")
    parser.add_argument("-sr", "--save_result", action="store_false", help="Save results (default True)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-irr", "--irradiance", default="EUVAC+", help="Irradiance model (default EUVAC+)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    parser.add_argument("-pc", "--plot_code", type=int, default=0, help="Plotting code,applicable if --prog==plot (default 0)")
    parser.add_argument("-fr", "--frequency", type=float, default=30, help="Frequency of oprrations in MHz (default 30 MHz)")
    parser.add_argument("-ex", "--exp", type=int, default=0, help="Program code [0-10] (default0)")
    args = parser.parse_args()
    args.event, args.rio, args.start, args.end = dt.datetime(2011,9,7,22,38), "mcmu",\
            dt.datetime(2011,9,7,22,28), dt.datetime(2011,9,7,23,28)
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     " , k , "->" , str(vars(args)[k]))
    
    case = "plot"
    start, end =  dt.datetime(2011,9,7,22,30), dt.datetime(2011,9,7,23,0)
    TElec = np.linspace(0.75,1.75,101).round(2)
    if case == "flare":
        for t in TElec:
            print(" TElec:", t)
            Model(args.rio, args.event, args, _dir_="proc/outputs/tElec/{date}/{rio}/")._exp_("TElec", {"TElec": t})
    if case == "plot":
        matplotlib.rcParams["xtick.labelsize"] = 6
        matplotlib.rcParams["ytick.labelsize"] = 6
        matplotlib.rcParams["mathtext.default"] = "default"
        font = {"family": "serif", "color":  "black", "weight": "normal", "size": 6}
        fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 6}
        fmt = mdates.DateFormatter("%H:%M")
        print("Here")
        sim = Simulation(args.event, args.rio)
        _dir = "proc/outputs/tElec/{date}/{rio}/".format(date=args.event.strftime("%Y.%m.%d.%H.%M"), rio=args.rio)
        if not os.path.exists(_dir): os.system("mkdir -p "+_dir)
        print("Here")
        X = []
        plt.style.use("seaborn-bright")
        fig, axes = plt.subplots(figsize=(6, 2), nrows=1, ncols=2, dpi=150)
        ax = axes[0]
        ax.xaxis.set_major_formatter(fmt)
        cmap = matplotlib.cm.get_cmap("Reds")
        Mx = np.zeros((60, len(TElec)))
        print("Here")
        Riometer().get_riometer_file(None, args.event, args.rio)
        _abs_ = utils.read_riometer(args.event, args.rio)
        _abs_ = _abs_[(_abs_.date > start) & (_abs_.date < end-dt.timedelta(minutes=1))]
        print("Here")
        for i, t in enumerate(TElec):
            print(i, t)
            fname = _dir + "flare.TElec_%.2f.nc.gz"%t
            if not os.path.exists(fname.replace(".gz", "")): sim.get_flare_file(None, fname)
            nc = Dataset(fname.replace(".gz", ""))
            times = num2date(nc.variables["time"][:], nc.variables["time"].units, nc.variables["time"].calendar)
            times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
            times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
            m = pd.DataFrame()
            m["date"] = times
            m["absorp"] = utils.smooth(utils.int_absorption(nc.variables["abs.ah.sn.o"][:], model["alts"], extpoint=68), 5)
            m["absorp"] = m["absorp"]*0.62
            Mx[:,i] = m.absorp.tolist()
            e = utils.estimate_error(m, _abs_)
            X.append(e)
            print(fname)
        ax.plot(_abs_.date, _abs_.absorp, "ko", alpha=0.9, markersize=.5, label=r"$\beta_{R}$", lw=.4)
        mn, st = 1.2*np.median(Mx, axis=1), 1.98*np.std(Mx, axis=1)
        ax.plot(m.date, mn, color="r", linewidth=0.8, ls="--", label=r"$\beta_m$")
        ax.fill_between(m.date, mn - st, mn + st, color="r", alpha=0.5, label="95% CI")
        X = np.array(X)
        ax.set_xlim(start, end)
        ax.text(0.5, 1.05, r"(a) %s UT, %s @30 MHz, $T_d=\frac{T^{90}}{T^{90}_{base}}$"%(args.event.strftime("%Y-%m-%d"), 
            args.rio.upper()), horizontalalignment="center",
            verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        ax.set_ylim(-.1,2.5)
        ax.legend(loc=1, scatterpoints=3, fontsize=6, ncol=1, frameon=True)
        ax.set_xlabel("Time (UT)", fontdict=font)
        ax.set_ylabel("Absorption, dB", fontdict=font)
        ax = axes[1]
        ax.grid(False, axis="y")
        ax.set_xlabel(r"Temperature ratio, $\frac{T^{90}}{T^{90}_{base}}$", fontdict=font)
        ax.set_yticklabels([])
        ax = ax.twinx()
        ax.plot(TElec, X, "ro", markersize=0.3, alpha=.6)
        ax.set_xlim(.75,1.75)
        ax.axvline(TElec[np.argmin(X)], ls="--", lw=0.4, color="b")
        ax.set_ylabel("RMSE", fontdict=font)
        ax.text(0.5, 1.05, "(b) Impact of Temperature on RMSE", horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        fonttext["size"] = 4
        ax.text(TElec[np.argmin(X)], np.min(X)-0.005, r"$T_d$=%.2f"%TElec[np.argmin(X)], horizontalalignment="center",
                verticalalignment="center", fontdict=fonttext, rotation=90)
        fig.autofmt_xdate()
        fig.savefig("figs/Figure08.png", bbox_inches="tight")
        Riometer().clean_local_file(args.event, args.rio)
        #if os.path.exists(_dir): os.system("rm -rf "+_dir)