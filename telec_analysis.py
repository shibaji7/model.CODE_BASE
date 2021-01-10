#!/usr/bin/env python

""" temperature_analysis.py: simulate python program for absorption calculation for different temperature analysis """

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

import os
import numpy as np
import pandas as pd

np.random.seed(0)

import sys
sys.path.append("models/")
import datetime as dt
import argparse
from dateutil import parser as dparser
from netCDF4 import Dataset, num2date
import glob

import utils
from model import Model
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prog", default="flare", help="Program code [bgc/flare] (default bgc)")
    parser.add_argument("-r", "--rio", default="ott", help="Riometer code (default ott)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,22), help="Start date (default 2015-3-11T16:22)",
            type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,15,30), help="Start date (default 2015-3-11T15:30)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,17,30), help="End date (default 2015-3-11T17:30)",
            type=dparser.isoparse)
    parser.add_argument("-g", "--save_goes", action="store_false", help="Save goes data (default True)")
    parser.add_argument("-sat", "--sat", type=int, default=15, help="Satellite number (default 15)")
    parser.add_argument("-rm", "--save_riom", action="store_false", help="Save riometer data (default True)")
    parser.add_argument("-ps", "--plot_summary", action="store_true", help="Plot summary report (default False)")
    parser.add_argument("-sr", "--save_result", action="store_false", help="Save results (default True)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-irr", "--irradiance", default="EUVAC+", help="Irradiance model (default EUVAC+)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    parser.add_argument("-pc", "--plot_code", type=int, default=0, help="Plotting code,applicable if --prog==plot (default 0)")
    parser.add_argument("-sps", "--species", type=int, default=0, help="Species Type (default 0)")
    parser.add_argument("-fr", "--frequency", type=float, default=30, help="Frequency of oprrations in MHz (default 30 MHz)")
    args = parser.parse_args()
    case = "est"
    ox = pd.read_csv("config/temperature_analysis.csv", parse_dates=["dn"])
    dx = pd.read_csv("config/flares.csv", parse_dates=["dn","start","end"])
    Tr, Err = [], []
    for i, o in ox.iterrows():
        dn = o["dn"]
        op = dx[dx.dn==o.dn]
        start, end = op["start"].tolist()[0], op["end"].tolist()[0]
        u = False
        for rio in o["rios"].split("~"):
            args.rio = rio
            args.event = dn
            args.start = start
            args.end = end
            if case == "bgc":
                if not os.path.exists("data/tElec/{dn}/bgc.{stn}.nc.gz".format(dn=dn.strftime("%Y.%m.%d.%H.%M"), stn=rio)):
                    cmd = "python2.7 models/bgc.py -r {r} -ev {ev} -s {s} -e {e} -v -fr {fr}".format(r=rio,
                            ev=dn.strftime("%Y-%m-%dT%H:%M"), s=start.strftime("%Y-%m-%dT%H:%M"),
                            e=end.strftime("%Y-%m-%dT%H:%M"), fr=args.frequency)
                    print(" -->", cmd)
                    os.system(cmd)
                    utils.add_chi(args.event, args.rio, args.start, args.end)
                    print(" SZA estimation done")
            elif case == "flare":
                TElec = np.linspace(0.75,1.75,101)
                for t in TElec:
                    if not os.path.exists("data/tElec/{dn}/flare.{stn}.TElec[%.2f].nc.gz".format(dn=dn.strftime("%Y.%m.%d.%H.%M"), stn=rio)%t): 
                        print("data/tElec/{dn}/flare.{stn}.TElec[%.2f].nc.gz".format(dn=dn.strftime("%Y.%m.%d.%H.%M"), stn=rio)%t)
                        Model(rio, args.event, args, _dir_="data/tElec/{date}")._exp_("TElec", {"TElec": t})
                        u = True
            elif case == "est":
                files = glob.glob("data/tElec/{dn}/flare.{r}.TElec*".format(dn=args.event.strftime("%Y.%m.%d.%H.%M"), r=rio))
                files.sort()
                _abs_ = utils.read_riometer(args.event, args.rio)
                _abs_ = _abs_[(_abs_.date > start) & (_abs_.date < end-dt.timedelta(minutes=1))]
                Mx = np.zeros((int((end-start).total_seconds()/60), len(files)))
                tr, err, errm = [], [], []
                for j,f in enumerate(files):
                    tr.append(0.75+j*.01)
                    os.system("gzip -d " + f)
                    nc = Dataset(f.replace(".gz", ""))
                    os.system("gzip " + f.replace(".gz", ""))
                    times = num2date(nc.variables["time"][:], nc.variables["time"].units, nc.variables["time"].calendar)
                    times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
                    times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
                    m = pd.DataFrame()
                    m["date"] = times
                    m["hf_abs"] = utils.smooth(utils.int_absorption(nc.variables["abs.ah.sn.o"][:], model["alts"], extpoint=68), 5)
                    m = m[(m.date >= start) & (m.date < end)]
                    Mx[:,i] = m.hf_abs.tolist()
                    #e = utils.estimate_error(m, _abs_)
                    #err.append(e)
                    #emax = np.abs(np.mean(m.hf_abs) - np.mean(_abs_.hf_abs))
                    #errm.append(emax)
                    o = {
                            "sn": utils.smooth(utils.int_absorption(nc.variables["abs.ah.sn.o"][:], model["alts"], extpoint=68), 5),
                            "dr": nc.variables["drap"][:],
                        }
                    pf = utils.Performance(stn=rio, ev=dn, times=times, model=o, start=start, end=end, bar=4, alt=np.nan)
                    pf._skill_()._params_()
                    #print(pf.attrs)
                    err.append(pf.attrs["mRMSE_sn"])
                fig, axes = plt.subplots(figsize=(4, 4), nrows=1, ncols=1, dpi=100)
                ax = axes
                ax.set_ylabel("Error", fontdict=font)
                ax.set_xlabel(r"Temperature ratio, $\frac{T^{90}}{T^{90}_{base}}$", fontdict=font)
                ax.plot(tr, err, "ro", markersize=1)
                #ax.plot(tr, errm, "bo", markersize=1)
                #ax.axvline(tr[np.argmin(err)], color="r", lw=1.2)
                #ax.axvline(tr[np.argmin(errm)], color="b", lw=1.2)
                fig.savefig("_images_/te_analysis_%s.%s.png"%(args.event.strftime("%Y-%m-%d-%H-%M"), rio), bbox_inches="tight")
                Err.append(np.min(err))
                Tr.append(tr[np.argmin(err)])
                print(rio,dn,tr[np.argmin(err)], np.min(err))
                #print(pf.attrs)
        #if i==1: break
    if case == "est":
        df = pd.DataFrame()
        df["tr"], df["err"] = Tr, Err
        df.to_csv("data.csv", index=False, header=True)
        print(Tr)
        matplotlib.rcParams["xtick.labelsize"] = 6
        matplotlib.rcParams["ytick.labelsize"] = 6
        matplotlib.rcParams["mathtext.default"] = "default"
        font = {"family": "serif", "color":  "black", "weight": "normal", "size": 6}
        fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 6}
        fmt = matplotlib.dates.DateFormatter("%H:%M")
        fig, axes = plt.subplots(figsize=(4, 4), nrows=1, ncols=1, dpi=100)
        ax = axes
        ax.set_ylabel("Density", fontdict=font)
        ax.set_xlabel(r"Temperature ratio, $\frac{T^{90}}{T^{90}_{base}}$", fontdict=font)
        #ax.plot(TElec, X, "ro", markersize=0.3, alpha=.6)
        ax.hist(df.tr, bins=np.arange(.75,1.75,.05), alpha=0.5, color="r", density=True)
        ax.set_xlim(.75,1.75)
        #ax.axvline(TElec[np.argmin(X)], ls="--", lw=0.4, color="b")
        #ax.text(0.5, 1.05, "(b) Impact of Temperature on RMSE", horizontalalignment="center",
        #        verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        fonttext["size"] = 4
        #ax.text(TElec[np.argmin(X)], 0.745, r"$T_d$=%.2f"%TElec[np.argmin(X)], horizontalalignment="center",
        #        verticalalignment="center", fontdict=fonttext, rotation=90)
        fig.autofmt_xdate()
        fig.savefig("_images_/te_analysis.png", bbox_inches="tight")
    os.system("rm models/*.pyc")
    os.system("rm -rf models/__pycache__")
