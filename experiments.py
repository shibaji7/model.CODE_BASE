#!/usr/bin/env python

"""experiments.py: experiments python program for different experiment applications"""

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
sys.path.append("models/experiments/")
import os
import numpy as np
import argparse
import pandas as pd
import datetime as dt
from netCDF4 import Dataset, num2date
from dateutil import parser as dparser
import glob

from constant import *
import utils
from absorption import *
import case0
from model import Model


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
    fmt = matplotlib.dates.DateFormatter(fmtr)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=ivl))
    return ax

def coloring_twaxes(ax, atype="left", col="red", twcol="k", fmtr="%H", ivl=60):
    ax.spines[atype].set_color(col)
    ax.tick_params(axis="y", which="both", colors=twcol)
    ax.yaxis.label.set_color(twcol)
    fmt = matplotlib.dates.DateFormatter(fmtr)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=ivl))
    return ax

def _case0_(args):
    """ Impact of the I0 and frequency """

    chi = np.deg2rad(np.linspace(0,90,91))
    f0 = 10**np.linspace(-6,-1,31) * 1e3
    fo = 10**np.linspace(np.log10(.1), np.log10(200), 100)

    ev, start, end = dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,15,30), dt.datetime(2015,3,11,17,30)
    l, r = 52, 53
    _f0_ = case0._Case0_(start, end)[40:53]
    fname = "data/sim/case0.nc.gz"
    os.system("gzip -d "+fname)
    _nc = Dataset(fname.replace(".gz", ""))
    os.system("gzip "+fname.replace(".gz", ""))
    pg = utils.PointGrid("ott", ev, start, end, 30, v=False)
    _lo_,_qo_ = [],[]
    b = pg.igrf["B"][l:r,:]
    pg._col_.nu_FT = pg._col_.nu_FT[l:r,:]
    pg._col_.nu_av_CC = pg._col_.nu_av_CC[l:r,:]
    pg._col_.nu_av_MB = pg._col_.nu_av_MB[l:r,:]
    pg._col_.nu_SN["total"] = pg._col_.nu_SN["total"][l:r,:]
    ne = _nc.variables["ne"][l:r,:]
    for _f_ in fo:
        print(" Frequency - ", _f_, " MHz")
        u = Absorption(b, pg._col_, ne, fo=_f_*1e6)
        _lo_.append([utils.int_absorption(u.AH["SN"]["O"], pg.alts, extpoint=68, llim = 60, ulim = 110),
             utils.int_absorption(u.AH["AV_CC"]["O"], pg.alts, extpoint=64, llim = 60, ulim = 110),
             utils.int_absorption(u.AH["AV_MB"]["O"], pg.alts, extpoint=64, llim = 60, ulim = 110),
             utils.int_absorption(u.SW["FT"]["O"], pg.alts, extpoint=64, llim = 60, ulim = 110)])
        continue
    _lo_ = np.array(_lo_)

    ne = _nc.variables["ne"][40:53,:]
    nfo = np.linspace(1,70,50)
    for i, _ in enumerate(_f0_):
        _k_ = []
        for _f_ in nfo:
            print(" Frequency, I - ", _f_, " MHz,", _f0_[i], "W/m2")
            u = Absorption(b, pg._col_, ne[i:i+1,:], fo=_f_*1e6)
            _k_.append([utils.int_absorption(u.AH["SN"]["O"], pg.alts, extpoint=68, llim = 60, ulim = 110),
                utils.int_absorption(u.AH["AV_CC"]["O"], pg.alts, extpoint=64, llim = 60, ulim = 110),
                utils.int_absorption(u.AH["AV_MB"]["O"], pg.alts, extpoint=64, llim = 60, ulim = 110),
                utils.int_absorption(u.SW["FT"]["O"], pg.alts, extpoint=64, llim = 60, ulim = 110)])
        _k_ = np.array(_k_)[:,:,0]
        _qo_.append([10**utils.extrap1d(_k_[:,0], np.log10(nfo))([1])[0], 
            10**utils.extrap1d(_k_[:,1], np.log10(nfo))([1])[0],
            10**utils.extrap1d(_k_[:,2], np.log10(nfo))([1])[0],
            10**utils.extrap1d(_k_[:,3], np.log10(nfo))([1])[0]])
    _qo_ = np.array(_qo_)

    haf0 = 93.5 * (f0**0.25)
    l0 = 4.37e3 * (.22**0.5) / (fo)**2
    haf1 = 10*np.log10(f0*1e-3) + 65
    l1 = ((10*np.log10(2.2e-4) + 65)/fo)**1.5

    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["mathtext.default"] = "default"
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}
    #fig, axes = plt.subplots(figsize=(6, 6), nrows=2, ncols=2, dpi=150)
    #fig.subplots_adjust(hspace=.3, wspace=.1)
    fig, ax = plt.subplots(figsize=(3, 3), nrows=1, ncols=1, dpi=100)

    #ax = axes[0,0]
    #ax.loglog(f0*1e-3, haf0, "r", linewidth=1.2, label="Sato (1975)")
    #ax.loglog(f0*1e-3, haf1, "b", linewidth=1.2, label="DARP2")
    #ax.set_ylabel("HAF, MHz", fontdict=font)
    #ax.set_xlim(1e-6,1e-1)
    #ax.legend(loc=2, scatterpoints=3, fontsize=8, frameon=True)
    #ax.text(0.2, 1.05, r"(a) $\chi=0^o$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

    #ax = axes[0,1]
    #ax.set_yticks([])
    #ax = ax.twinx()
    #ax.loglog(fo, l0, "r", linewidth=1.2, label="Sato (1975)")
    ax.loglog(fo, l1, "darkred", ls="--", linewidth=0.8, label="DARP")
    ax.set_xlim(1,200)
    ax.set_ylim(1,1e5)
    ax.set_ylabel("Absorption, dB", fontdict=font)
    #ax.legend(loc=1, scatterpoints=3, fontsize=8, frameon=True)
    ax.text(0.5, 1.05, r"$\chi=0^o$, $I_{\infty}=2.2\times 10^{-4}$ $Wm^{-2}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

    #ax = axes[1,0]
    #ax.loglog(_f0_, _qo_[:,0], "ro", markersize=1.2, label=r"$\beta_{ah}(\nu_{sn})$")
    #ax.loglog(_f0_, _qo_[:,1], "go", markersize=0.8, label=r"$\beta_{ah}(\nu_{av}^{cc})$")
    #ax.loglog(_f0_, _qo_[:,2], "bo", markersize=1.2, label=r"$\beta_{ah}(\nu_{av}^{mb})$")
    #ax.loglog(_f0_, _qo_[:,3], "ko", markersize=1.2, label=r"$\beta_{sw}(\nu_{me})$")
    #ax.set_ylabel("HAF, MHz", fontdict=font)
    #ax.set_xlabel(r"SXR, $Wm^{-2}$", fontdict=font)
    #ax.set_xlim(1e-6,1e-1)
    #ax.legend(loc=2, scatterpoints=3, fontsize=8, frameon=True)
    #ax.text(0.2, 1.05, r"(b) $\chi=0^o$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

    #ax = axes[1,1]
    ax.set_xlabel("Frequency, MHz", fontdict=font)
    #ax.set_yticks([])
    #ax = ax.twinx()
    ax.loglog(fo, utils.smooth(_lo_[:,0,0], 11), "r", linewidth=1.2, label=r"$\beta_{ah}(\nu_{sn})$")
    ax.loglog(fo, utils.smooth(_lo_[:,1,0], 11), "g", linewidth=0.8, label=r"$\beta_{ah}(\nu_{av}^{cc})$")
    ax.loglog(fo, utils.smooth(_lo_[:,2,0], 11), "b", linewidth=1.2, label=r"$\beta_{ah}(\nu_{av}^{mb})$")
    ax.loglog(fo, utils.smooth(_lo_[:,3,0], 11), "k", linewidth=1.2, label=r"$\beta_{sw}(\nu_{me})$")
    ax.set_ylim(1,1e5)
    ax.set_xlim(1,200)
    ax.set_ylabel("Absorption, dB", fontdict=font)
    ax.legend(loc=1, scatterpoints=3, fontsize=8, frameon=True)
    ax.text(0.5, 1.05, r"$\chi=0^o$, $I_{\infty}=2.2\times 10^{-4}$ $Wm^{-2}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

    fig.savefig("_images_/case0.png", bbox_inches="tight")
    return

def _case1_(args):
    """ Impact of special event case study """
    evs, rs, starts, ends = [dt.datetime(2017,9,5), dt.datetime(2017,9,6), dt.datetime(2017,9,7)],\
                ["sps","sps","sps"],\
                [dt.datetime(2017,9,5,17), dt.datetime(2017,9,6,11), dt.datetime(2017,9,7,13,30)],\
                [dt.datetime(2017,9,5,19,30), dt.datetime(2017,9,6,17), dt.datetime(2017,9,7,19)]
    for ev, r, start, end in zip(evs, rs, starts, ends):
        if args.prog == "bgc":
            cmd = "python simulate.py -p bgc -r {r} -ev {ev} -s {s} -e {e} -v -fr {fr}".format(r=r,
                    ev=ev.strftime("%Y-%m-%dT%H:%M"), s=start.strftime("%Y-%m-%dT%H:%M"),
                    e=end.strftime("%Y-%m-%dT%H:%M"), fr=5.24)
            print(" "+ cmd)
            os.system(cmd)
        elif args.prog == "flare":
            cmd = "python simulate.py -p flare -r {r} -ev {ev} -s {s} -e {e} -v -fr {fr} -rm".format(r=r,
                    ev=ev.strftime("%Y-%m-%dT%H:%M"), s=start.strftime("%Y-%m-%dT%H:%M"),
                    e=end.strftime("%Y-%m-%dT%H:%M"), fr=5.24)
            print(" "+ cmd)
            os.system(cmd)
    if args.prog == "plot": 
        fmt = matplotlib.dates.DateFormatter("%H")
        fig, axes = plt.subplots(figsize=(9, 5), nrows=2, ncols=3, dpi=150)
        fig.subplots_adjust(hspace=.1, wspace=.3)

        i = 0
        CC = ["M2.3", "X9.3", "X1.7"]
        for ev, start, end in zip(evs, starts, ends):
            _X_ = pd.read_csv("config/dat/case1.ev{t}.csv".format(t=i))
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
            #ax.legend(loc=1, scatterpoints=3, fontsize=8, frameon=True)
            #ax.text(0.35, 1.05, "(a.%d) "%(i+1) + ev.strftime("%Y-%m-%d")+" UT", horizontalalignment="center", 
            #        verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

            fname = "data/sim/{date}/flare.sps.nc.gz".format(date=ev.strftime("%Y.%m.%d.%H.%M"))
            os.system("gzip -d " + fname)
            nc = Dataset(fname.replace(".gz",""))
            os.system("gzip " + fname.replace(".gz",""))
            times = num2date(nc.variables["time"][:], nc.variables["time"].units, nc.variables["time"].calendar)
            times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
            times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
            ax.plot(times, 2*nc.variables["drap"][:], "darkred", ls="--", linewidth=0.8, label="DRAP2")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.ah.sn.o"][:], model["alts"], extpoint=68), "r",
                                        linewidth=1.2, label=r"$\beta_{ah}(\nu_{sn})$")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.ah.av.cc.o"][:], model["alts"], extpoint=64), "g",
                    linewidth=0.8, label=r"$\beta_{ah}(\nu_{av}^{cc})$")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.ah.av.mb.o"][:], model["alts"], extpoint=64), "b",
                    linewidth=1.2, label=r"$\beta_{ah}(\nu_{av}^{mb})$")
            ax.plot(times, 2*utils.int_absorption(nc.variables["abs.sw.ft.o"][:], model["alts"], extpoint=64), "k",
                    linewidth=0.8, label=r"$\beta_{ah}(\nu_{av}^{cc})$")
            ax.set_xlim(start, end)
            ax.scatter(_X_[_X_.model=="Y"].dt, _X_[_X_.model=="Y"].db, s=1.2, color="darkred", 
                    alpha=0.8, label="Levine et al. (2019)")
            if i==2: ax.legend(bbox_to_anchor=(1.12, 0.9), scatterpoints=3, fontsize=8, frameon=True)
            #ax.text(0.5, 1.05, "(b.%d) "%(i+1) + ev.strftime("%Y-%m-%d")+" UT, @6.4 MHz", horizontalalignment="center", 
            #        verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
            i += 1

        axes[1,0].set_xlabel("Time (UT)", fontdict=font)
        axes[1,1].set_xlabel("Time (UT)", fontdict=font)
        axes[1,2].set_xlabel("Time (UT)", fontdict=font)
        font["color"] = "k"
        fig.autofmt_xdate(rotation=25,ha="center")
        fig.savefig("_images_/case1.png", bbox_inches="tight")
    return

def _case2_(args):
    """ Testing electron temperature dependence """
    args.event, args.rio, args.start, args.end = dt.datetime(2011,9,7,22,38), "mcmu",\
            dt.datetime(2011,9,7,22,10), dt.datetime(2011,9,7,23,20)
    start, end =  dt.datetime(2011,9,7,22,30), dt.datetime(2011,9,7,23,0)
    TElec = np.linspace(0.75,1.75,101)
    if args.prog == "bgc":
        cmd = "python simulate.py -p bgc -r {r} -ev {ev} -s {s} -e {e} -v".format(r=args.rio,
                ev=args.event.strftime("%Y-%m-%dT%H:%M"), s=args.start.strftime("%Y-%m-%dT%H:%M"),
                e=args.end.strftime("%Y-%m-%dT%H:%M"))
        print(" "+ cmd)
        os.system(cmd)
    elif args.prog == "flare":
        for t in TElec:
            print(" TElec:", t)
            Model(args.rio, args.event, args)._exp_("TElec", {"TElec": t})
    elif args.prog == "plot":
        matplotlib.rcParams["xtick.labelsize"] = 6
        matplotlib.rcParams["ytick.labelsize"] = 6
        matplotlib.rcParams["mathtext.default"] = "default"
        font = {"family": "serif", "color":  "black", "weight": "normal", "size": 6}
        fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 6}
        fmt = matplotlib.dates.DateFormatter("%H:%M")
        fig, axes = plt.subplots(figsize=(6, 2), nrows=1, ncols=2, dpi=100)
        #ax = axes[0]
        #import bootstrapped.bootstrap as bs
        #import bootstrapped.stats_functions as bs_stats
        #from scipy.stats import norm
        #x = np.linspace(0.5,2,151)
        #loc = bs.bootstrap(x, stat_func=bs_stats.mean).value
        #scale = bs.bootstrap(x, stat_func=bs_stats.std).value
        #ax.plot(x, norm.pdf(x, loc=loc, scale=scale), "r", lw=0.8, alpha=0.6)
        #ax.text(0.5, 1.05, r"(a) Distribution of $T_d$", horizontalalignment="center",
        #        verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        files = glob.glob("data/sim/{dn}/flare*TElec*".format(dn=args.event.strftime("%Y.%m.%d.%H.%M")))
        ax = axes[0]
        ax.xaxis.set_major_formatter(fmt)
        files.sort()
        X = []
        _abs_ = utils.read_riometer(args.event, args.rio)
        _abs_ = _abs_[(_abs_.date > start) & (_abs_.date < end-dt.timedelta(minutes=1))]
        cmap = matplotlib.cm.get_cmap("Reds")
        Mx = np.zeros((int((end-start).total_seconds()/60), len(files)))
        for i,f in enumerate(files):
            os.system("gzip -d " + f)
            nc = Dataset(f.replace(".gz", ""))
            os.system("gzip " + f.replace(".gz", ""))
            times = num2date(nc.variables["time"][:], nc.variables["time"].units, nc.variables["time"].calendar)
            times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
            times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
            #if np.mod(i,20)==0: ax.plot(times, utils.smooth(utils.int_absorption(nc.variables["abs.ah.sn.o"][:],
            #    model["alts"], extpoint=68), 5), color=cmap(.2+(i/200)), 
            #    linewidth=0.6, ls="--", label=r"$T_d$=%.2f"%TElec[i])
            m = pd.DataFrame()
            m["date"] = times
            m["hf_abs"] = utils.smooth(utils.int_absorption(nc.variables["abs.ah.sn.o"][:], model["alts"], extpoint=68), 5)
            m = m[(m.date >= start) & (m.date < end)]
            Mx[:,i] = m.hf_abs.tolist()
            e = utils.estimate_error(m, _abs_)
            X.append(e)
        ax.plot(_abs_.date, _abs_.hf_abs, "ko", alpha=0.4, markersize=0.1, label=r"$\beta_{R}$", lw=.4)
        mn, st = 1.2*np.median(Mx, axis=1), 1.98*np.std(Mx, axis=1)
        ax.plot(m.date, mn, color="r", linewidth=0.8, ls="--", label=r"$\beta_m$")
        ax.fill_between(m.date, mn - st, mn + st, color="r", alpha=0.5, label="95% CI")
        X = np.array(X)
        ax.set_xlim(start, end)
        ax.text(0.5, 1.05, r"(b) %s UT, %s @30 MHz, $T_d=\frac{T^{90}}{T^{90}_{base}}$"%(args.event.strftime("%Y-%m-%d"), 
            args.rio.upper()), horizontalalignment="center",
            verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        ax.set_ylim(-.1,2.5)
        ax.legend(loc=1, scatterpoints=3, fontsize=4, ncol=1, frameon=True)
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
        ax.text(0.5, 1.05, "(c) Impact of Temperature on RMSE", horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        fonttext["size"] = 4
        ax.text(TElec[np.argmin(X)], 0.745, r"$T_d$=%.2f"%TElec[np.argmin(X)], horizontalalignment="center",
                verticalalignment="center", fontdict=fonttext, rotation=90)
        fig.autofmt_xdate()
        fig.savefig("_images_/case2.png", bbox_inches="tight")
    return


def _stats_(args):
    """ Estimate and plot statistics """
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prog", default="flare", help="Program code [bgc/flare] (default bgc)")
    parser.add_argument("-r", "--rio", default="ott", help="Riometer code (default ott)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,22), help="Start date (default 2015-3-11T16:22)",
            type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,16), help="Start date (default 2015-3-11T15:30)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,17), help="End date (default 2015-3-11T17:30)",
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
    parser.add_argument("-fr", "--frequency", type=float, default=30, help="Frequency of oprrations in MHz (default 30 MHz)")
    parser.add_argument("-ex", "--exp", type=int, default=0, help="Program code [0-10] (default0)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     " , k , "->" , str(vars(args)[k]))
    if args.exp == 0: _case0_(args)
    if args.exp == 1: _case1_(args)
    if args.exp == 2: _case2_(args)
    if args.exp == 3: _stats_(args)
    else: print("\n Program not implemented")
    print("")
    if os.path.exists("models/__pycache__"): os.system("rm -rf models/__pycache__")
    if os.path.exists("models/experiments/__pycache__"): os.system("rm -rf models/experiments/__pycache__")
