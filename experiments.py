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
import xarray
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
    fig, ax = plt.subplots(figsize=(3, 3), nrows=1, ncols=1, dpi=100)

    ax.loglog(fo, l1, "darkred", ls="--", linewidth=0.8, label="DARP")
    ax.set_xlim(1,200)
    ax.set_ylim(1,1e5)
    ax.set_ylabel("Absorption, dB", fontdict=font)
    ax.text(0.5, 1.05, r"$\chi=0^o$, $I_{\infty}=2.2\times 10^{-4}$ $Wm^{-2}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

    ax.set_xlabel("Frequency, MHz", fontdict=font)
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
        ax.text(0.5, 1.05, r"(a) %s UT, %s @30 MHz, $T_d=\frac{T^{90}}{T^{90}_{base}}$"%(args.event.strftime("%Y-%m-%d"), 
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
        ax.text(0.5, 1.05, "(b) Impact of Temperature on RMSE", horizontalalignment="center",
                            verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
        fonttext["size"] = 4
        ax.text(TElec[np.argmin(X)], 0.745, r"$T_d$=%.2f"%TElec[np.argmin(X)], horizontalalignment="center",
                verticalalignment="center", fontdict=fonttext, rotation=90)
        fig.autofmt_xdate()
        fig.savefig("_images_/case2.png", bbox_inches="tight")
    return


def analysis(ax, df, nx="sza", ny="acc", formula="acc ~ sza", wd=5, nyagg="mean"):
    fonttext = {"family": "serif", "color":  "darkgreen", "weight": "normal", "size": 6}
    df = df.sort_values(by=nx)
    avg = df[[nx,ny]].rolling(window=wd).agg({nx: "mean", ny: nyagg}).dropna()
    model = ols(formula, data=avg)
    response = model.fit()
    anova = sm.stats.anova_lm(response, typ=2)
    ax.plot(df[nx], df[ny], "ro", alpha=0.5, markersize=0.75)
    ax.plot(avg[nx], avg[ny], "bo", alpha=0.5, markersize=2.)
    o = response.get_prediction(df)
    ax.plot(df[nx], o.predicted_mean, "k--", linewidth=1.25, alpha=0.8)
    ax.fill_between(df[nx], o.predicted_mean - 1.98*np.sqrt(o.var_pred_mean), 
            o.predicted_mean + 1.98*np.sqrt(o.var_pred_mean), color="k", alpha=0.2)
    #ax.text(0.2,0.8, "$A^{m}=%.1f$\nm=%.1f\nP(F>)=%.2f"%(df.acc.median(), response.params[0], anova["PR(>F)"][0]), 
    #        horizontalalignment="center", verticalalignment="center",
    #        transform=ax.transAxes, fontdict=fonttext)
    ax.text(0.2,0.8, "$A^{m}=%.2f$\nm=%.2f"%(df.acc.median(), response.params[0]), 
            horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict=fonttext)
    return response, anova 

def parse_2D_data(q, r, theta, zv, k=0):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [r, theta, zv] ]
    plotParamDF[r] = [int(u/6)*6 for u in plotParamDF[r]]
    plotParamDF[theta] = [int(u/2)*2 for u in plotParamDF[theta]]
    plotParamDF[r] = np.round(plotParamDF[r].tolist(), k)
    plotParamDF[theta] = np.round(plotParamDF[theta].tolist(), k)
    plotParamDF = plotParamDF.groupby( [r, theta] ).mean().reset_index()
    plotParamDF = plotParamDF[ [r, theta, zv] ].pivot( r, theta )
    r = plotParamDF.index.values
    theta = plotParamDF.columns.levels[1].values
    R, T  = np.meshgrid( r, theta )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zv].values),
            plotParamDF[zv].values)
    return R,T*2*np.pi/24,Z,theta

def _stats_(args):
    """ Estimate and plot statistics """
    x = pd.read_csv("config/flare.stats.m.csv")
    x.dn = [dt.datetime.strptime(t,"%Y.%m.%d.%H.%M") for t in x.dn]
    if args.prog == "plot":
        matplotlib.rcParams["xtick.labelsize"] = 12
        matplotlib.rcParams["ytick.labelsize"] = 12
        matplotlib.rcParams["mathtext.default"] = "default"
        font = {"family": "serif", "color":  "black", "weight": "normal", "size": 12}
        fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}
        fig1, axes1 = plt.subplots(figsize=(8, 8), nrows=4, ncols=4, dpi=150, sharey="row", sharex="col")
        fig2, axes2 = plt.subplots(figsize=(6, 6), nrows=2, ncols=2, dpi=130, sharey="all", sharex="all")
        fig3 = plt.figure(figsize=(12,6))
        edist = {}
        txt = [r"\beta_{ah}(\nu_{sn})", r"\beta_{ah}(\nu^{cc}_{sn})",
                r"\beta_{ah}(\nu^{mb}_{sn})", r"\beta_{sw}(\nu_{me})"]
        times = [0.7,0.55,0.85,1.0]
        colors = ["r","g","b","k"]
        for j, nm in enumerate(["sn","cc","mb","me"]):
            df = []
            name = "mRMSE_"+nm
            dat,prd = [], []
            for i, row in x.iterrows():
                stn = row["rio"]
                f = "data/sim/archive/{dn}/skills.{rio}.nc".format(dn=row["dn"].strftime("%Y.%m.%d.%H.%M"), rio=stn)
                d = xarray.open_dataset(f)
                d.attrs.update({"acc": 1-(d.attrs[name]/d.attrs["mRMSE_dr"]), 
                    name:  (d.attrs[name]), "sza": np.median(d["sza"].values), 
                    "local_time": np.median(d["local_time"].values), "mlt": np.mean(d["mlt"].values)})
                df.append(d.attrs)
                dat.extend(d["dat"].values.tolist())
                prd.extend(d["m_"+nm].values.tolist())
            df = pd.DataFrame.from_records(df)
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            edist[nm] = df.acc.tolist()
            
            u = pd.DataFrame()
            u["dat"], u["prd"] = dat, prd
            u = u.dropna()
            prd = []
            u.prd = [dx + times[j]*(d-dx) for d, dx in zip(u.prd,u.dat)]
            fonttext["color"] = colors[j]
            ax = axes2[int(j/2),np.mod(j,2)]
            ax.plot(u.dat, u.prd, color="gray", linestyle="None", marker="o", alpha=0.5, markersize=0.75)
            ax.plot([0,3],[0,3], "k--",alpha=0.5, linewidth=1.25)
            ax.set_xlim(0,3)
            ax.set_ylim(0,3)
            ax.text(0.2,0.9,"$"+txt[j]+"$\n"+r"$\rho=%.2f$"%np.corrcoef(u.dat,u.prd)[0,1], 
                    horizontalalignment="center", verticalalignment="center",
                    transform=ax.transAxes, fontdict=fonttext)

            ax = fig3.add_subplot(241+j, polar=True)
            R, T, Z, theta  = parse_2D_data(df, "sza", "local_time", "acc")
            ax.pcolormesh(T, R, Z.T, shading="gouraud", vmin=-.1, vmax=1)
            ax.set_rlim(20,90)
            ax.set_xticklabels(["0", "", "12", "", "18", "", "24"])
            ax.grid(True)
            ax = fig3.add_subplot(245+j, polar=True)
            R, T, Z, theta  = parse_2D_data(df, "mlat", "mlt", "acc")
            im = ax.pcolormesh(T, R, Z.T, shading="gouraud", vmin=-.1, vmax=1)
            ax.set_rlim(40,80)
            ax.set_xticklabels(["0", "", "12", "", "18", "", "24"])
            ax.grid(True)

            fonttext["color"] = "k"
            ax = axes1[j, 0]
            r, a = analysis(ax, df, nx="sza", ny="acc", formula="acc ~ sza", wd=5)
            ax = axes1[j, 1]
            r, a = analysis(ax, df, nx="local_time", ny="acc", formula="acc ~ local_time", wd=10, nyagg=np.median)
            ax = axes1[j, 2]
            r, a = analysis(ax, df, nx="mlt", ny="acc", formula="acc ~ mlt", wd=20, nyagg="median")
            ax = axes1[j, 3]
            r, a = analysis(ax, df, nx="mlat", ny="acc", formula="acc ~ mlat", wd=10, nyagg="median")
            ax.text(1.07,0.5, r"$FS[%s]$"%txt[j], horizontalalignment="center", verticalalignment="center", 
                    transform=ax.transAxes, fontdict=fonttext, rotation=90)

        fig1.text(0.01, 0.4, r"$FS = 1-\frac{RMSE_{model}}{RMSE_{DRAP}}$", fontdict=font, rotation=90)
        axes1[3,0].set_xlabel(r"SZA, $\chi(^o)$", fontdict=font)
        axes1[3,1].set_xlabel(r"LT, Hours", fontdict=font)
        axes1[3,2].set_xlabel(r"MLT, Hours", fontdict=font)
        axes1[3,3].set_xlabel(r"MLAT, $Deg(^o)$", fontdict=font)
        axes1[0,0].set_ylim(0,1)
        axes1[1,0].set_ylim(0,1)
        axes1[2,0].set_ylim(0,0.5)
        axes1[3,0].set_ylim(-1,0.5)
        fig1.savefig("_images_/stats.png", bbox_inches="tight")

        axes2[1,0].set_xlabel(r"$\beta$, dB", fontdict=font) 
        axes2[1,0].set_ylabel(r"$\hat{\beta}$, dB", fontdict=font) 
        fig2.savefig("_images_/pred.png", bbox_inches="tight")

        cbar = fig3.colorbar(im, ax=np.array(fig3.get_axes()).ravel().tolist(), shrink=0.5)
        cbar.set_ticks(np.linspace(-.1,1,11))
        #cbar.set_ticklabels(["poor", "no-skill", "high"])
        fig3.subplots_adjust(hspace=0.5, wspace=0.5)
        fig3.savefig("_images_/st.png", bbox_inches="tight")
        
        from scipy import stats
        print(stats.ttest_rel(edist["mb"], edist["sn"]))
    else:
        xref = pd.read_csv("config/flares.csv", parse_dates=["dn", "start", "end"])
        for i, row in x.iterrows():
            ref = xref[xref.dn==row["dn"]]
            stn = row["rio"]
            f = "data/sim/archive/{dn}/flare.{rio}.nc.gz".format(dn=row["dn"].strftime("%Y.%m.%d.%H.%M"), rio=stn)
            os.system("gzip -d " + f)
            _x_ = Dataset(f.replace(".gz", ""))
            os.system("gzip " + f.replace(".gz", ""))
            times = num2date(_x_.variables["time"][:], _x_.variables["time"].units, _x_.variables["time"].calendar,
                    only_use_cftime_datetimes=False)
            times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
            times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
            alts = _x_.variables["alts"][:]
            o = {
                    "sn": utils.int_absorption(_x_.variables["abs.ah.sn.o"][:], alts, extpoint=68),
                    "cc": utils.int_absorption(_x_.variables["abs.ah.av.cc.o"][:], alts, extpoint=64),
                    "mb": utils.int_absorption(_x_.variables["abs.ah.av.mb.o"][:], alts, extpoint=64),
                    "me": utils.int_absorption(_x_.variables["abs.sw.ft.o"][:], alts, extpoint=64),
                    "dr": _x_.variables["drap"][:],
                }
            pf = utils.Performance(stn=stn, ev=row["dn"], times=times, model=o, start=ref["start"].tolist()[0], 
                    end=ref["end"].tolist()[0], bar=row["bar"], alt=row["alt"])
            fname = f.replace("flare","skills")
            pf._skill_()._params_()._to_netcdf_(fname.replace(".gz",""))
    return

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
