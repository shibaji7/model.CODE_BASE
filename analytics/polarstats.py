import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#plt.style.use("config/alt.mplstyle")
import numpy as np
import pandas as pd
import xarray
import datetime as dt
from scipy.stats import kendalltau
from scipy.optimize import curve_fit
import scipy.stats as stats

size = 10
from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=size, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = size
matplotlib.rcParams["ytick.labelsize"] = size
matplotlib.rcParams["mathtext.default"] = "default"
font = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": size}
fonttext = {"family": "serif", "color":  "red", "weight": "normal", "size": size}
fontlabel = {"family": "serif", "color":  "darkgreen", "weight": "normal", "size": size*1.5}

case = 1

if case == 1:
    def parse_2D_data(q, r, t, zv, k=0):
        """
        Method converts scans to "beam" and "slist" or gate
        """
        plotParamDF = q[ [r, t, zv] ]
        _r, _t = np.arange(0,90,9), np.arange(0,24,1)
        rr, tt = np.meshgrid(_r, _t)
        Z = np.zeros_like(rr)*np.nan
        for j in range(len(_r)-1):
            for i in range(len(_t)-1):
                dat = plotParamDF[(plotParamDF[r] >= _r[j]) & (plotParamDF[r] < _r[j+1])
                        & (plotParamDF[t] >= _t[i]) & (plotParamDF[t] < _t[i+1])]
                if len(dat) > 0: Z[i,j] = dat[zv].mean()

        _r, _t = np.arange(0,1,1/len(_r)), np.arange(0,360,360./len(_t))
        rr, tt = np.meshgrid(_r, _t)
        u = pd.DataFrame(np.array([rr.ravel(), tt.ravel(), Z.ravel()]).T, columns=["r","t","z"])
        u.z = u.z + 0.2
        u = u.dropna()
        x = u.groupby(by="r").mean().reset_index()
        print(r, np.round(kendalltau(x.r, x.z), 2))
        #print(t, np.round(kendalltau(u.t, u.z),2))
        #print(r, np.round(kendalltau(u.r, u.z),2))
        return u.r.tolist(), u.t.tolist(), u.z.tolist(), rr, tt, Z


    fig = plt.figure(dpi=180, figsize=(5,10))
    labels = [r"$\beta_{ah}(\nu_{sn})$", r"$\beta_{ah}(\nu^{cc}_{av})$",
                            r"$\beta_{ah}(\nu^{mb}_{av})$", r"$\beta_{sw}(\nu_{me})$"]
    for j, nm in enumerate(["sn","avcc","avmb","sw"]):
        df = pd.concat([pd.read_csv("config/skills_X.csv"),pd.read_csv("config/skills_M.csv")])
        df = df[df["S_"+nm]>-.3]
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        ax = plt.subplot(421+j*2, projection="polar")
        ax.set_rlim(0,1)
        ax.set_theta_zero_location("S")
        r, t, v, rr, tt, Z = parse_2D_data(df, "sza", "local_time", "S_"+nm)
        im = ax.pcolormesh(np.deg2rad(tt), rr, Z, cmap="RdBu", alpha=0.75, vmax=0.6, vmin=-.2, shading="auto")
        txt = labels[j] + "\n" + r"$\bar{\mathcal{S}}_F$=%.3f"%np.mean(v) + "\n" +\
                r"$\lbrace\mathcal{S}_F\rbrace\sim$[%.2f,%.2f]"%(np.quantile(v,.025), np.quantile(v,.975))
        ax.text(np.deg2rad(225),1.8, txt, ha="center", va="center", fontdict=fonttext)
        ax.grid(True)
        if j==0: ax.set_title(r"$\mathcal{S}_F\left(LT, SZA\right)$",fontdict=fontlabel)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if j==0: 
            ax.set_xticklabels(["0", 3, "6", 9, "12LT", 15, "18", 21], fontdict=font)
            ax.set_yticklabels([r"$0.1\pi$", r"$0.2\pi$", r"$0.3\pi$", r"$0.4\pi$", r"$0.5\pi$"], fontdict=font)

        ax = plt.subplot(421+j*2+1, projection="polar")
        ax.set_rlim(0,1)
        ax.set_theta_zero_location("S")
        r, t, v, rr, tt, Z = parse_2D_data(df, "mlat", "mlt", "S_"+nm)
        im = ax.pcolormesh(np.deg2rad(tt), rr, Z, cmap="RdBu", alpha=0.75, vmax=.6, vmin=-.2, shading="auto")
        txt = labels[j] + "\n" + r"$\bar{\mathcal{S}}_F$=%.3f"%np.mean(v) + "\n" +\
                r"$\mathcal{S}_F\sim$[%.2f,%.2f]"%(np.quantile(v,.025), np.quantile(v,.975))
        ax.grid(True)
        if j==0: ax.set_title(r"$\mathcal{S}_F\left(MLT, MLAT\right)$",fontdict=fontlabel)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if j==0: 
            ax.set_xticklabels(["0", 3, "6", 9, "12MLT", 15, "18", 21], fontdict=font)
            ax.set_yticklabels([r"$0.1\pi$", r"$0.2\pi$", r"$0.3\pi$", r"$0.4\pi$", r"$0.5\pi$"], fontdict=font)
    cb = fig.colorbar(im, ax=plt.gcf().get_axes(), shrink=0.4)
    cb.set_label(r"$\mathcal{S}_F$", fontdict={"size":15})
    fig.savefig("figs/Figure061.png", bbox_inches="tight")

case = 2
if case == 2:
    def analysis(func, ax, df, nx="sza", ny="S_sn",bins=None, col="k", mul=0):
        fonttext = {"family": "serif", "color":  "darkgreen", "weight": "normal", "size": 6}
        df = df.sort_values(by=nx)
        x, y = [], {"ymean":[], "yerr":[]}
        for i in range(len(bins)-1):
            avg = df[(df[nx]>=bins[i]) & (df[nx]<bins[i+1])]
            if len(avg) > 0:
                x.append(0.5*(bins[i]+bins[i+1]))
                y["ymean"].append(np.median(avg[ny]) + mul)
                y["yerr"].append(np.median(np.abs(avg[ny]-np.median(avg[ny]))))
        u = pd.DataFrame()
        u[nx], u[ny] = x, y["ymean"]
        u = u[u[ny] > 0]
        popt, _ = curve_fit(func, u[nx], u[ny])
        linestyle = {"linestyle":"None", "linewidth":0.5, "markeredgewidth":0.8, "elinewidth":0.6, "capsize":1, "marker": "o", "ms":1.5}
        ax.errorbar(x, y["ymean"], yerr=np.array(y["yerr"])*0.6, color=col, **linestyle)
        ax.axhline(0, color="k", ls="--", lw="0.8")
        ax.plot(x, func(np.array(x), *popt), color=col, lw=0.6)
        if col=="r": ax.text(0.8, 1.05, r"$\tau_{X}$=%.2f"%np.round(kendalltau(u[ny], func(u[nx], *popt)),2)[0],
                             ha="center", va="center", fontdict={"color":"red", "size":8}, transform=ax.transAxes)
        if col=="b": ax.text(0.2, 1.05, r"$\tau_{M}$=%.2f"%np.round(kendalltau(u[ny], func(u[nx], *popt)),2)[0], 
                             ha="center", va="center", fontdict={"color":"blue", "size":8}, transform=ax.transAxes)
        return


    fig1, axes1 = plt.subplots(figsize=(8, 8), nrows=4, ncols=4, dpi=150, sharey="row", sharex="col")
    for cls, col, mul, peaks in zip(["X", "M"], ["r", "b"], [0.08, 0.1], [[12,10],[15,15]]):
        for j, nm in enumerate(["sn","avcc","avmb","sw"]):
            df = pd.concat([pd.read_csv("config/skills_%s.csv"%cls)])
            df = df[df["S_"+nm]>-.3]
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            fonttext["color"] = "k"
            ax = axes1[j, 0]
            analysis(lambda x, a, b: np.exp(a-b/np.cos(np.deg2rad(x))), ax, df, nx="sza", ny="S_"+nm, bins=np.arange(0,90,3), col=col,
                     mul=mul)
            ax = axes1[j, 1]
            analysis(lambda x, a, b: a*np.cos(0.5*np.pi*((x-peaks[0])/12)), ax, df, nx="local_time", 
                     ny="S_"+nm, bins=np.arange(0,24,1), col=col, mul=mul)
            ax = axes1[j, 2]
            analysis(lambda x, a, b: a*np.cos(0.5*np.pi*((x-peaks[1])/12)), ax, df, nx="mlt", ny="S_"+nm, bins=np.arange(0,24,1), col=col,
                     mul=mul)
            ax = axes1[j, 3]
            analysis(lambda x, a, b: np.exp(a-b/np.cos(np.deg2rad(x))), ax, df, nx="mlat", ny="S_"+nm, bins=np.arange(0,90,5), col=col,
                     mul=mul)
            ax.text(1.07, 0.5, labels[j], ha="center", va="center", fontdict={"color":"k", "size":10}, 
                    transform=ax.transAxes, rotation=90)
    
    font["color"] = "k"
    axes1[3,0].set_xlabel(r"SZA, $\chi(^o)$", fontdict=font)
    axes1[3,1].set_xlabel(r"LT, Hours", fontdict=font)
    axes1[3,2].set_xlabel(r"MLT, Hours", fontdict=font)
    axes1[3,3].set_xlabel(r"MLAT, $Deg(^o)$", fontdict=font)
    axes1[0,0].set_ylim(-0.1,0.7)
    axes1[1,0].set_ylim(-0.1,0.7)
    axes1[2,0].set_ylim(-0.1,0.7)
    axes1[3,0].set_ylim(-0.1,0.7)
    font["size"] = size*1.5
    fig1.text(0.01, 0.4, r"$\mathcal{S}_F = 1-\frac{RMSE_{model}}{RMSE_{DRAP}}$", fontdict=font, rotation=90)
    fig1.savefig("figs/Figure06.png", bbox_inches="tight")