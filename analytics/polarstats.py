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

size = 7
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
df = pd.concat([pd.read_csv("config/skills_X.csv"),pd.read_csv("config/skills_M.csv")])
df = df[df.S_sn>-.3]
for j, nm in enumerate(["sn","avcc","avmb","sw"]):
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    ax = plt.subplot(421+j*2, projection="polar")
    ax.set_rlim(0,1)
    ax.set_theta_zero_location("S")
    r, t, v, rr, tt, Z = parse_2D_data(df, "sza", "local_time", "S_"+nm)
    im = ax.pcolormesh(np.deg2rad(tt), rr, Z, cmap="RdBu", alpha=0.75, vmax=0.6, vmin=-.2, shading="auto")
    txt = labels[j] + "\n" + r"$\bar{\mathcal{S}}_F$=%.3f"%np.mean(v) + "\n" +\
            r"$\mathcal{S}_F\sim$[%.2f,%.2f]"%(np.quantile(v,.025), np.quantile(v,.975))
    ax.text(np.deg2rad(225),1.5, txt, ha="center", va="center", fontdict=fonttext)
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
    ax.text(np.deg2rad(225),1.5, txt, ha="center", va="center", fontdict=fonttext)
    ax.grid(True)
    if j==0: ax.set_title(r"$\mathcal{S}_F\left(MLT, MLAT\right)$",fontdict=fontlabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if j==0: 
        ax.set_xticklabels(["0", 3, "6", 9, "12MLT", 15, "18", 21], fontdict=font)
        ax.set_yticklabels([r"$0.1\pi$", r"$0.2\pi$", r"$0.3\pi$", r"$0.4\pi$", r"$0.5\pi$"], fontdict=font)
cb = fig.colorbar(im, ax=plt.gcf().get_axes(), shrink=0.4)
cb.set_label(r"$\mathcal{S}_F$", fontdict={"size":15})
fig.savefig("../_images_/polar.png", bbox_inches="tight")
