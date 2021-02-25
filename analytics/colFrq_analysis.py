#!/usr/bin/env python

"""colFreq_analysis.py: Collision frequency Analysis"""

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
import numpy as np
from netCDF4 import Dataset, num2date
from fetch_data import Simulation

import sys
sys.path.append("models/")
import utils
from absorption import *

if __name__ == "__main__":
    fo = 10**np.linspace(np.log10(.1), np.log10(200), 201)
    
    l, r = 52, 53
    ev, start, end = dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,15,30), dt.datetime(2015,3,11,17,30)
    fname = "config/case0.nc.gz"
    os.system("gzip -d " + fname)
    _nc = Dataset(fname.replace(".gz", ""))
    os.system("gzip " + fname.replace(".gz", ""))
    sim = Simulation(ev, "ott")
    sim.create_remote_local_dir()
    sim.get_bgc_file()
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

    l0 = 4.37e3 * (.22**0.5) / (fo)**2
    l1 = ((10*np.log10(2.2e-4) + 65)/fo)**1.5

    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["mathtext.default"] = "default"
    font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
    fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}
    fig, ax = plt.subplots(figsize=(4,4), nrows=1, ncols=1, dpi=150)

    ax.loglog(fo, l1, "darkred", ls="--", linewidth=0.8, label="DARP")
    ax.set_xlim(1,200)
    ax.set_ylim(1,1e5)
    ax.set_ylabel("Absorption, dB", fontdict=font)
    ax.text(0.5, 1.05, r"$\chi=0^o$, $I_{\infty}=2.2\times 10^{-4}$ $Wm^{-2}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)
    ax.axvline(5, color="k", ls="--", lw=0.8)

    wd = 11
    ax.set_xlabel("Frequency, MHz", fontdict=font)
    ax.loglog(fo, utils.smooth(_lo_[:,0,0], wd), "r", linewidth=1.2, label=r"$\beta_{ah}(\nu_{sn})$")
    ax.loglog(fo, utils.smooth(_lo_[:,1,0], wd), "g", linewidth=0.8, label=r"$\beta_{ah}(\nu_{av}^{cc})$")
    ax.loglog(fo, utils.smooth(_lo_[:,2,0], wd), "b", linewidth=1.2, label=r"$\beta_{ah}(\nu_{av}^{mb})$")
    ax.loglog(fo, utils.smooth(_lo_[:,3,0], wd), "k", linewidth=1.2, label=r"$\beta_{sw}(\nu_{me})$")
    ax.set_ylim(1,1e5)
    ax.set_xlim(1,200)
    ax.set_ylabel("Absorption, dB", fontdict=font)
    ax.legend(loc=1, scatterpoints=3, fontsize=8, frameon=True)
    ax.text(0.5, 1.05, r"$\chi=0^o$, $I_{\infty}=2.2\times 10^{-4}$ $Wm^{-2}$", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict=fonttext)

    fig.savefig("figs/Figure07.png", bbox_inches="tight")