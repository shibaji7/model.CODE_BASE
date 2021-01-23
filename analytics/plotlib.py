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
#plt.style.use("config/alt.mplstyle")
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import datetime as dt
import aacgmv2

np.set_printoptions(formatter={"float_kind": lambda x:"{:.2f}".format(x)})

def plot_rio_locations():
    """ Plot all riometers used for the study """
    fig = plt.figure(figsize=(5,9),dpi=180)
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

if __name__ == "__main__":
    ###############################################
    # Run one time plots for final use
    ###############################################
    plot_rio_locations()
    pass