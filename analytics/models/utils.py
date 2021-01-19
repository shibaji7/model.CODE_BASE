#!/usr/bin/env python

"""utils.py: utils is dedicated to utility functions."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import array
import datetime as dt
from netCDF4 import Dataset, num2date
import scipy.integrate as intg
from pysolar.solar import get_altitude
import calendar
import copy
import verify
import xarray
from timezonefinder import TimezoneFinder
from dateutil import tz
import aacgmv2

if sys.version_info.major > 2:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

from collision import *
from absorption import *
from constant import *

def extrap1d(x,y,kind="linear"):
    """ This method is used to extrapolate 1D paramteres """
    interpolator = interp1d(x,y,kind=kind)
    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        if x < xs[0]: return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]: return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else: return interpolator(x)
    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))
    return ufunclike

def get_riom_loc(stn):
    """ This method is to get the location of the riometer """
    _o = pd.read_csv("config/riometers.csv")
    _o = _o[_o.rio==stn]
    lat, lon = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180
    return lat, lon

def read_goes(dn, arc=False):
    """ This method is used to fetch GOES x-ray data for a given day """
    gzfname = "data/tElec/{dnx}/goes/goes.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
    fname = "data/tElec/{dnx}/goes/goes.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
    if arc:
        gzfname = "data/tElec/archive/{dnx}/goes/goes.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
        fname = "data/tElec/archive/{dnx}/goes/goes.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
    os.system("gzip -d " + gzfname)
    _o = pd.read_csv(fname,parse_dates=["date"])
    os.system("gzip {fname}".format(fname=fname))
    return _o

def read_riometer(dn, stn, arc=False):
    """ This method is used to fetch riometer absorption data for a given day and station """
    gzfname = "data/tElec/{dnx}/rio/{stn}.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
    fname = "data/tElec/{dnx}/rio/{stn}.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
    if arc:
        gzfname = "data/tElec/archive/{dnx}/rio/{stn}.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
        fname = "data/tElec/archive/{dnx}/rio/{stn}.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
    if os.path.exists(gzfname): 
        os.system("gzip -d " + gzfname)
        _o = pd.read_csv(fname,parse_dates=["date"])
        os.system("gzip {fname}".format(fname=fname))
    else: _o = pd.DataFrame()
    return _o

def get_height_integrated_absorption(beta, height):
    """ This method is used to calculate height integrated absorption """
    beta[np.isnan(beta)] = 0.
    beta[beta < 0.] = 0.
    beta_L = intg.trapz(beta)
    return beta_L

def calculate_sza(dates, lat, lon, alts):
    """
    This method is used to estimate the solar zenith angle for a specific date and
    sepcific location in space. Note that this method uses skyfield api to estimate 
    solar zenith angle. This has been validated against NOAA website values.
    """
    sza = np.zeros((len(dates), len(alts)))
    for i, d in enumerate(dates):
        for j, a in enumerate(alts):
            d = d.replace(tzinfo=dt.timezone.utc)
            sza[i,j] = 90. - get_altitude(lat, lon, d)
    return sza

class PointGrid(object):
    """
    This class initializes all the parameters for a lat, lon and 0,500 km altitiudes profiles. This is a 2D
    grid for one latitude an longitude X axis time with 1m resolution  Y axis altitude 1km resolution.
    """
    
    def __init__(self, rio, ev, stime, etime, bins = 37, freq=30, v=False, fname="data/sim/{dn}/"):
        self.rio = rio
        self.alts = model["alts"]
        self.start_time = stime
        self.end_time = etime
        self.ev = ev
        self.lat, self.lon = get_riom_loc(rio)
        self.bins = bins
        self.freq = freq
        
        d = int((etime-stime).total_seconds()/60.)
        self.dn = [stime + dt.timedelta(seconds = i*60) for i in range(d)]
       
        fname = (fname + "bgc.{stn}.nc.gz").format(dn=self.ev.strftime("%Y.%m.%d.%H.%M"), stn=self.rio)
        os.system("gzip -d "+fname)
        self._nc = Dataset(fname.replace(".gz", ""))
        os.system("gzip "+fname.replace(".gz", ""))

        self.igrf = {
                "Bx":self._nc.variables["igrf.bx"][:],
                "By":self._nc.variables["igrf.by"][:],
                "Bz":self._nc.variables["igrf.bz"][:],
                "B":self._nc.variables["igrf.b"][:]
                }
        self.iri = {
                "Ne":self._nc.variables["iri.ne"][:],
                "Ni":self._nc.variables["iri.ni"][:],
                "Te":self._nc.variables["iri.te"][:],
                "Ti":self._nc.variables["iri.ti"][:],
                "ions":{
                    "NO+":self._nc.variables["iri.ions.no+"][:],
                    "O+":self._nc.variables["iri.ions.o+"][:],
                    "O2+":self._nc.variables["iri.ions.o2+"][:]
                    }
                }
        self.msis = {
                "Tn":self._nc.variables["msis.tn"][:],
                "rho":self._nc.variables["msis.rho"][:],
                "AR":self._nc.variables["msis.ar"][:],
                "H":self._nc.variables["msis.h"][:],
                "HE":self._nc.variables["msis.he"][:],
                "N2":self._nc.variables["msis.n2"][:],
                "O":self._nc.variables["msis.o"][:],
                "O2":self._nc.variables["msis.o2"][:],
                "O_anomalous":self._nc.variables["msis.o_a"][:],
                "nn":self._nc.variables["msis.nn"][:],

                "NO":self._nc.variables["msis.no"][:],
                "CO":self._nc.variables["msis.co"][:],
                "H2O":self._nc.variables["msis.h2o"][:],
                "CO2":self._nc.variables["msis.co2"][:],
                }
        self.Ne = np.zeros((len(self.dn),len(self.alts)))
        self.chi = self._nc.variables["chi"][:]
        self._col_ = Collision.load(self._nc)
        self._abs_ = Absorption.load(self._nc)
        if v: print("\n Grid point %.2f,%.2f is loaded." % (self.lat,self.lon))
        return

    def update_grid(self, cm, _ix_="all"):
        self.ne = cm.Ne[::60, :]
        self.ni = cm.Np[::60, :]
        self.ni_e = cm.Nm[::60, :]
        self.ni_x = cm.Nxp[::60, :]
        self._abs_ = Absorption(self.igrf["B"], self._col_, self.ne, fo=self.freq*1e6)
        self.drap = Absorption._drap_(self.ev, self.dn, self.rio, self.freq)
        self.sato = Absorption._sato_(self.ev, self.dn, self.rio, self.freq)
        return


def add_chi(ev, rio, start, end):
    """ Add SZA to the Bgc file """
    lat, lon = get_riom_loc(rio)
    d = int((end-start).total_seconds()/60.)
    dn = [start + dt.timedelta(seconds = i*60) for i in range(d)]
    fname = "data/tElec/{dn}/bgc.{stn}.nc.gz".format(dn=ev.strftime("%Y.%m.%d.%H.%M"), stn=rio)
    os.system("gzip -d "+fname)
    rootgrp = Dataset(fname.replace(".gz",""), "a")
    chi = rootgrp.createVariable("chi", "f8", ("ntimes","nalts"))
    chi[:] = calculate_sza(dn, lat, lon, model["alts"])
    chi.description = "Solar Zenith Angle"
    chi.uints = "Deg(o)"
    print(rootgrp.variables.keys())
    rootgrp.close()
    os.system("gzip "+fname.replace(".gz",""))
    return

def extp(x, y, xlim, kind="slinear", scale="log"):
    """ Extrapolate NaN values for smooth outputs. """
    if scale == "log":
        fn = extrap1d(x[x>xlim], np.log10(y[x>xlim]), kind=kind)
        ynew = np.concatenate((10**fn(x[x<=xlim]), y[x>xlim]))
    else:
        fn = extrap1d(x[x>xlim], y[x>xlim], kind=kind)
        ynew = np.concatenate((fn(x[x<=xlim]), y[x>xlim]))
    return ynew

def int_absorption(_a, _h, extpoint=68, llim = 60, ulim = 150, method="trapz"):
    """ Height integrate HF absorption """
    _o = []

    def line_integration(y, x, method="trapz"):
        from scipy.integrate import simps, trapz
        if method == "simps": z = simps(y, x)
        elif method == "trapz": z = trapz(y, x)
        else: z = None
        return z

    for _ix in range(_a.shape[0]):
        _u = pd.DataFrame()
        _u["h"], _u["a"] = _h, extp(_h, _a[_ix,:], xlim=extpoint)
        _u = _u[(_u.h>=llim) & (_u.h<=ulim)]
        _o.append(line_integration(_u["a"], _u["h"], method=method))
    return np.array(_o)

def smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def estimate_error(m, d, kind="rmse"):
    """ Estimate error between model and data """
    xsec = [(x-m.date.tolist()[0]).total_seconds() for x in m.date]
    xnsec =  [(x-m.date.tolist()[0]).total_seconds() for x in d.date]
    dx = interp1d(xsec, m.hf_abs)(xnsec)
    e = np.sqrt(np.mean((dx-np.array(d.hf_abs.tolist()))**2))
    return e

def store_cmd(args):
    """ Store the commands """

    return

class Performance(object):
    """ Class to estimate Skillset """

    def __init__(self, stn, ev, times, model, start, end, bar=4., alt=None):
        """ Initialize the parameters """
        self.stn = stn
        self.ev = ev
        self.times = times
        self.model = model
        self.start = start
        self.end = end
        self.bar = bar
        self.alt = alt
        self._read_data_()
        return

    def _read_data_(self):
        """ Read data from GOES and Riometer """
        gos = read_goes(self.ev, False)
        rio = read_riometer(self.ev, self.stn, False)
        self.gos = gos[(gos.date>=self.start) & (gos.date<self.end)]
        if len(rio) > 0:
            rio = rio[(rio.date>=self.start) & (rio.date<=self.end)]
            if not np.isnan(self.bar): self.rio = rio[rio.hf_abs <= self.bar]
            else: self.rio = rio
        elif np.isnan(self.alt) and not np.isnan(self.bar): self.alt = self.bar
        y = np.array(self.gos.B_AVG.tolist())
        yn = (y - np.min(y)) / (np.max(y) - np.min(y))
        if np.isnan(self.alt): self.mx = np.max(self.rio.hf_abs.tolist())
        else: self.mx = self.alt
        self.yx = self.mx * yn
        return

    def _skill_(self):
        """ Estimate skills """
        self.acc, self.attrs = {}, {}
        dic = {"MSE":"MSE_{r}", "RMSE":"RMSE_{r}", "MAE":"MAE_{r}", "MdAE":"MdAE_{r}",
                "nRMSE":"nRMSE_{r}", "MASE":"MASE_{r}", "MAPE":"MAPE_{r}", "MdAPE":"MdAPE_{r}",
                "MdSymAcc":"MdSymAcc_{r}"}
        self.acc.update({"t": {"dims": ("t"), "data":self.gos.date.tolist()}})
        for k in self.model.keys():
            d = pd.DataFrame()
            d["date"], d["hf_abs"] = self.times, self.model[k]
            d = d[(d.date>=self.start) & (d.date<self.end)]
            self.attrs.update(dict((dic[m].format(r=k), v) for (m,v) in verify.accuracy(np.array(d.hf_abs), self.yx).items()))
            self.attrs.update(dict((dic[m].format(r=k), v) for (m,v) in verify.scaledAccuracy(np.array(d.hf_abs), self.yx).items()))
            self.attrs.update({"mRMSE_" + k: np.sqrt(np.abs(np.max(d.hf_abs)-self.mx))})
            self.attrs.update({"mPeak_" + k: np.max(d.hf_abs)})
            self.acc.update({"e_" + k: {"dims": ("t"), "data": self.yx - np.array(d.hf_abs)}})
            self.acc.update({"m_" + k: {"dims": ("t"), "data": np.array(d.hf_abs)}})
        self.acc.update({"dat": {"dims": ("t"), "data": self.yx}})
        self.attrs.update({"dPeak": self.mx})
        return self

    def _to_mag_(self, times, lat, lon):
        mlats, mlons, mlts = [], [], []
        for t in times:
            mlat, mlon, mlt = aacgmv2.get_aacgm_coord(lat, lon, 100, t, method="TRACE")
            mlats.append(mlat)
            mlons.append(mlon)
            mlts.append(mlt)
        return mlats, mlons, mlts

    def _params_(self):
        """ Extract parameters """
        times = self.gos.date.tolist()
        lat, lon = get_riom_loc(self.stn)
        self.attrs.update({"lat":lat, "lon":lon, "stn": self.stn, "event": self.ev.strftime("%Y.%m.%d.%H.%M")})
        self.acc.update({"sza": {"dims": ("t"), 
            "data": calculate_sza(times, lat, lon, np.array([100])).ravel()}})
        tf = TimezoneFinder()
        from_zone = tz.tzutc()
        to_zone = tz.gettz(tf.timezone_at(lng=lon, lat=lat))
        LT = [t.replace(tzinfo=from_zone).astimezone(to_zone).to_pydatetime() for t in times] 
        now = self.start.replace(tzinfo=from_zone).astimezone(to_zone).to_pydatetime().replace(hour=0,minute=0,second=0)
        LT = [(x - now).total_seconds()/3600. for x in LT]
        self.acc.update({"local_time": {"dims": ("t"), "data": LT}})
        mlats, mlons, mlts = self._to_mag_(times, lat, lon)
        self.acc.update({"mlt": {"dims": ("t"), "data": mlts}})
        self.attrs.update({"mlat": np.mean(mlats)})
        self.attrs.update({"mlon": np.mean(mlons)})
        return self

    def _to_netcdf_(self, fname):
        """ Save to netCDF4 (.nc) file """
        ds = xarray.Dataset.from_dict(self.acc)
        ds.attrs = self.attrs
        print("---------------------Skills----------------------")
        print(ds)
        print("-------------------------------------------------")
        ds.to_netcdf(fname,mode="w")
        return
