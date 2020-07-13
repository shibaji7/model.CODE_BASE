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

def download_goes_data(dn, sat=15, v=False):
    """ Download GOES data """
    def _get_month_bounds_(start_time):
        """ This method is used to get the first and last date of the month """
        month_start = start_time.replace(day = 1).strftime("%Y%m%d")
        _, month_end = calendar.monthrange(start_time.year, start_time.month)
        month_end = (start_time.replace(day = 1) + dt.timedelta(days=month_end-1)).strftime("%Y%m%d")
        return month_start, month_end
    fname = "data/sim/{dnx}/goes/goes.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
    if not os.path.exists(fname+".gz"):
        month_start, month_end = _get_month_bounds_(dn)
        url = "https://satdat.ngdc.noaa.gov/sem/goes/data/avg/{year}/{month}/goes{sat}/netcdf/"\
                "g{sat}_xrs_1m_{mstart}_{mend}.nc".format(year=dn.year, month="%02d"%dn.month, sat=sat, 
                        mstart=month_start, mend=month_end)
        if v: print("\n Download file -from- " + url)
        tag_vars = ["A_AVG","B_AVG"]
        fn = fname.replace(".csv",".nc")
        os.system("wget -O {fn} {url}".format(fn=fn, url=url))
        if os.path.exists(fn):
            nc = Dataset(fn)
            tt = nc.variables["time_tag"]
            jd = np.array(num2date(tt[:],tt.units))
            data = {}
            for var in tag_vars:  data[var] = nc.variables[var][:]
            data["date"] = jd
            data_dict = pd.DataFrame(data)
            data_dict.to_csv(fname, index=False, header=True)
            os.system("gzip {fname}".format(fname=fname))
            if v: print("\n File saved  -to- " + fname)
            os.remove(fn)
        else: print("\n Unable to download file.")
    return

def download_riometer(dn, stn, v=False):
    """
    This method is used to download riometer absorption data from UCalgary ftp server.
    It stores the dataset into the local drive for future run. It only downloads 1m resolution dataset.
    URL - http://data.phys.ucalgary.ca/sort_by_instrument/riometer/GO-Canada_Rio/txt/
    """
    fname = "data/sim/{dnx}/rio/{stn}.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
    if stn != "ott" and  not os.path.exists(fname+".gz"):
        f_name = "norstar_k2_rio-%s_%s_v01.txt" % (stn, dn.strftime("%Y%m%d"))
        base_url = "http://data.phys.ucalgary.ca/sort_by_instrument/riometer/GO-Canada_Rio/txt"\
                "/{year}/{month}/{day}/".format(year=dn.year, month="%02d"%dn.month, day="%02d"%dn.day)
        uri = base_url + f_name
        tag_vars = ["date","hf_abs"]
        os.system("wget -O {fn} {url}".format(fn=f_name, url=uri))
        if os.path.exists(f_name):
            if v: print("\n Download file -from- " + uri)
            with open(f_name) as c: lines = c.read().split("\n")
            data = []
            for line in lines[13:-2]:
                x = np.nan
                line = list(filter(None,line.replace("\n","").split(" ")))
                try: 
                    x = float(line[2])
                    data.append([dt.datetime.strptime(line[0]+" "+line[1],"%d/%m/%y %H:%M:%S"),x])
                except: continue
            if len(data) > 0: 
                data_dict = pd.DataFrame(data,columns=tag_vars)
                data_dict.to_csv(fname, index=False, header=True)
                os.system("gzip {fname}".format(fname=fname))
                if v: print("\n File saved  -to- " + fname)
            os.remove(f_name)
        else: print("\n Unable to download file.")
    elif stn == "ott" and  not os.path.exists(fname+".gz"):
        f_name = "/home/shibaji/model_run/riometers/ott_{year}-{month}-{day}.csv".format(year=dn.year, 
                month="%02d"%dn.month, day="%02d"%dn.day)
        data_dict = pd.read_csv(f_name, index_col=0)
        data_dict = (data_dict[["DATE","_ABS"]]).rename(columns={"DATE":"date", "_ABS":"hf_abs"})
        data_dict.to_csv(fname, index=False, header=True)
        os.system("gzip {fname}".format(fname=fname))
        if v: print("\n File saved  -to- " + fname)
    return

def get_riom_loc(stn):
    """ This method is to get the location of the riometer """
    _o = pd.read_csv("config/riometers.csv")
    _o = _o[_o.rio==stn]
    lat, lon = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180
    return lat, lon

def read_goes(dn):
    """ This method is used to fetch GOES x-ray data for a given day """
    gzfname = "data/sim/{dnx}/goes/goes.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
    fname = "data/sim/{dnx}/goes/goes.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
    os.system("gzip -d " + gzfname)
    _o = pd.read_csv(fname,parse_dates=["date"])
    os.system("gzip {fname}".format(fname=fname))
    return _o

def read_riometer(dn, stn):
    """ This method is used to fetch riometer absorption data for a given day and station """
    gzfname = "data/sim/{dnx}/rio/{stn}.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
    fname = "data/sim/{dnx}/rio/{stn}.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"), stn=stn)
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
    
    def __init__(self, rio, ev, stime, etime, bins = 37, freq=30, v=False):
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
       
        fname = "data/sim/{dn}/bgc.{stn}.nc.gz".format(dn=self.ev.strftime("%Y.%m.%d.%H.%M"), stn=self.rio)
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
    fname = "data/sim/{dn}/bgc.{stn}.nc.gz".format(dn=ev.strftime("%Y.%m.%d.%H.%M"), stn=rio)
    os.system("gzip -d "+fname)
    rootgrp = Dataset(fname.replace(".gz",""), "a")
    chi = rootgrp.createVariable("chi", "f8", ("ntimes","nalts"))
    chi[:] = calculate_sza(dn, lat, lon, model["alts"])
    chi.description = "Solar Zenith Angle"
    chi.uints = "Deg(o)"
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

    def __init__(self, stn, ev, times,  model, start, end, bar=4.):
        """ Initialize the parameters """
        self.stn = stn
        self.ev = ev
        self.times = times
        self.model = model
        self.start = start
        self.end = end
        self.bar = bar
        self._read_data_()
        self._skill_()
        return

    def _read_data_(self):
        """ Read data from GOES and Riometer """
        gos = read_goes(self.ev)
        rio = read_riometer(self.ev, self.stn)
        self.gos = gos[(gos.date>=self.start) & (gos.date<=self.end)]
        rio = rio[(rio.date>=self.start) & (rio.date<=self.end)]
        self.rio = rio[rio.hf_abs <= self.bar]
        y = np.array(self.gos.B_AVG.tolist())
        yn = (y - np.min(y)) / (np.max(y) - np.min(y))
        self.mx = np.max(self.rio.hf_abs.tolist())
        self.yx = self.mx * yn
        return

    def _skill_(self):
        """ Estimate skills """
        for k in self.model.keys():
            d = pd.DataFrame()
            d["date"], d["hf_abs"] = self.times, self.model[k]
            d = d[(d.date>=self.start) & (d.date<=self.end)]
            e = np.sqrt(np.mean((self.yx-np.array(d.hf_abs))**2))
            print("RMSE -", e)
            print("Err (mx)-", abs(self.mx-np.max(d.hf_abs)))
        return
