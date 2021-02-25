"""bgc.py: bgc is used to manage and run the background model and store to .nc.gz file"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np

np.random.seed(0)
import dask

import datetime as dt
import argparse
from dateutil import parser as dparser
import pandas as pd
from pyglow import Point as P
from netCDF4 import Dataset, date2num
import time
from scipy.interpolate import interp1d
from pysolar.solar import get_altitude
import pytz

import sys
sys.path.append("models/")
from collision import *
from absorption import *
from constant import *
import utils
from fetch_data import Simulation

def get_riom_loc(stn):
    """ This method is to get the location of the riometer """
    _o = pd.read_csv("config/riometers.csv")
    _o = _o[_o.rio==stn]
    lat, lon = _o["lat"].tolist()[0], np.mod( (_o["lon"].tolist()[0] + 180), 360 ) - 180
    return lat, lon

dirc = "tElec" # tElec, sim
class Bgc(object):
    """ Bgc class is dedicated to run in pyton 2.7 and calculate and store the background ionosphere """

    def __init__(self, rio, ev, lat, lon, stime, etime, freq):
        """ Initialize all patameters """
        self.rio = rio
        self.alts = model["alts"]
        self.ev = ev
        self.start_time = stime.replace(tzinfo=pytz.utc)
        self.end_time = etime.replace(tzinfo=pytz.utc)
        self.lat = lat
        self.lon = lon
        self.species = ["O2","N2","O","NO","CO","CO2","H2O"]
        
        self.sim = Simulation(self.ev, self.rio)
        if self.sim.check_riometer_data_exists() and self.sim.check_bgc_not_exists():
        #if self.sim.check_bgc_not_exists():
            self.sim.create_remote_local_dir()

            self.fname = "proc/outputs/{dn}/{stn}/bgc.nc.gz".format(dn=self.ev.strftime("%Y.%m.%d.%H.%M"), stn=self.rio)
            if os.path.exists(self.fname): os.remove(self.fname)
            self.ncfile = self.fname.replace(".gz","")
        
            d = int((etime-stime).total_seconds()/60.)
            self.dn = [stime + dt.timedelta(seconds = i*60) for i in range(d)]
        
            self.igrf = {
                    "Bx":np.zeros((len(self.dn),len(self.alts))),
                    "By":np.zeros((len(self.dn),len(self.alts))),
                    "Bz":np.zeros((len(self.dn),len(self.alts))),
                    "B":np.zeros((len(self.dn),len(self.alts)))
                    }
            self.iri = {
                    "Ne":np.zeros((len(self.dn),len(self.alts))),
                    "Ni":np.zeros((len(self.dn),len(self.alts))),
                    "Te":np.zeros((len(self.dn),len(self.alts))),
                    "Ti":np.zeros((len(self.dn),len(self.alts))),
                    "ions":{
                        "NO+":np.zeros((len(self.dn),len(self.alts))),
                        "O+":np.zeros((len(self.dn),len(self.alts))),
                        "O2+":np.zeros((len(self.dn),len(self.alts))),
                        }
                    }
            self.chi = np.zeros((len(self.dn),len(self.alts)))
            self.msis = {
                    "Tn":np.zeros((len(self.dn),len(self.alts))),
                    "rho":np.zeros((len(self.dn),len(self.alts))),
                    "AR":np.zeros((len(self.dn),len(self.alts))),
                    "H":np.zeros((len(self.dn),len(self.alts))),
                    "HE":np.zeros((len(self.dn),len(self.alts))),
                    "N2":np.zeros((len(self.dn),len(self.alts))),
                    "O":np.zeros((len(self.dn),len(self.alts))),
                    "O2":np.zeros((len(self.dn),len(self.alts))),
                    "O_anomalous":np.zeros((len(self.dn),len(self.alts))),
                    "nn": np.zeros((len(self.dn),len(self.alts))),
                    
                    "NO": np.zeros((len(self.dn),len(self.alts))),
                    "CO": np.zeros((len(self.dn),len(self.alts))),
                    "H2O": np.zeros((len(self.dn),len(self.alts))),
                    "CO2": np.zeros((len(self.dn),len(self.alts))),
                    }
            results = []
            _dir_ = os.path.dirname(os.path.abspath(__file__))
            for I,u in enumerate(self.dn):
                self.msis["H2O"][I,:] = self._get_params_("H2O")
                self.msis["NO"][I,:] = self._get_params_("NO")
                self.msis["CO"][I,:] = self._get_params_("CO")
                self.msis["CO2"][I,:] = self._get_params_("CO2")
                for J,h in enumerate(self.alts):
                    self.populate_grid(I, J, u, h, lat, lon)
                    #results.append(result)
            #dask.compute(*results)
            self._col_ = Collision(self.msis, self.iri, self.iri["Ne"], self.iri["Te"], self.iri["Ti"])
            self._abs_ = Absorption(self.igrf["B"], self._col_, self.iri["Ne"], fo=freq*1e6)
            print("\n Grid point %.2f,%.2f is downloaded." % (lat,lon))
            os.chdir(_dir_)
            self.save()
        return

    def populate_grid(self, I, J, u, h, lat, lon):
        p = P(u,lat,lon,h)
        p.run_iri()
        p.run_msis()
        p.run_igrf()
        self.chi[I,J] = utils.calculate_sza([u], lat, lon, [h])[0,0]
        self.igrf["Bx"][I,J] = p.Bx # in Tesla
        self.igrf["By"][I,J] = p.By # in Tesla
        self.igrf["Bz"][I,J] = p.Bz # in Tesla
        self.igrf["B"][I,J] = p.B # in Tesla
        self.iri["Ne"][I,J] = p.ne * 1e6 # in m^-3
        self.iri["Ni"][I,J] = np.nansum(list(p.ni.values())) * 1e6 # in m^-3
        
        self.iri["ions"]["NO+"][I,J] = p.ni["NO+"] * 1e6 # in m^-3
        self.iri["ions"]["O2+"][I,J] = p.ni["O2+"] * 1e6 # in m^-3
        self.iri["ions"]["O+"][I,J] = p.ni["O+"] * 1e6 # in m^-3
        
        self.iri["Te"][I,J] = p.Te # in kelvin K
        self.iri["Ti"][I,J] = p.Ti # in kelvin K
        self.msis["Tn"][I,J] = p.Tn_msis # in kelvin K
        self.msis["rho"][I,J] = p.rho # in gm/cm^-3
        self.msis["AR"][I,J] = p.nn["AR"] * 1e6 # in m^-3
        self.msis["H"][I,J] = p.nn["H"] * 1e6 # in m^-3
        self.msis["HE"][I,J] = p.nn["HE"] * 1e6 # in m^-3
        self.msis["N2"][I,J] = p.nn["N2"] * 1e6 # in m^-3
        self.msis["O"][I,J] = p.nn["O"] * 1e6 # in m^-3
        self.msis["O2"][I,J] = p.nn["O2"] * 1e6 # in m^-3
        self.msis["O_anomalous"][I,J] = p.nn["O_anomalous"] * 1e6 # in m^-3
        nn = (p.nn["AR"] + p.nn["H"] + p.nn["HE"] + p.nn["N"] \
              + p.nn["N2"] + p.nn["O"] + p.nn["O2"] + p.nn["O_anomalous"]) * 1e6 # in m^-3
        self.msis["nn"][I,J] = nn # in m^-3
        print(" Done Params (u, h, lat, lon):", u, h, lat, lon)
        return 0
    
    def _get_params_(self, sp):
        """ Fetch minor species from other system """
        fname = "config/f.e20.FXSD.f19_f19.001.cam.h0.2000-01.nc"
        _nc = Dataset(fname)
        T = _nc.variables["T"][0,:,0,0]
        h = (_nc.variables["Z3"][0,:,0,0]*pconst["Re"]/(pconst["Re"]-_nc.variables["Z3"][0,:,0,0])) / 1.e3
        P = _nc.variables["lev"][:] * 1e2
        minor_species = ["NO","CO","H2O","CO2"]
        ion_species = ["O2+","N2+","O+","NO+"]
        na = P * pconst["avo"] / (pconst["R"] * T)
        _m = _nc.variables[sp][0,:,0,0] * na
        _o = interp1d(h,_m)(self.alts)
        return _o

    def save(self):
        """ Save the file to local store """

        def _set_(key, val, desc, units, format="f8", shape=("ntimes","nalts")):
            p = self.rootgrp.createVariable(key,format, shape)
            p.description = desc
            p.uints = units
            p[:] = val
            return

        self.rootgrp = Dataset(self.ncfile, "w", format="NETCDF4")
        self.rootgrp.description = "HF Absorption Model: Background Ionosphere (R:{rio})".format(rio=self.rio)
        self.rootgrp.history = "Created " + time.ctime(time.time())
        self.rootgrp.source = "SuperDARN HF Absorption Model"
        self.rootgrp.createDimension("nalts", len(self.alts))
        self.rootgrp.createDimension("ntimes", len(self.dn))
        alts = self.rootgrp.createVariable("alts","f4",("nalts",))
        alts.description = "Altitude values, in km"
        alts[:] = self.alts
        times = self.rootgrp.createVariable("time", "f8", ("ntimes",))
        times.units = "hours since 1970-01-01 00:00:00.0"
        times.calendar = "julian"
        times[:] = date2num(self.dn,units=times.units,calendar=times.calendar)
        _set_("chi", self.chi, "Solar Zenith Angle", "Deg(o)")
        _set_("igrf.bx", self.igrf["Bx"], "IGRF Bx component height-time profile", "T")
        _set_("igrf.by", self.igrf["By"], "IGRF By component height-time profile", "T")
        _set_("igrf.bz", self.igrf["Bz"], "IGRF Bz component height-time profile", "T")
        _set_("igrf.b", self.igrf["B"], "IGRF |B| component height-time profile", "T")
        _set_("iri.ne", self.iri["Ne"], "IRI Ne height-time profile", "m^-3")
        _set_("iri.ni", self.iri["Ni"], "IRI Ni height-time profile", "m^-3")
        _set_("iri.te", self.iri["Te"], "IRI Te height-time profile", "K")
        _set_("iri.ti", self.iri["Ti"], "IRI Ti height-time profile", "K")
        _set_("iri.ions.no+", self.iri["ions"]["NO+"], "IRI NO+ height-time profile", "m^-3")
        _set_("iri.ions.o2+", self.iri["ions"]["O2+"], "IRI O2+ height-time profile", "m^-3")
        _set_("iri.ions.o+", self.iri["ions"]["O+"], "IRI O+ height-time profile", "m^-3")
        _set_("msis.tn", self.msis["Tn"], "MSIS Tn height-time profile", "K")
        _set_("msis.rho", self.msis["rho"], "MSIS density height-time profile", "gm/cm^-3")
        _set_("msis.ar", self.msis["AR"], "MSIS Argon height-time profile", "m^-3")
        _set_("msis.h", self.msis["H"], "MSIS Hydrogen height-time profile", "m^-3")
        _set_("msis.he", self.msis["HE"], "MSIS Helium height-time profile", "m^-3")
        _set_("msis.n2", self.msis["N2"], "MSIS Nitogen height-time profile", "m^-3")
        _set_("msis.o", self.msis["O"], "MSIS Oxigen-atom height-time profile", "m^-3")
        _set_("msis.o2", self.msis["O2"], "MSIS Oxigen-molecule height-time profile", "m^-3")
        _set_("msis.o_a", self.msis["O_anomalous"], "MSIS Anomalous-oxigen height-time profile", "m^-3")
        _set_("msis.nn", self.msis["nn"], "MSIS density height-time profile", "m^-3")
        _set_("msis.no", self.msis["NO"], "MSIS NO height-time profile", "m^-3")
        _set_("msis.co", self.msis["CO"], "MSIS CO height-time profile", "m^-3")
        _set_("msis.h2o", self.msis["H2O"], "MSIS H2O height-time profile", "m^-3")
        _set_("msis.co2", self.msis["CO2"], "MSIS CO2 height-time profile", "m^-3")
        _set_("col.ft", self._col_.nu_FT, "Collision (FT) height-time profile", "Hz")
        _set_("col.av.cc", self._col_.nu_av_CC, "Collision (Avraged - CC) height-time profile", "Hz")
        _set_("col.av.mb", self._col_.nu_av_MB, "Collision (Avraged - MB) height-time profile", "Hz")
        _set_("col.av.sn", self._col_.nu_SN["total"], "Collision (Avraged - SN) height-time profile", "Hz")
        _set_("col.av.sn.en", self._col_.nu_SN["en"]["total"], "Collision (Avraged - sn.en) height-time profile", "Hz")
        _set_("col.av.sn.en.n2", self._col_.nu_SN["en"]["N2"], "Collision (Avraged - sn.n2) height-time profile", "Hz")
        _set_("col.av.sn.en.o", self._col_.nu_SN["en"]["O"], "Collision (Avraged - sn.o) height-time profile", "Hz")
        _set_("col.av.sn.en.he", self._col_.nu_SN["en"]["He"], "Collision (Avraged - sn.he) height-time profile", "Hz")
        _set_("col.av.sn.en.o2", self._col_.nu_SN["en"]["O2"], "Collision (Avraged - sn.o2) height-time profile", "Hz")
        _set_("col.av.sn.en.h", self._col_.nu_SN["en"]["H"], "Collision (Avraged - sn.h) height-time profile", "Hz")
        _set_("col.av.sn.ei", self._col_.nu_SN["ei"]["total"], "Collision (Avraged - sn.ei) height-time profile", "Hz")
        _set_("col.av.sn.ei.o2+", self._col_.nu_SN["ei"]["O2+"], "Collision (Avraged - sn.o2+) height-time profile", "Hz")
        _set_("col.av.sn.ei.o+", self._col_.nu_SN["ei"]["O+"], "Collision (Avraged - sn.o+) height-time profile", "Hz")
        _set_("abs.ah.ft.o", self._abs_.AH["FT"]["O"], "Absorption (AH-FT-O) height-time profile", "dB")
        _set_("abs.ah.ft.x", self._abs_.AH["FT"]["X"], "Absorption (AH-FT-X) height-time profile", "dB")
        _set_("abs.ah.ft.r", self._abs_.AH["FT"]["R"], "Absorption (AH-FT-R) height-time profile", "dB")
        _set_("abs.ah.ft.l", self._abs_.AH["FT"]["L"], "Absorption (AH-FT-L) height-time profile", "dB")
        _set_("abs.ah.ft.no", self._abs_.AH["FT"]["no"], "Absorption (AH-SFT-no) height-time profile", "dB")
        _set_("abs.ah.sn.o", self._abs_.AH["SN"]["O"], "Absorption (AH-SN-O) height-time profile", "dB")
        _set_("abs.ah.sn.x", self._abs_.AH["SN"]["X"], "Absorption (AH-SN-X) height-time profile", "dB")
        _set_("abs.ah.sn.r", self._abs_.AH["SN"]["R"], "Absorption (AH-SN-R) height-time profile", "dB")
        _set_("abs.ah.sn.l", self._abs_.AH["SN"]["L"], "Absorption (AH-SN-L) height-time profile", "dB")
        _set_("abs.ah.av.cc.o", self._abs_.AH["AV_CC"]["O"], "Absorption (AH-AV_CC-O) height-time profile", "dB")
        _set_("abs.ah.av.cc.x", self._abs_.AH["AV_CC"]["X"], "Absorption (AH-AV_CC-X) height-time profile", "dB")
        _set_("abs.ah.av.cc.r", self._abs_.AH["AV_CC"]["R"], "Absorption (AH-AV_CC-R) height-time profile", "dB")
        _set_("abs.ah.av.cc.l", self._abs_.AH["AV_CC"]["L"], "Absorption (AH-AV_CC-L) height-time profile", "dB")
        _set_("abs.ah.av.mb.o", self._abs_.AH["AV_MB"]["O"], "Absorption (AH-AV_MB-O) height-time profile", "dB")
        _set_("abs.ah.av.mb.x", self._abs_.AH["AV_MB"]["X"], "Absorption (AH-AV_MB-X) height-time profile", "dB")
        _set_("abs.ah.av.mb.r", self._abs_.AH["AV_MB"]["R"], "Absorption (AH-AV_MB-R) height-time profile", "dB")
        _set_("abs.ah.av.mb.l", self._abs_.AH["AV_MB"]["L"], "Absorption (AH-AV_MB-L) height-time profile", "dB")
        _set_("abs.sw.ft.o", self._abs_.SW["FT"]["O"], "Absorption (SW-FT-O) height-time profile", "dB")
        _set_("abs.sw.ft.x", self._abs_.SW["FT"]["X"], "Absorption (SW-FT-X) height-time profile", "dB")
        _set_("abs.sw.ft.r", self._abs_.SW["FT"]["R"], "Absorption (SW-FT-R) height-time profile", "dB")
        _set_("abs.sw.ft.l", self._abs_.SW["FT"]["L"], "Absorption (SW-FT-L) height-time profile", "dB")
        _set_("abs.sw.sn.o", self._abs_.SW["SN"]["O"], "Absorption (SW-SN-O) height-time profile", "dB")
        _set_("abs.sw.sn.x", self._abs_.SW["SN"]["X"], "Absorption (SW-SN-X) height-time profile", "dB")
        _set_("abs.sw.sn.r", self._abs_.SW["SN"]["R"], "Absorption (SW-SN-R) height-time profile", "dB")
        _set_("abs.sw.sn.l", self._abs_.SW["SN"]["L"], "Absorption (SW-SN-L) height-time profile", "dB")

        self.rootgrp.close()
        os.system("gzip " + self.ncfile)
        self.sim.save_bgc_file()
        self.sim.clear_local_folders()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rio", default="ott", help="Riometer code (default ott)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,20), help="Start date (default 2015-3-11T16:22)",
            type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,16), help="Start date (default 2015-3-11T15:30)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,17), help="End date (default 2015-3-11T17:30)",
            type=dparser.isoparse)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-fr", "--frequency", type=float, default=30., help="Op. Freq. in MHz (default 30MHz)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for Bgc simulation ")
        for k in vars(args).keys():
            print("     " + k + "->" + str(vars(args)[k]))
    lat, lon = get_riom_loc(args.rio)
    if args.verbose: print("\n Location - (" + str(lat) + "," + str(lon) + ")")
    Bgc(args.rio, args.event, lat, lon, args.start, args.end, args.frequency*1e6)
    os.system("rm -rf models/*.pyc")
