"""models.py: models is used to manage and run the model"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import datetime as dt
from netCDF4 import Dataset, date2num
import time

import utils
from euvac import Euvac
from gpi import GPI
import plot_lib as plib

from absorption import *
from collision import *

class Model(object):
    """ 1D model class to run 1D (hight) model functions """

    def __init__(self, rio, ev, args):
        """ Initialize all the parameters """
        self.rio = rio
        self.ev = ev
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        if self.species == 0: self.sps = ["O2","N2","O"]
        elif self.species == 1: self.sps = ["O2","N2","O","NO","CO","CO2","H2O"]
        self._dir_ = "data/sim/{date}".format(date=self.ev.strftime("%Y.%m.%d.%H.%M"))
        self._init_()
        if hasattr(self, "clear") and self.clear: self._clean_()
        if hasattr(self, "save_goes") and self.save_goes: self._save_goes_()
        if hasattr(self, "save_riom") and self.save_riom: self._save_riom_()
        return

    def _save_riom_(self):
        """ Download and save riometer file """
        utils.download_riometer(self.ev, self.rio, self.verbose)
        return

    def _save_goes_(self):
        """ Download and save goes file """
        utils.download_goes_data(self.ev, sat=self.sat, v=self.verbose)
        return

    def _init_(self):
        """ Initialize the data folder """
        if not os.path.exists(self._dir_):  os.system("mkdir -p " + self._dir_)
        if not os.path.exists("{_dir_}/goes".format(_dir_=self._dir_)): os.system("mkdir {_dir_}/goes".format(_dir_=self._dir_))
        if not os.path.exists("{_dir_}/rio".format(_dir_=self._dir_)): os.system("mkdir {_dir_}/rio".format(_dir_=self._dir_))
        return

    def _clean_(self):
        """ Clean folders and historical files """
        os.system("rm -rf {_dir_}".format(_dir_=self._dir_))
        self._init_()
        return

    def _plot_comp_(self):
        """ Plot comparsion simulation """
        plib.model_outputs(self.pg, [32, 52])
        plib.event_study(self.ev, self.rio, self.pg, self.start, self.end)
        return

    def _save_(self, fname="data/sim/{dn}/flare.{stn}.nc.gz"):
        """ Save simulation """
        plib.event_study(self.ev, self.rio, self.pg, self.start, self.end, fname=self._dir_+"/event.{rio}.png".format(rio=self.rio))
        
        def _set_(key, val, desc, units, format="f8", shape=("ntimes","nalts")):
            p = rootgrp.createVariable(key,format, shape)
            p.description = desc
            p.uints = units
            p[:] = val
            return
        
        fname = fname.format(dn=self.ev.strftime("%Y.%m.%d.%H.%M"), stn=self.rio)
        if os.path.exists(fname): os.remove(fname)
        rootgrp = Dataset(fname.replace(".gz",""), "w", format="NETCDF4")
        rootgrp.description = "HF Absorption Model: EUVAC+ Ionosphere (R:{rio})""".format(rio=self.rio)
        rootgrp.history = "Created " + time.ctime(time.time())
        rootgrp.source = "SuperDARN HF Absorption Model"
        rootgrp.createDimension("nalts", len(self.pg.alts))
        rootgrp.createDimension("ntimes", len(self.pg.dn))
        alts = rootgrp.createVariable("alts","f4",("nalts",))
        alts.description = "Altitude values, in km"
        alts[:] = self.pg.alts
        times = rootgrp.createVariable("time", "f8", ("ntimes",))
        times.units = "hours since 1970-01-01 00:00:00.0"
        times.calendar = "julian"
        times[:] = date2num(self.pg.dn,units=times.units,calendar=times.calendar)
        _set_("abs.ah.ft.o", self.pg._abs_.AH["FT"]["O"], "Absorption (AH-FT-O) height-time profile", "dB")
        _set_("abs.ah.ft.x", self.pg._abs_.AH["FT"]["X"], "Absorption (AH-FT-X) height-time profile", "dB")
        _set_("abs.ah.ft.r", self.pg._abs_.AH["FT"]["R"], "Absorption (AH-FT-R) height-time profile", "dB")
        _set_("abs.ah.ft.l", self.pg._abs_.AH["FT"]["L"], "Absorption (AH-FT-L) height-time profile", "dB")
        _set_("abs.ah.ft.no", self.pg._abs_.AH["FT"]["no"], "Absorption (AH-SFT-no) height-time profile", "dB")
        _set_("abs.ah.sn.o", self.pg._abs_.AH["SN"]["O"], "Absorption (AH-SN-O) height-time profile", "dB")
        _set_("abs.ah.sn.x", self.pg._abs_.AH["SN"]["X"], "Absorption (AH-SN-X) height-time profile", "dB")
        _set_("abs.ah.sn.r", self.pg._abs_.AH["SN"]["R"], "Absorption (AH-SN-R) height-time profile", "dB")
        _set_("abs.ah.sn.l", self.pg._abs_.AH["SN"]["L"], "Absorption (AH-SN-L) height-time profile", "dB")
        _set_("abs.ah.av.cc.o", self.pg._abs_.AH["AV_CC"]["O"], "Absorption (AH-AV_CC-O) height-time profile", "dB")
        _set_("abs.ah.av.cc.x", self.pg._abs_.AH["AV_CC"]["X"], "Absorption (AH-AV_CC-X) height-time profile", "dB")
        _set_("abs.ah.av.cc.r", self.pg._abs_.AH["AV_CC"]["R"], "Absorption (AH-AV_CC-R) height-time profile", "dB")
        _set_("abs.ah.av.cc.l", self.pg._abs_.AH["AV_CC"]["L"], "Absorption (AH-AV_CC-L) height-time profile", "dB")
        _set_("abs.ah.av.mb.o", self.pg._abs_.AH["AV_MB"]["O"], "Absorption (AH-AV_MB-O) height-time profile", "dB")
        _set_("abs.ah.av.mb.x", self.pg._abs_.AH["AV_MB"]["X"], "Absorption (AH-AV_MB-X) height-time profile", "dB")
        _set_("abs.ah.av.mb.r", self.pg._abs_.AH["AV_MB"]["R"], "Absorption (AH-AV_MB-R) height-time profile", "dB")
        _set_("abs.ah.av.mb.l", self.pg._abs_.AH["AV_MB"]["L"], "Absorption (AH-AV_MB-L) height-time profile", "dB")
        _set_("abs.sw.ft.o", self.pg._abs_.SW["FT"]["O"], "Absorption (SW-FT-O) height-time profile", "dB")
        _set_("abs.sw.ft.x", self.pg._abs_.SW["FT"]["X"], "Absorption (SW-FT-X) height-time profile", "dB")
        _set_("abs.sw.ft.r", self.pg._abs_.SW["FT"]["R"], "Absorption (SW-FT-R) height-time profile", "dB")
        _set_("abs.sw.ft.l", self.pg._abs_.SW["FT"]["L"], "Absorption (SW-FT-L) height-time profile", "dB")
        _set_("abs.sw.sn.o", self.pg._abs_.SW["SN"]["O"], "Absorption (SW-SN-O) height-time profile", "dB")
        _set_("abs.sw.sn.x", self.pg._abs_.SW["SN"]["X"], "Absorption (SW-SN-X) height-time profile", "dB")
        _set_("abs.sw.sn.r", self.pg._abs_.SW["SN"]["R"], "Absorption (SW-SN-R) height-time profile", "dB")
        _set_("abs.sw.sn.l", self.pg._abs_.SW["SN"]["L"], "Absorption (SW-SN-L) height-time profile", "dB")
        _set_("ne", self.pg.ne, "Electron density height-time profile", "m^-3")
        _set_("ni", self.pg.ni, "Ion density height-time profile", "m^-3")
        _set_("ni-", self.pg.ni_e, "Ion(-) density height-time profile", "m^-3")
        _set_("nix", self.pg.ni_x, "Ion(x+) density height-time profile", "m^-3")
        drap = rootgrp.createVariable("drap", "f8", ("ntimes",))
        drap[:] = self.pg.drap.ravel()
        drap.units = "dB"
        drap.description = "Absorption estimated by DRAP2"
        sato = rootgrp.createVariable("sato", "f8", ("ntimes",))
        sato[:] = self.pg.sato.ravel()
        sato.units = "dB"
        sato.description = "Absorption estimated by Sato et al. (1975)"

        rootgrp.close()
        os.system("gzip "+fname.replace(".gz",""))
        return

    def run(self):
        """ Run the model """
        print("\n Modified freq - ", self.frequency)
        self.pg = utils.PointGrid(self.rio, self.ev, self.start, self.end, freq=self.frequency, v=self.verbose)
        self.ir = Euvac.get_solar_flux(self.ev, self.start, self.end)
        self.cm = GPI(self.pg, self.sps, self.ir).exe(verbose=self.verbose)
        self.pg.update_grid(self.cm)
        if hasattr(self, "plot_summary") and self.plot_summary: self._plot_comp_()
        if hasattr(self, "save_result") and self.save_result: self._save_()
        return

    def _exp_(self, name, params):
        """ Experiment specific model run. Provide scale factors of the parameter """
        print("\n Experiment - ", name)
        self.pg = utils.PointGrid(self.rio, self.ev, self.start, self.end, freq=self.frequency, v=self.verbose)
        self.ir = Euvac.get_solar_flux(self.ev, self.start, self.end)
        if name == "TElec":
            fname = "data/sim/{dn}/flare.{stn}.TElec[%.2f].nc.gz"%params["TElec"]
            lam = 1.
            self.pg.iri["Te"] = self.pg.iri["Te"] * params["TElec"]
            self.pg._col_ = Collision(self.pg.msis, self.pg.iri, self.pg.iri["Ne"], self.pg.iri["Te"], self.pg.iri["Ti"])
        if name == "lambda": 
            fname = "data/sim/{dn}/flare.{stn}.lambda[%.2f].nc.gz"%params["lambda"]
            lam = params["lambda"]
        self.cm = GPI(self.pg, self.sps, self.ir, lam_const=lam).exe(verbose=self.verbose)
        self.pg.update_grid(self.cm)
        if hasattr(self, "save_result") and self.save_result: self._save_(fname)
