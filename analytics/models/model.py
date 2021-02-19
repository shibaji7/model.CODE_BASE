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
import traceback

import utils
from euvac import Euvac
from gpi import GPI
import plot_lib as plib

from absorption import *
from collision import *

from fetch_data import get_session, Goes, Riometer, Simulation

class Model(object):
    """ 1D model class to run 1D (hight) model functions """

    def __init__(self, rio, ev, args, _dir_="proc/outputs/{date}/{rio}/"):
        """ Initialize all the parameters """
        self.rio = rio
        self.ev = ev
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        if self.species == 0: self.sps = ["O2","N2","O"]
        elif self.species == 1: self.sps = ["O2","N2","O","NO","CO","CO2","H2O"]
        self._dir_ = _dir_.format(date=self.ev.strftime("%Y.%m.%d.%H.%M"), rio=self.rio)
        self.sim = Simulation(self.ev, self.rio)
        self.conn = get_session()
        self.check_run = False
        if self.sim.check_riometer_data_exists(self.conn) and self.sim.check_goes_exists(self.conn) and\
            not self.sim.check_bgc_not_exists(self.conn) and not self.sim.check_flare_not_exists(self.conn):
            self._init_()
            self.check_run = True
            self.files = []
        return

    def _init_(self):
        """ Initialize the data folder """
        if not os.path.exists(self._dir_): self.sim.create_remote_local_dir(self.conn)
        self.download_goes_riometers_bgc()
        return
    
    def download_goes_riometers_bgc(self):
        Goes().get_goes_file(self.conn, self.ev)
        Riometer().get_riometer_file(self.conn, self.ev, self.rio)
        self.sim.get_bgc_file(self.conn)
        return
    
    def clean(self):
        Goes().clean_local_file(self.ev)
        Riometer().clean_local_file(self.ev, self.rio)
        self.sim.clear_local_folders()
        self.conn.close()
        return

    def _plot_comp_(self):
        """ Plot comparsion simulation """
        plib.model_outputs(self.pg, [32, 52])
        plib.event_study(self.ev, self.rio, self.pg, self.start, self.end)
        return

    def _save_(self, fname="proc/outputs/{date}/{rio}/flare.nc.gz"):
        """ Save simulation """
        try:
            plib.event_study(self.ev, self.rio, self.pg, self.start, self.end, fname=self._dir_+"event.{rio}.png".format(rio=self.rio))
            self.files.append(self._dir_+"event.{rio}.png".format(rio=self.rio))
        except: traceback.print_exc()
        
        def _set_(key, val, desc, units, format="f8", shape=("ntimes","nalts")):
            p = rootgrp.createVariable(key,format, shape)
            p.description = desc
            p.uints = units
            p[:] = val
            return
        
        fname = fname.format(date=self.ev.strftime("%Y.%m.%d.%H.%M"), rio=self.rio)
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
        self.sim.save_flare_file(self.conn)
        if len(self.files) > 0: self.sim.save_image_files(self.conn, self.files)
        return

    def run(self):
        """ Run the model """
        if self.check_run:
            print("\n Modified freq - ", self.frequency)
            try:
                self.pg = utils.PointGrid(self.rio, self.ev, self.start, self.end, freq=self.frequency, v=self.verbose)
                self.ir = Euvac.get_solar_flux(self.ev, self.start, self.end)
                self.cm = GPI(self.pg, self.sps, self.ir).exe(verbose=self.verbose)
                self.pg.update_grid(self.cm)
                if hasattr(self, "plot_summary") and self.plot_summary: self._plot_comp_()
                if hasattr(self, "save_result") and self.save_result: self._save_()
            except:
                traceback.print_exc()
            self.clean()
        return

    def _exp_(self, name, params, save_fname=None):
        """ Experiment specific model run. Provide scale factors of the parameter """
        print("\n Experiment - ", name)
        self.pg = utils.PointGrid(self.rio, self.ev, self.start, self.end, freq=self.frequency, v=self.verbose, fname="data/tElec/{dn}/")
        self.ir = Euvac.get_solar_flux(self.ev, self.start, self.end)
        if name == "TElec":
            if save_fname is None: save_fname = "data/tElec/{dn}/flare.{stn}.TElec[%.2f].nc.gz"%params["TElec"]
            lam = 1.
            self.pg.iri["Te"] = self.pg.iri["Te"] * params["TElec"]
            self.pg._col_ = Collision(self.pg.msis, self.pg.iri, self.pg.iri["Ne"], self.pg.iri["Te"], self.pg.iri["Ti"])
        if name == "lambda": 
            if save_fname is None: save_fname = "data/tElec/{dn}/flare.{stn}.lambda[%.2f].nc.gz"%params["lambda"]
            lam = params["lambda"]
        self.cm = GPI(self.pg, self.sps, self.ir, lam_const=lam).exe(verbose=self.verbose)
        self.pg.update_grid(self.cm)
        if hasattr(self, "save_result") and self.save_result: self._save_(save_fname)
