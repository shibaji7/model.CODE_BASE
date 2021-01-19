#!/usr/bin/env python

"""case0.py: case0 experiment python program"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import numpy as np
import datetime as dt
from netCDF4 import Dataset, date2num
import time

from constant import *
import utils
from euvac import Euvac, EuvacBase
from gpi import GPI



def _Case0_(start, end):
    """ Case0 experiment code """
    def _set_(key, val, desc, units, format="f8", shape=("ntimes","nalts")):
        p = rootgrp.createVariable(key,format, shape)
        p.description = desc
        p.uints = units
        p[:] = val
        return

    fname = "data/sim/case0.nc.gz"
    ev, start, end = dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,15,30), dt.datetime(2015,3,11,17,30)
    X = utils.read_goes(ev)
    f0 = np.array(X[(X.date>=start) & (X.date<=end)].B_AVG)

    if not os.path.exists(fname): 
        pg = utils.PointGrid("ott", ev, start, end, 30, v=False)
        pg.chi[:] = 0.
        ir = Euvac.get_solar_flux(ev, start, end)
        cm = GPI(pg, ["O2","N2","O","NO","CO","CO2","H2O"], ir).exe(verbose=True)
        pg.update_grid(cm)
    
        rootgrp = Dataset(fname.replace(".gz",""), "w", format="NETCDF4")
        rootgrp.description = "HF Absorption Model: EUVAC+ Ionosphere, Case0"""
        rootgrp.history = "Created " + time.ctime(time.time())
        rootgrp.source = "SuperDARN HF Absorption Model"
        rootgrp.createDimension("nalts", len(pg.alts))
        rootgrp.createDimension("ntimes", len(pg.dn))
        alts = rootgrp.createVariable("alts","f4",("nalts",))
        alts.description = "Altitude values, in km"
        alts[:] = pg.alts
        times = rootgrp.createVariable("time", "f8", ("ntimes",))
        times.units = "hours since 1970-01-01 00:00:00.0"
        times.calendar = "julian"
        times[:] = date2num(pg.dn,units=times.units,calendar=times.calendar)
        _set_("abs.ah.ft.o", pg._abs_.AH["FT"]["O"], "Absorption (AH-FT-O) height-time profile", "dB")
        _set_("abs.ah.ft.x", pg._abs_.AH["FT"]["X"], "Absorption (AH-FT-X) height-time profile", "dB")
        _set_("abs.ah.ft.r", pg._abs_.AH["FT"]["R"], "Absorption (AH-FT-R) height-time profile", "dB")
        _set_("abs.ah.ft.l", pg._abs_.AH["FT"]["L"], "Absorption (AH-FT-L) height-time profile", "dB")
        _set_("abs.ah.ft.no", pg._abs_.AH["FT"]["no"], "Absorption (AH-SFT-no) height-time profile", "dB")
        _set_("abs.ah.sn.o", pg._abs_.AH["SN"]["O"], "Absorption (AH-SN-O) height-time profile", "dB")
        _set_("abs.ah.sn.x", pg._abs_.AH["SN"]["X"], "Absorption (AH-SN-X) height-time profile", "dB")
        _set_("abs.ah.sn.r", pg._abs_.AH["SN"]["R"], "Absorption (AH-SN-R) height-time profile", "dB")
        _set_("abs.ah.sn.l", pg._abs_.AH["SN"]["L"], "Absorption (AH-SN-L) height-time profile", "dB")
        _set_("abs.ah.av.cc.o", pg._abs_.AH["AV_CC"]["O"], "Absorption (AH-AV_CC-O) height-time profile", "dB")
        _set_("abs.ah.av.cc.x", pg._abs_.AH["AV_CC"]["X"], "Absorption (AH-AV_CC-X) height-time profile", "dB")
        _set_("abs.ah.av.cc.r", pg._abs_.AH["AV_CC"]["R"], "Absorption (AH-AV_CC-R) height-time profile", "dB")
        _set_("abs.ah.av.cc.l", pg._abs_.AH["AV_CC"]["L"], "Absorption (AH-AV_CC-L) height-time profile", "dB")
        _set_("abs.ah.av.mb.o", pg._abs_.AH["AV_MB"]["O"], "Absorption (AH-AV_MB-O) height-time profile", "dB")
        _set_("abs.ah.av.mb.x", pg._abs_.AH["AV_MB"]["X"], "Absorption (AH-AV_MB-X) height-time profile", "dB")
        _set_("abs.ah.av.mb.r", pg._abs_.AH["AV_MB"]["R"], "Absorption (AH-AV_MB-R) height-time profile", "dB")
        _set_("abs.ah.av.mb.l", pg._abs_.AH["AV_MB"]["L"], "Absorption (AH-AV_MB-L) height-time profile", "dB")
        _set_("abs.sw.ft.o", pg._abs_.SW["FT"]["O"], "Absorption (SW-FT-O) height-time profile", "dB")
        _set_("abs.sw.ft.x", pg._abs_.SW["FT"]["X"], "Absorption (SW-FT-X) height-time profile", "dB")
        _set_("abs.sw.ft.r", pg._abs_.SW["FT"]["R"], "Absorption (SW-FT-R) height-time profile", "dB")
        _set_("abs.sw.ft.l", pg._abs_.SW["FT"]["L"], "Absorption (SW-FT-L) height-time profile", "dB")
        _set_("abs.sw.sn.o", pg._abs_.SW["SN"]["O"], "Absorption (SW-SN-O) height-time profile", "dB")
        _set_("abs.sw.sn.x", pg._abs_.SW["SN"]["X"], "Absorption (SW-SN-X) height-time profile", "dB")
        _set_("abs.sw.sn.r", pg._abs_.SW["SN"]["R"], "Absorption (SW-SN-R) height-time profile", "dB")
        _set_("abs.sw.sn.l", pg._abs_.SW["SN"]["L"], "Absorption (SW-SN-L) height-time profile", "dB")
        _set_("ne", pg.ne, "Electron density height-time profile", "m^-3")
        _set_("ni", pg.ni, "Ion density height-time profile", "m^-3")
        _set_("ni-", pg.ni_e, "Ion(-) density height-time profile", "m^-3")
        _set_("nix", pg.ni_x, "Ion(x+) density height-time profile", "m^-3")
        drap = rootgrp.createVariable("drap", "f8", ("ntimes",))
        drap[:] = pg.drap.ravel()
        drap.units = "dB"
        drap.description = "Absorption estimated by DRAP2"
        rootgrp.close()
        os.system("gzip "+fname.replace(".gz",""))
    return f0
