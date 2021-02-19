#!/usr/bin/env python

"""absorption.py: absorption is dedicated to absorption related function."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
from scipy.integrate import quad
import math

from constant import *
import utils

# ===================================================================================
# These are special function dedicated to the Sen-Wyller absorption calculation.
# 
# Sen, H. K., and Wyller, A. A. ( 1960), On the Generalization of Appleton-Hartree magnetoionic Formulas
# J. Geophys. Res., 65( 12), 3931- 3950, doi:10.1029/JZ065i012p03931.
# 
# ===================================================================================
def C(p, y):

    def gamma_factorial(N):
        n = int(str(N).split(".")[0])
        f = N-n
        if f > 0.: fact = math.factorial(n) * math.gamma(f)
        else: fact = math.factorial(n)
        return fact

    func = lambda t: t**p * np.exp(-t) / (t**2 + y**2)
    cy, abserr = quad(func, 0, np.inf)
    return cy / gamma_factorial(p)


def calculate_sw_RL_abs(Bo, Ne, nu, fo=30e6, nu_sw_r = 1.):
    if Ne > 0. and Bo > 0. and nu > 0. and (not np.isnan(Ne)) \
            and (not np.isnan(Bo)) and (not np.isnan(nu)):
        k = (2*np.pi*fo) / pconst["c"]
        w = 2 * np.pi * fo
        nu_sw = nu * nu_sw_r
        wh = pconst["q_e"] * Bo / pconst["m_e"]
        yo, yx = (w+wh) / nu_sw, (w-wh) / nu_sw
        nL = 1 - ( ( Ne * pconst["q_e"]**2/(2*pconst["m_e"]*w*pconst["eps0"]*nu_sw) )
                        * np.complex(yo*C(1.5,yo),2.5*C(2.5,yo)) )
        nR = 1 - ( ( Ne*pconst["q_e"]**2/(2*pconst["m_e"]*w*pconst["eps0"]*nu_sw) )
                        * np.complex(yx*C(1.5,yx),2.5*C(2.5,yx)) )
        R, L = np.abs(nR.imag * 8.68 * k * 1e3), np.abs(nL.imag * 8.68 * k * 1e3)
    else:
        R,L = np.nan, np.nan
    return R, L


def calculate_sw_OX_abs(Bo, Ne, nu, fo=30e6, nu_sw_r = 1.):
    if Ne > 0. and Bo > 0. and nu > 0. and (not np.isnan(Ne)) \
            and (not np.isnan(Bo)) and (not np.isnan(nu)):
        k = (2*np.pi*fo) / pconst["c"]
        w = 2 * np.pi * fo
        nu_sw = nu * nu_sw_r
        wh = pconst["q_e"] * Bo / pconst["m_e"]
        wo2 = Ne * pconst["q_e"]**2 / (pconst["m_e"] * pconst["eps0"])
        yo, yx = (w) / nu_sw, (w) / nu_sw
        y = (w) / nu_sw

        ajb = (wo2 / (w*nu_sw)) * ((y*C(1.5,y)) + 1.j*(2.5*C(2.5,y)))
        c = (wo2 / (w*nu_sw)) * yx * C(1.5,yx)
        d = 2.5 * (wo2 / (w*nu_sw)) * C(1.5,yx)
        e = (wo2 / (w*nu_sw)) * yo * C(1.5,yo)
        f = 2.5 * (wo2 / (w*nu_sw)) * C(1.5,yo)

        eI = 1 - ajb
        eII = 0.5* ( (f-d) + (c-e)*1.j )
        eIII = ajb - (0.5 * ((c+e) + 1.j*(d+f)))

        Aa = 2*eI*(eI+eIII)
        Bb = (eIII*(eI+eII)) + eII**2
        Cc = 2*eI*eII
        Dd = 2*eI
        Ee = 2*eIII

        nO = np.sqrt(Aa / (Dd+Ee))
        nX = np.sqrt((Aa+Bb) / (Dd+Ee))
        O, X = np.abs(nO.imag * 8.68 * k * 1e3), np.abs(nX.imag * 8.68 * k * 1e3)
    else:
        O,X = np.nan, np.nan
    return O, X

# ===================================================================================
# This class is used to estimate O,X,R & L mode absorption height profile.
# ===================================================================================
class Absorption(object):
    """
    This class is used to estimate O,X,R & L mode absorption height profile.

    Bo = geomagnetic field
    coll = collision frequency
    Ne = electron density
    fo = operating frequency
    """

    def __init__(self, Bo, coll, Ne, fo=30e6, nu_sw_r=1., _run_=True):
        if _run_:
            self.Bo = Bo
            self.Ne = Ne
            self.coll = coll
            self.fo = fo
            self.nu_sw_r = nu_sw_r
            self.drap_abs = None

            self.w = 2 * np.pi * fo
            self.k = (2 * np.pi * fo) / pconst["c"]
            self.AH = {
                    "FT": {
                            "O": None,
                            "X": None,
                            "R": None,
                            "L": None,
                            "no": None,
                            },
                    "SN": {
                            "O": None,
                            "X": None,
                            "R": None,
                            "L": None,
                            },
                    "AV_CC": {
                            "O": None,
                            "X": None,
                            "R": None,
                            "L": None,
                        },
                    "AV_MB": {
                            "O": None,
                            "X": None,
                            "R": None,
                            "L": None,
                        },
                    }
            self.SW = {
                    "FT": {
                            "O": np.zeros(Bo.shape),
                            "X": np.zeros(Bo.shape),
                            "R": np.zeros(Bo.shape),
                            "L": np.zeros(Bo.shape),
                            },
                    "SN": {
                            "O": np.zeros(Bo.shape),
                            "X": np.zeros(Bo.shape),
                            "R": np.zeros(Bo.shape),
                            "L": np.zeros(Bo.shape),
                            },
                    }
            self.estimate()
        return

    def estimate_AH(self):
        # =========================================================
        # Using FT collision frequency
        # =========================================================
        X,Z = (self.Ne * pconst["q_e"]**2) / (pconst["eps0"] * pconst["m_e"] * self.w**2), self.coll.nu_FT / self.w
        x,jz = X,Z*1.j
        n = np.sqrt(1 - (x / (1-jz)))
        self.AH["FT"]["no"] = np.abs( 8.68 * self.k * 1e3 * n.imag )

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO,nX = np.sqrt( 1 - ( x / (1-jz) ) ), \
                np.sqrt( 1 - ( (2*x*(1-x-jz)) / ( (2*(1-x-jz)*(1-jz)) - (2*YT**2) ) ) )
        self.AH["FT"]["O"], self.AH["FT"]["X"] = np.abs( 8.68 * self.k * 1e3 * nO.imag ), \
                np.abs( 8.68 * self.k * 1e3 * nX.imag )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL,nR = np.sqrt(1 - (x/((1-jz) + YL))), np.sqrt(1 - (x/((1-jz)-YL)))
        self.AH["FT"]["R"], self.AH["FT"]["L"] = np.abs(8.68 * self.k * 1e3 * nR.imag), \
                                    np.abs(8.68 * self.k * 1e3 * nL.imag)

        # ========================================================
        # Using SN collision frequency  quite_model   
        # ========================================================
        Z = self.coll.nu_SN["total"] / self.w
        jz = Z*1.j
        n = np.sqrt(1 - (x / (1-jz)))
        self.AH["SN"]["no"] = np.abs( 8.68 * self.k * 1e3 * n.imag )

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO,nX = np.sqrt( 1 - ( x / (1-jz) ) ), \
                np.sqrt( 1 - ( (2*x*(1-x-jz)) / ( (2*(1-x-jz)*(1-jz)) - (2*YT**2) ) ) )
        self.AH["SN"]["O"], self.AH["SN"]["X"] = np.abs( 8.68 * self.k * 1e3 * nO.imag ), \
                np.abs( 8.68 * self.k * 1e3 * nX.imag )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL,nR = np.sqrt(1 - (x/((1-jz) + YL))), np.sqrt(1 - (x/((1-jz)-YL)))
        self.AH["SN"]["R"], self.AH["SN"]["L"] = np.abs(8.68 * self.k * 1e3 * nR.imag), \
                                    np.abs(8.68 * self.k * 1e3 * nL.imag)

        # =========================================================
        # Using AV_CC collision frequency quite_model
        # =========================================================
        Z = self.coll.nu_av_CC / self.w
        jz = Z*1.j
        n = np.sqrt(1 - (x / (1-jz)))
        self.AH["AV_CC"]["no"] = np.abs( 8.68 * self.k * 1e3 * n.imag )

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO,nX = np.sqrt( 1 - ( x / (1-jz) ) ), \
                np.sqrt( 1 - ( (2*x*(1-x-jz)) / ( (2*(1-x-jz)*(1-jz)) - (2*YT**2) ) ) )
        self.AH["AV_CC"]["O"], self.AH["AV_CC"]["X"] = np.abs( 8.68 * self.k * 1e3 * nO.imag ), \
                np.abs( 8.68 * self.k * 1e3 * nX.imag )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL,nR = np.sqrt(1 - (x/((1-jz) + YL))), np.sqrt(1 - (x/((1-jz)-YL)))
        self.AH["AV_CC"]["R"], self.AH["AV_CC"]["L"] = np.abs(8.68 * self.k * 1e3 * nR.imag), \
                                    np.abs(8.68 * self.k * 1e3 * nL.imag)

        # =========================================================
        # Using AV_MB collision frequency quite_model
        # =========================================================
        Z = self.coll.nu_av_MB / self.w
        jz = Z*1.j
        n = np.sqrt(1 - (x / (1-jz)))
        self.AH["AV_MB"]["no"] = np.abs( 8.68 * self.k * 1e3 * n.imag )

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO,nX = np.sqrt( 1 - ( x / (1-jz) ) ), \
                np.sqrt( 1 - ( (2*x*(1-x-jz)) / ( (2*(1-x-jz)*(1-jz)) - (2*YT**2) ) ) )
        self.AH["AV_MB"]["O"], self.AH["AV_MB"]["X"] = np.abs( 8.68 * self.k * 1e3 * nO.imag ), \
                np.abs( 8.68 * self.k * 1e3 * nX.imag )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL,nR = np.sqrt(1 - (x/((1-jz) + YL))), np.sqrt(1 - (x/((1-jz)-YL)))
        self.AH["AV_MB"]["R"], self.AH["AV_MB"]["L"] = np.abs(8.68 * self.k * 1e3 * nR.imag), \
                                    np.abs(8.68 * self.k * 1e3 * nL.imag)
        return

    def estimate_SW(self):
        I,J = self.Bo.shape
        nu_sw_r = self.nu_sw_r #* np.random.uniform(0)
        # ===================================================
        # Using FT collistion frequency
        # ===================================================
        nu = self.coll.nu_FT
        for i in range(I):
            for j in range(J):
                self.SW["FT"]["O"][i,j], self.SW["FT"]["X"][i,j]= \
                                calculate_sw_OX_abs(self.Bo[i,j], self.Ne[i,j], nu[i,j],self.fo,nu_sw_r=nu_sw_r)
                self.SW["FT"]["R"][i,j], self.SW["FT"]["L"][i,j]= \
                                calculate_sw_RL_abs(self.Bo[i,j], self.Ne[i,j], nu[i,j],self.fo,nu_sw_r=nu_sw_r)
                pass
            pass

        # ==================================================
        # Using SN collistion frequency
        # ==================================================
        nu = self.coll.nu_SN["total"]
        for i in range(I):
            for j in range(J):
                self.SW["SN"]["O"][i,j], self.SW["SN"]["X"][i,j]= \
                                calculate_sw_OX_abs(self.Bo[i,j], self.Ne[i,j], nu[i,j],self.fo,nu_sw_r=nu_sw_r)
                self.SW["SN"]["R"][i,j], self.SW["SN"]["L"][i,j]= \
                                calculate_sw_RL_abs(self.Bo[i,j], self.Ne[i,j], nu[i,j],self.fo,nu_sw_r=nu_sw_r)
                pass
            pass
        return

    def estimate(self):
        self.estimate_AH()
        self.estimate_SW()
        return

    @staticmethod
    def load(nc):
        _abs_ = Absorption(None, None, None, None, None, _run_=False)
        _abs_.AH = {
                 "FT": {
                     "O": nc.variables["abs.ah.ft.o"][:],
                     "X": nc.variables["abs.ah.ft.x"][:],
                     "R": nc.variables["abs.ah.ft.r"][:],
                     "L": nc.variables["abs.ah.ft.l"][:],
                     "no": nc.variables["abs.ah.ft.no"][:],
                     },
                 "SN": {
                     "O": nc.variables["abs.ah.sn.o"][:],
                     "X": nc.variables["abs.ah.sn.x"][:],
                     "R": nc.variables["abs.ah.sn.r"][:],
                     "L": nc.variables["abs.ah.sn.l"][:],
                     },
                 "AV_CC": {
                     "O": nc.variables["abs.ah.av.cc.o"][:],
                     "X": nc.variables["abs.ah.av.cc.x"][:],
                     "R": nc.variables["abs.ah.av.cc.r"][:],
                     "L": nc.variables["abs.ah.av.cc.l"][:],
                     },
                 "AV_MB": {
                     "O": nc.variables["abs.ah.av.mb.o"][:],
                     "X": nc.variables["abs.ah.av.mb.x"][:],
                     "R": nc.variables["abs.ah.av.mb.r"][:],
                     "L": nc.variables["abs.ah.av.mb.l"][:],
                     },
                 }
        _abs_.SW = SW = {
                "FT": {
                    "O": nc.variables["abs.sw.ft.o"][:],
                    "X": nc.variables["abs.sw.ft.x"][:],
                    "R": nc.variables["abs.sw.ft.r"][:],
                    "L": nc.variables["abs.sw.ft.l"][:],
                    },
                "SN": {
                    "O": nc.variables["abs.sw.sn.o"][:],
                    "X": nc.variables["abs.sw.sn.x"][:],
                    "R": nc.variables["abs.sw.sn.r"][:],
                    "L": nc.variables["abs.sw.sn.l"][:],
                    },
                }
        return _abs_

    @staticmethod
    def _drap_(ev, dates, stn, frq):
        """ Estimation based on DRAP2 model """
        xray = utils.read_goes(ev)
        xray = xray[(xray.date>=dates[0]) & (xray.date<=dates[-1])]
        haf = 10*np.log10(np.array(xray.B_AVG)) + 65
        af0 = (haf/frq)**1.5
        lat, lon = utils.get_riom_loc(stn)
        sza = utils.calculate_sza(dates, lat, lon, [100])
        af = af0 * abs(np.cos(np.deg2rad(sza.ravel())))**0.75
        return af

    @staticmethod
    def _sato_(ev, dates, stn, frq):
        """ Estimation based on Sato 1975 model """
        xray = utils.read_goes(ev)
        xray = xray[(xray.date>=dates[0]) & (xray.date<=dates[-1])]
        lat, lon = utils.get_riom_loc(stn)
        sza = utils.calculate_sza(dates, lat, lon, [100])
        af = 4.37e3 * (1/frq)**2 * (xray.B_AVG**0.5) * np.abs(np.cos(np.deg2rad(sza.ravel())))
        return af
