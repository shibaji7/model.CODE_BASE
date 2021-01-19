#!/usr/bin/env python

"""collision.py: collision is dedicated to find all types of collision functions."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2019, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import numpy as np
np.seterr(divide="ignore", invalid="ignore")

from constant import *

# ======================================================================================
# Global class with all static functions to calculate collision frequencies
# ======================================================================================
class Collision(object):
    """
    This class is a global class to estimate any types atmosphreic collision profile.
    
    msis = MSIS module
    iri = IRI module
    Ne = electron density
    Te = electron temperature
    Ti = ion temperature
    """

    def __init__(self, msis, iri, Ne, Te, Ti, frac = 1.0, _run_ = True):
        self.alts = np.arange(500)
        self.Ne = Ne
        self.Te = Te
        self.Ti = Ti
        self.msis = msis
        self.iri = iri
        self.frac = frac
        self.nu_FT = None
        self.nu_av_CC = None
        self.nu_av_MB = None
        self.nu_SN = {
                "en": {
                        "N2": None,
                        "O2": None,
                        "O": None,
                        "H": None,
                        "He": None,
                        "total": None
                    },
                "ei": {
                        "O2+": None,
                        "O+": None,
                        "total": None
                    },
                "total": None,
                }
        if _run_:
            self.calculate_FT_collision_frequency(self.frac)
            self.calculate_SN_en_collision_frequency()
            self.calculate_SN_ei_collision_frequency()
            self.nu_av_CC = self.nu_FT * 2.5
            self.nu_av_MB = self.nu_FT * 1.5
        return

    def calculate_FT_collision_frequency(self, frac = 1.0):
        """
        This method only provides the Friedrich-Tonker electron neutral collision frequency
    
        nn <float> = neutral density
        Tn <float> = neutral temperature
        Te <float> = electron temperatute
        
        
        nu <float> = collision frequency
        https://azformula.com/physics/dimensional-formulae/what-is-dimensional-formula-of-temperature/
        """
        p = self.msis["nn"] * self.msis["Tn"] * pconst["boltz"]
        nu = (2.637e6/np.sqrt(self.Te) + 4.945e5) * p
        self.nu_FT = frac * nu
        return


    def atmospheric_ion_neutral_collision_frequency(self):
        """
        This method only provides the atmsphreic ion neutral collision frequency from collision theory
        
        nn <float> = neutral density
        
        nu <float> = collision frequency
        """
        nu = 3.8e-11*self.msis["nn"]
        return nu

    def calculate_SN_ei_collision_frequency(self,gamma=0.5572,zi=2):
        """
        This method provides electron ion collision frequency profile, nu_ei

        Ne <float> = Electron density in m^-3
        Ni <float> = Ion density in m^-3
        Te <float> = Electron temperature in K
        Ti <float> = Ion temperature in K
        zi <integer 1/2> = Ion Z number

        nu <float> = collision frequency
        """
        e = pconst["q_e"]
        k = pconst["boltz"]
        me = pconst["m_e"]
        eps0 = pconst["eps0"]
        k_e = 1/(4 * np.pi * eps0)
        for key in self.nu_SN["ei"]:
            if key != "total":
                Ni = self.iri["ions"][key]
                ki2 = 4 * np.pi * Ni*1e6 * e**2 * zi**2 * k_e/(k*self.Ti)
                ke2 = 4 * np.pi * self.Ne * e**2 * k_e/(k*self.Te)
                ki = np.sqrt(ki2)
                ke = np.sqrt(ke2)
                lam = np.log(4*k*self.Te/(gamma**2*zi*e**2*k_e*ke)) - (((ke2+ki2)/ki2)*np.log(np.sqrt((ke2+ki2)/ke2)))
                self.nu_SN["ei"][key] = 4*np.sqrt(2*np.pi)*Ni*(zi*e**2*k_e)**2*lam/(3*np.sqrt(me)*(k*self.Te)**(1.5))
            pass
        self.nu_SN["ei"]["total"] = self.nu_SN["ei"]["O2+"] + self.nu_SN["ei"]["O+"]
        self.nu_SN["total"] = self.nu_SN["total"] + self.nu_SN["ei"]["total"]
        return
    
    def calculate_SN_en_collision_frequency(self):
        """
        This method provides electron neutral collision frequency profile, nu_en

        nn <float> = Neutral density m^-3
        Te <float> = Electron temerature in K
        """
        self.nu_SN["en"]["N2"] = 1e-6 * 2.33e-11 * self.msis["N2"] * (1 - (1.12e-4 * self.Te)) * self.Te
        self.nu_SN["en"]["O2"] = 1e-6 * 1.82e-10 * self.msis["O2"] * (1 + (3.6e-2 * np.sqrt(self.Te))) * np.sqrt(self.Te)
        self.nu_SN["en"]["O"] = 1e-6 * 8.9e-11 * self.msis["O"] * (1 + (5.7e-4 * self.Te)) * np.sqrt(self.Te)
        self.nu_SN["en"]["He"] = 1e-6 * 4.6e-10 * self.msis["HE"] * np.sqrt(self.Te)
        self.nu_SN["en"]["H"] = 1e-6 * 4.5e-9 * self.msis["H"] * (1-(1.35e-4 * self.Te)) * np.sqrt(self.Te)
        self.nu_SN["en"]["total"] = self.nu_SN["en"]["N2"] + self.nu_SN["en"]["O2"] + \
                                    self.nu_SN["en"]["O"] + self.nu_SN["en"]["He"] + \
                                    self.nu_SN["en"]["H"]
        self.nu_SN["total"] = self.nu_SN["en"]["total"] #* np.random.uniform(0.7,0.75)
        return

    def atmospheric_collision_frequency(ni, nn, T):
        """
        This method provides atmospheric collision profile based on ion and neural densities and electron temerature
        
        ni <float> = Ion density m^-3
        nn <float> = Neutral density m^-3
        T <float> = Electron temerature in K

        nu <float> = collision frequency
        """
        na_profile = lambda T,nn: (1.8*1e-8*nn*np.sqrt(T/300))
        ni_profile = lambda T,ni: (6.1*1e-3*ni*(300/T)*np.sqrt(300/T))
        nu = (ni_profile(T,ni) + na_profile(T,nn))
        return nu

    @staticmethod
    def load(nc):
        _col_ = Collision(None, None, None, None, None, None, _run_=False)
        _col_.nu_FT = nc.variables["col.ft"][:]
        _col_.nu_av_CC = nc.variables["col.av.cc"][:]
        _col_.nu_av_MB = nc.variables["col.av.mb"][:]
        _col_.nu_SN["total"] = nc.variables["col.av.sn"][:]
        _col_.nu_SN["en"]["total"] = nc.variables["col.av.sn.en"][:]
        _col_.nu_SN["en"]["N2"] = nc.variables["col.av.sn.en.n2"][:]
        _col_.nu_SN["en"]["O"] = nc.variables["col.av.sn.en.o"][:]
        _col_.nu_SN["en"]["He"] = nc.variables["col.av.sn.en.he"][:]
        _col_.nu_SN["en"]["O2"] = nc.variables["col.av.sn.en.o2"][:]
        _col_.nu_SN["en"]["H"] = nc.variables["col.av.sn.en.h"][:]
        _col_.nu_SN["ei"]["O2+"] = nc.variables["col.av.sn.ei.o2+"][:]
        _col_.nu_SN["ei"]["O+"] = nc.variables["col.av.sn.ei.o+"][:]
        _col_.nu_SN["ei"]["total"] = nc.variables["col.av.sn.ei"][:]
        return _col_
