#!/usr/bin/env python

"""euvac.py: euvac instatntiate EUVAC model and EUCAV+ model definations."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2019, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import numpy as np
import pandas as pd
import pyglow
import datetime as dt


import utils
from constant import *

class EuvacBase(object):
    """
    This method is from Karthik's paper / Descertaion
    """

    """
    Number of wavelength bins & initialize model
    """
    n_wave = 37
    solspec = {
                "n_wave"     : n_wave      ,
                "current"    : np.array(n_wave),
                "reference"  : np.array(n_wave),
                "wave1"      : np.array(n_wave),
                "wave2"      : np.array(n_wave),
                "afac"       : np.array(n_wave),
                "xsec"       : {},
                "eta"        : {},
          }

    """
    Describe GOES wavelength bins in nm (in the descending order of bins)
    """
    solspec["GOES_wave1"] = np.array([0.05, 0.1]) * 1e-9 
    solspec["GOES_wave2"] = np.array([0.4, 0.8]) * 1e-9

    """
    Definition of wavelength bins in Amngstorm, reference spectrum, and A-factors
    """
    solspec["wave1"] = np.array([1700.00, 1650.00, 1600.00, 1550.00, 1500.00,
                     1450.00, 1400.00, 1350.00, 1300.00, 1250.00,
                     1200.00, 1215.67, 1150.00, 1100.00, 1050.00,
                     1027.00,  987.00,  975.00,  913.00,  913.00,
                      913.00,  798.00,  798.00,  798.00,  650.00,
                      650.00,  540.00,  320.00,  290.00,  224.00,
                      155.00,   70.00,   32.00,   18.00,    8.00,
                        4.00,    0.50])

    solspec["wave2"] = np.array([1750.00, 1700.00, 1650.00, 1600.00, 1550.00,
                     1500.00, 1450.00, 1400.00, 1350.00, 1300.00,
                     1250.00, 1215.67, 1200.00, 1150.00, 1100.00,
                     1050.00, 1027.00,  987.00,  975.00,  975.00,
                      975.00,  913.00,  913.00,  913.00,  798.00,
                      798.00,  650.00,  540.00,  320.00,  290.00,
                      224.00,  155.00,   70.00,   32.00,   18.00,
                        8.00,    4.00])

    """
    Reference solar spectrum, photon /cm^2 /s
    """
    solspec["reference"] = np.array([3.397e+11, 1.998e+11, 1.055e+11, 7.260e+10,
                     5.080e+10, 2.802e+10, 1.824e+10, 1.387e+10,
                     2.659e+10, 7.790e+09, 1.509e+10, 3.940e+11,
                     8.399e+09, 3.200e+09, 3.298e+09, 4.235e+09,
                 4.419e+09, 4.482e+09, 7.156e+08, 1.028e+09,
                 3.818e+08, 8.448e+08, 3.655e+09, 2.364e+09,
                 1.142e+09, 1.459e+09, 4.830e+09, 2.861e+09,
                 8.380e+09, 4.342e+09, 5.612e+09, 1.270e+09,
                 5.326e+08, 2.850e+07, 2.000e+06, 1.000e+04,
                 5.010e+01])
    """
    Scaling factor A as defined in EUVAC model
    """
    solspec["afac"] = np.array([5.937e-04, 6.089e-04, 1.043e-03, 1.125e-03,
                            1.531e-03, 1.202e-03, 1.873e-03, 2.632e-03,
                            2.877e-03, 2.610e-03, 3.739e-03, 4.230e-03,
                            2.541e-03, 2.099e-03, 3.007e-03, 4.825e-03,
                            5.021e-03, 3.950e-03, 4.422e-03, 4.955e-03,
                            4.915e-03, 5.437e-03, 5.261e-03, 5.310e-03,
                            3.680e-03, 5.719e-03, 5.857e-03, 1.458e-02,
                            7.059e-03, 2.575e-02, 1.433e-02, 9.182e-03,
                            1.343e-02, 6.247e-02, 2.000e-01, 3.710e-01,
                            6.240e-01])

    """
    Describing absorption crossscetion in m^2 in following order
            O, O2, N2, CO, aCO2, O3, bCO2, H2O, NO
    """
    solspec["xsec"]  = {"abs": {}, "br_ratio": {}}
    solspec["xsec"]["abs"]["_desc"] = "Tag 'xsec.abs' provides the absorption cross-section for different species m^2"
    solspec["xsec"]["abs"]["species"] = np.array(["O","O2","N2","NO"])
    
    # =======================================================================
    # These cross section values are from Karthik's model files
    # =======================================================================

    solspec["xsec"]["abs"]["O"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 3.79e+00, 4.10e+00, 3.00e+00, 4.79e+00,
                            8.52e+00, 1.31e+01, 1.07e+01, 7.72e+00, 6.02e+00,
                            3.78e+00, 1.32e+00, 3.25e-01, 1.05e-01, 1.13e-01,
                            1.70e-02, 2.27e-03]) * 1e-22
    solspec["xsec"]["abs"]["O2"]  = np.array([5.00e-01, 1.50e+00, 3.40e+00, 6.00e+00, 1.00e+01,
                           1.30e+01, 1.50e+01, 1.20e+01, 2.20e+00, 4.00e-01,
                           1.30e+01, 1.00e-02, 1.40e+00, 4.00e-01, 1.00e+00,
                           1.15e+00, 1.63e+00, 1.87e+01, 3.25e+01, 1.44e+01,
                           1.34e+01, 1.33e+01, 1.09e+01, 1.05e+01, 2.49e+01,
                           2.36e+01, 2.70e+01, 2.03e+01, 1.68e+01, 1.32e+01,
                           7.63e+00, 2.63e+00, 6.46e-01, 2.10e-01, 2.25e-01,
                           3.40e-02, 4.54e-03]) * 1e-22
    solspec["xsec"]["abs"]["N2"]  = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 2.55e+00, 1.15e+02, 1.44e+01,
                           2.18e+00, 7.17e+01, 1.31e+01, 2.14e+00, 5.45e+01,
                           2.30e+01, 2.31e+01, 1.97e+01, 1.17e+01, 9.94e+00,
                           5.09e+00, 1.53e+00, 3.46e-01, 1.14e+00, 1.41e-01,
                           2.01e-02, 2.53e-03]) * 1e-22
    solspec["xsec"]["abs"]["NO"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 1.85e+00, 2.04e+00,
                           2.04e+00, 0.00e+00, 2.41e+00, 3.70e+00, 6.48e+00,
                           3.70e+00, 7.82e+00, 1.98e+01, 2.41e+01, 2.41e+01,
                           2.41e+01, 1.55e+01, 1.55e+01, 1.55e+01, 1.36e+01,
                           1.40e+01, 2.00e+01, 2.02e+01, 2.30e+01, 2.40e+01,
                           2.22e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22
    solspec["xsec"]["abs"]["CO"]  = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 1.92e+00, 3.53e+00, 5.48e+00, 8.02e+00,
                           1.00e+01, 1.17e+01, 1.10e+01, 1.25e+01, 1.25e+01,
                           1.54e+01, 1.47e+01, 1.95e+01, 2.01e+01, 2.01e+01,
                           2.01e+01, 2.18e+01, 2.18e+01, 2.18e+01, 2.23e+01,
                           2.22e+01, 2.38e+01, 1.86e+01, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22
    solspec["xsec"]["abs"]["aCO2"]  = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 4.42e+00, 7.51e+00, 1.10e+01, 1.50e+01,
                           1.79e+01, 2.12e+01, 2.00e+01, 2.34e+01, 2.34e+01,
                           2.57e+01, 2.49e+01, 2.83e+01, 2.93e+01, 2.93e+01,
                           2.93e+01, 3.21e+01, 3.21e+01, 3.21e+01, 3.20e+01,
                           2.57e+01, 3.36e+01, 2.02e+01, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22
    solspec["xsec"]["abs"]["O3"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 1.25e+01, 9.20e+00,
                           9.20e+00, 9.20e+00, 9.16e+00, 9.50e+00, 9.50e+00,
                           9.50e+00, 1.47e+01, 1.47e+01, 1.47e+01, 2.74e+01,
                           2.02e+01, 3.33e+01, 7.74e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22
    solspec["xsec"]["abs"]["bCO2"] = np.array([5.00e-02, 1.00e-01, 1.50e-01, 3.00e-01, 4.00e-01,
                           5.50e-01, 5.00e-01, 5.00e-01, 8.00e-01, 5.00e-01,
                           2.00e-01, 0.00e+00, 0.00e+00, 1.85e+01, 1.48e+01,
                           1.42e+01, 1.66e+01, 3.98e+01, 7.41e+01, 7.41e+01,
                           7.41e+01, 1.74e+01, 1.74e+01, 1.74e+01, 3.23e+01,
                           3.31e+01, 3.33e+01, 2.50e+01, 2.34e+01, 2.33e+01,
                           1.11e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22
    solspec["xsec"]["abs"]["H2O"] = np.array([5.00e+00, 5.00e+00, 5.00e+00, 3.00e+00, 1.50e+00,
                           8.00e-01, 8.00e-01, 1.10e+00, 5.00e+00, 8.00e+00,
                           8.00e+00, 0.00e+00, 4.44e+00, 4.44e+00, 4.44e+00,
                           1.41e+01, 2.46e+01, 1.10e+01, 1.85e+01, 1.85e+01,
                           1.85e+01, 2.36e+01, 2.36e+01, 2.36e+01, 3.98e+01,
                           2.33e+01, 2.28e+01, 2.45e+01, 2.22e+01, 1.91e+01,
                           1.11e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22
    solspec["xsec"]["abs"]["CO2"]  = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 4.42e+00, 7.51e+00, 1.10e+01, 1.50e+01,
                           1.79e+01, 2.12e+01, 2.00e+01, 2.34e+01, 2.34e+01,
                           2.57e+01, 2.49e+01, 2.83e+01, 2.93e+01, 2.93e+01,
                           2.93e+01, 3.21e+01, 3.21e+01, 3.21e+01, 3.20e+01,
                           2.57e+01, 3.36e+01, 2.02e+01, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00]) * 1e-22

    """
    Describe the branching ratios
    """
    solspec["xsec"]["br_ratio"] = {"O2" : {}, "N2" : {}, "O" : {}, "NO" : {}, "CO": {}, "CO2": {}, "H2O": {}}
    solspec["xsec"]["br_ratio"]["O2"]["sub_type"] = np.array(["O2+","D1","Dissoc"])
    solspec["xsec"]["br_ratio"]["N2"]["sub_type"] = np.array(["N2+","N1","Dissoc"])
    solspec["xsec"]["br_ratio"]["O"]["sub_type"] = np.array(["4S","2D","2P"])
    solspec["xsec"]["br_ratio"]["NO"]["sub_type"] = np.array(["NO"])
    solspec["xsec"]["br_ratio"]["CO2"]["sub_type"] = np.array(["CO2"])
    solspec["xsec"]["br_ratio"]["H2O"]["sub_type"] = np.array(["H2O"])
    solspec["xsec"]["br_ratio"]["CO"]["sub_type"] = np.array(["CO"])
    
    # ===================================================
    # Branching values
    # ===================================================
    solspec["xsec"]["br_ratio"]["O2"]["values"] = {"O2+": None,"D1": None, "Dissoc": None}
    solspec["xsec"]["br_ratio"]["N2"]["values"] = {"N2+": None,"N1": None, "Dissoc": None}
    solspec["xsec"]["br_ratio"]["O"]["values"] = {"4S": None,"2D": None, "2P": None}
    solspec["xsec"]["br_ratio"]["NO"]["values"] = {"NO" : None}
    solspec["xsec"]["br_ratio"]["CO"]["values"] = {"CO" : None}
    solspec["xsec"]["br_ratio"]["CO2"]["values"] = {"CO2" : None}
    solspec["xsec"]["br_ratio"]["H2O"]["values"] = {"H2O" : None}

    solspec["xsec"]["br_ratio"]["NO"]["values"]["NO"] =  np.array([1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00])
    solspec["xsec"]["br_ratio"]["H2O"]["values"]["H2O"] =  np.array([1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00])
    solspec["xsec"]["br_ratio"]["CO2"]["values"]["CO2"] =  np.array([1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00])
    solspec["xsec"]["br_ratio"]["CO"]["values"]["CO"] =  np.array([1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00])
    solspec["xsec"]["br_ratio"]["O2"]["values"]["O2+"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.613,
                           0.830, 0.620, 0.786, 0.756, 0.534,
                           0.574, 0.549, 0.477, 0.672, 0.874,
                           0.759, 0.649, 0.624, 0.553, 0.347,
                           0.108, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])
    solspec["xsec"]["br_ratio"]["O2"]["values"]["D1"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.001, 0.108,
                           0.240, 0.351, 0.376, 0.447, 0.653,
                           0.892, 1.0, 1.0, 1.0, 1.0])
    solspec["xsec"]["br_ratio"]["O2"]["values"]["Dissoc"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.387,
                           0.170, 0.380, 0.214, 0.244, 0.466,
                           0.426, 0.451, 0.524, 0.327, 0.017,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])
    solspec["xsec"]["br_ratio"]["N2"]["values"]["N2+"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.429, 0.679, 1.000,
                           0.996, 0.908, 0.754, 0.747, 0.751,
                           0.717, 0.040, 0.040, 0.040, 0.040])
    solspec["xsec"]["br_ratio"]["N2"]["values"]["N1"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.005, 0.093, 0.246, 0.253, 0.249,
                           0.282, 0.960, 0.960, 0.960, 0.960])
    solspec["xsec"]["br_ratio"]["N2"]["values"]["Dissoc"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 0.571, 0.320, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])
    solspec["xsec"]["br_ratio"]["O"]["values"]["4S"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00,
                           1.00e+00, 1.00e+00, 0.930, 0.655, 0.298,
                           0.317, 0.346, 0.350, 0.367, 0.389,
                           0.393, 0.390, 0.390, 0.390, 0.390])
    solspec["xsec"]["br_ratio"]["O"]["values"]["2D"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.070, 0.337, 0.451,
                           0.424, 0.403, 0.402, 0.392, 0.377,
                           0.374, 0.378, 0.378, 0.378, 0.378])
    solspec["xsec"]["br_ratio"]["O"]["values"]["2P"] =  np.array([0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                           0.00e+00, 0.00e+00, 0.00e+00, 0.009, 0.252,
                           0.260, 0.246, 0.241, 0.233, 0.227,
                           0.226, 0.224, 0.224, 0.224, 0.224])
    
    """
    Describe secondary ionization ratios for O, N2 and O2
    """
    solspec["eta"]["_desc"] = "Secondary ionization ratio per EUVAC bin"
    solspec["eta"]["N2"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.031, 0.178, 0.361,
                            0.933, 2.859, 7.789, 10.834, 32.162,
                            80.880, 342.66])
    solspec["eta"]["O"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.127, 0.418, 0.694,
                            1.092, 2.192, 4.995, 71.378, 23.562,
                            50.593, 217.12])
    solspec["eta"]["O2"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.024, 0.105, 0.242,
                            0.579, 1.613, 4.271, 59.953, 20.290,
                            50.156, 210.83])
    solspec["eta"]["NO"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00])
    solspec["eta"]["CO"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00])
    solspec["eta"]["CO2"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00])
    solspec["eta"]["H2O"] = np.array([0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                            0.00e+00, 0.00e+00])

    """
    Threshold ionization wavelength in meters
    """
    solspec["ionization_wavelength"] = {"O2" : 102.8e-9, "O" : 91.03e-9, "N2" : 79.58e-9,\
           "NO" : 133.8e-9, "CO" : 88.49e-9, "H2O": 98.42e-9, "CO2": 90.04e-9}

    @staticmethod
    def get_solar_flux_by_f107(f107d, f107a):
        """
        This method is used to get the solar flux data for a given day by f107 and f107a
    
        f107d <float> = F10.7 data (Daily reading)
        f107a <float> = F10.7 81 day average data
    
        local <dict>
        """
        P = (f107d + f107a)/2.
        local = EuvacBase.solspec
        if P < 80.: local["current"] = 0.8*local["reference"] * 1e4
        else: local["current"] = local["reference"]*(1. + local["afac"]*(P-80.)) * 1e4
        local["wave1"] = local["wave1"] * 1e-10
        local["wave2"] = local["wave2"] * 1e-10
        local["_desc"] = "Tag 'current' provides the solar spectrum, photon in m^{-2} s^{-1}; 'wave1' and 'wave2' are the wavelengths in m [ bin boundaries (lower and upper)]"
        return local

    @staticmethod
    def get_solar_flux_by_day(dn):
        """self.local
        This method is used to estimate solar specific incomming flux for a day [37 EUVAC bins]
        
        dn <datetime> = Date of interest
        
        local <dict>
        """
        _, _, f107d, f107a, _, _,  _, _, _ = pyglow.get_kpap.get_kpap(dn)
        local = EuvacBase.get_solar_flux_by_f107(f107d, f107a)
        return local



# ============================================================================= 
# This part is only used for the photoionization (primary and secondary) and 
# modified by GOES X-Ray data lower bins
# =============================================================================
class Euvac(EuvacBase):
    """
    This model extends EUVAC model and build on top of that
    """
    
    def __init__(self, dn, start, end):
        self.dn = dn
        self.data_dict = utils.read_goes(self.dn)
        self.local = EuvacBase.get_solar_flux_by_day(self.dn)
        lambdas = np.array([np.mean([self.local["GOES_wave1"][0], self.local["GOES_wave2"][0]]), 
                    np.mean([self.local["GOES_wave1"][1], self.local["GOES_wave2"][1]])])
        
        binedE = pconst["h"] * pconst["c"] / lambdas
        xray = self.data_dict[(self.data_dict.date>=start) & (self.data_dict.date<=end)]
        W = xray[["A_AVG","B_AVG"]].values
        pe0 = W[:,0] / binedE[0]
        pe1 = W[:,1] / binedE[1]
        print("\n Max-min range for smaller wavelength bin (0.05-0.4) nm: " + \
                "%s -> %s"%(np.format_float_scientific(max(pe0),precision=2), np.format_float_scientific(min(pe0),precision=2)))
        print(" Max-min range for longer wavelength bin (0.1-0.8) nm: " + \
                "%s -> %s"%(np.format_float_scientific(max(pe1),precision=2), np.format_float_scientific(min(pe1),precision=2)))
        self.local["ref_GOES"] = np.array([pe0,pe1])
        self._update_xsec()
        return

    def _update_xsec(self):
        """
        Update cross section for all the species (minor)
        """
        species = {"NO":[31,37],"H2O":[31,37],"CO":[31,37],"CO2":[31,37]}
        for sps in species:
            rng = species[sps]
            for i in range(rng[0],rng[1]):
                self.local["xsec"]["abs"][sps][i] = self.get_Xsec(self.local["wave1"][i], self.local["wave2"][i], sps)
                pass
            pass
        return

    def get_Xsec(self, wave1, wave2, sps):
        """
        This method is used to modify the absorption cross section for the
        """
        w = np.linspace(wave1,wave2,10)
        gram_per_mole = {"CO2": 44., "CO": 28., "NO": 30., "H2O": 18.}
        df = pd.read_csv("config/xsec.%s.csv"%sps,sep="|")
        lamb = pconst["h"] * pconst["c"] / (np.array(df["Photon"]) * 1e6 \
                * pconst["q_e"])
        uxsec = (np.array(df["TotC"]) + np.array(df["TotI"])) \
                * (gram_per_mole[sps] / pconst["avo"]) * 1e-3
        func = utils.extrap1d(lamb, uxsec)
        xsec = func(w)
        return np.max(xsec)
    
    @staticmethod
    def get_solar_flux(dn, start, end):
        _euvac = Euvac(dn, start, end)
        return _euvac


if __name__ == "__main__":
    Euvac.get_solar_flux(dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,16), dt.datetime(2015,3,11,17))
    pass
