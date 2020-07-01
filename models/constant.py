#!/usr/bin/env python

"""constant.py: constant instatntiate all the constants."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np

"""
Model constants
"""
model = {
        "alt_lower_bound":10.,
        "alt_upper_bound":500.,
        "alts": np.arange(10,500),
        }

"""
Physical constants
"""
pconst = {
            "boltz" : 1.38066e-23,       # Boltzmann constant  in Jule K^-1
            "h"     : 6.626e-34  ,       # Planks constant  in ergs s
            "c"     : 2.9979e08  ,       # in m s^-1
            "avo"   : 6.023e23   ,       # avogadro's number           
            "Re"    : 6371.e3    ,
            "amu"   : 1.6605e-27 ,
            "q_e"   : 1.602e-19  ,       # Electron charge in C
            "m_e"   : 9.109e-31  ,       # Electron mass in kg
            "g"     : 9.81       ,       # Gravitational acceleration on the surface of the Earth
            "eps0"  : 1e-9/(36*np.pi),
            "R"     : 8.31       ,       # J mol^-1 K^-1
            }

mass =  {
        "O3"    : 48. ,
        "O2"    : 32. ,
        "O"     : 16. ,
        "N2"    : 28. ,
        "AR"    : 40. ,
        "Na"    : 23. ,
        "He"    : 4.  ,
        "NO"    : 30. ,
        "N4s"   : 14. ,
        "N2d"   : 14. ,
        "CH4"   : 16. ,
        "H2"    : 2.  ,
        "CO"    : 28. ,
        "CO2"   : 44. ,
        "H2O"   : 18. ,
        "Hox"   : 1.  ,
        "H"     : 1.
        }

reaction_rate_constants = {
                # N2+ + e- -> N + N
                "N2" : lambda Te, Ti, Tn: 2.0e-7 * (300/Tn)**.39 * 1e-6,
                # O2+ + e- -> O + O
                "O2" : lambda Te, Ti, Tn: 1.95e-7 * (300/Tn)**.70 * 1e-6,
                # O+ + e- -> O + hv
                "O"  : lambda Te, Ti, Tn: 3.7e-12 * (250/Te)**.70 * 1e-6,
                }


EUVAC_IONIZATION_MULTIPLIER = 60
FDM_IONIZATION_MULTIPLIER = 60
FFM_IONIZATION_MULTIPLIER = 60

