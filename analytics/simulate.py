#!/usr/bin/env python

"""simulate.py: simulate python program for absorption calculation"""

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

np.random.seed(0)

import sys
sys.path.append("models/")
import datetime as dt
import argparse
from dateutil import parser as dparser

from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rio", default="ott", help="Riometer code (default ott)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,20), help="Start date (default 2015-3-11T16:22)",
                        type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,16), help="Start date (default 2015-3-11T15:30)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,17), help="End date (default 2015-3-11T17:30)",
            type=dparser.isoparse)
    parser.add_argument("-ps", "--plot_summary", action="store_true", help="Plot summary report (default False)")
    parser.add_argument("-sr", "--save_result", action="store_false", help="Save results (default True)")
    parser.add_argument("-irr", "--irradiance", default="EUVAC+", help="Irradiance model (default EUVAC+)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    parser.add_argument("-pc", "--plot_code", type=int, default=0, help="Plotting code,applicable if --prog==plot (default 0)")
    parser.add_argument("-sps", "--species", type=int, default=0, help="Species Type (default 0)")
    parser.add_argument("-fr", "--frequency", type=float, default=30, help="Frequency of oprrations in MHz (default 30 MHz)")
    parser.add_argument("-p", "--prog", default="flare", help="Program code [flare] (default bgc)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     " , k , "->" , str(vars(args)[k]))
    if args.prog == "flare": Model(args.rio, args.event, args).run()
    else: print("\n Program not implemented")
    print("")
    if os.path.exists("models/__pycache__"): os.system("rm -rf models/__pycache__")