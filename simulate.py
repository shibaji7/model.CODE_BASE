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

import utils
from model import Model
from plot_lib import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prog", default="flare", help="Program code [bgc/flare] (default bgc)")
    parser.add_argument("-r", "--rio", default="ott", help="Riometer code (default ott)")
    parser.add_argument("-ev", "--event", default=dt.datetime(2015,3,11,16,22), help="Start date (default 2015-3-11T16:22)",
                        type=dparser.isoparse)
    parser.add_argument("-s", "--start", default=dt.datetime(2015,3,11,15,30), help="Start date (default 2015-3-11T15:30)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2015,3,11,17,30), help="End date (default 2015-3-11T17:30)",
            type=dparser.isoparse)
    parser.add_argument("-g", "--save_goes", action="store_false", help="Save goes data (default True)")
    parser.add_argument("-sat", "--sat", type=int, default=15, help="Satellite number (default 15)")
    parser.add_argument("-rm", "--save_riom", action="store_false", help="Save riometer data (default True)")
    parser.add_argument("-ps", "--plot_summary", action="store_true", help="Plot summary report (default False)")
    parser.add_argument("-sr", "--save_result", action="store_false", help="Save results (default True)")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-irr", "--irradiance", default="EUVAC+", help="Irradiance model (default EUVAC+)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    parser.add_argument("-pc", "--plot_code", type=int, default=0, help="Plotting code,applicable if --prog==plot (default 0)")
    parser.add_argument("-fr", "--frequency", type=float, default=30, help="Frequency of oprrations in MHz (default 30 MHz)")
    args = parser.parse_args()
    utils.store_cmd(args)
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     " , k , "->" , str(vars(args)[k]))
    if args.prog == "bgc": 
        os.system("python2.7 models/bgc.py -r {r} -ev {ev} -s {s} -e {e} -v -fr {fr}".format(r=args.rio, 
            ev=args.event.strftime("%Y-%m-%dT%H:%M"), s=args.start.strftime("%Y-%m-%dT%H:%M"), 
            e=args.end.strftime("%Y-%m-%dT%H:%M"), fr=args.frequency))
        utils.add_chi(args.event, args.rio, args.start, args.end)
        print("\n SZA estimation done")
    elif args.prog == "flare": Model(args.rio, args.event, args).run()
    elif args.prog == "plot":
        if args.plot_code == 0: plot_rio_locations() 
        else: print("\n Plotting function is not implemented")
    else: print("\n Program not implemented")
    print("")
    if os.path.exists("models/__pycache__"): os.system("rm -rf models/__pycache__")
