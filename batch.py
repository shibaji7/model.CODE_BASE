#!/usr/bin/env python

"""batch.py: simulate python program for absorption calculation for all riometers"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import datetime as dt
import argparse
from dateutil import parser as dparser

import sys
sys.path.append("models/")

import utils
from model import Model

def _bgc_(args):
    """ Run background models """

    def _run_(r):
        cmd = "python2.7 models/bgc.py -r {r} -ev {ev} -s {s} -e {e} -v -fr {fr}".format(r=r,
                ev=args.event.strftime("%Y-%m-%dT%H:%M"), s=args.start.strftime("%Y-%m-%dT%H:%M"),                                                               e=args.end.strftime("%Y-%m-%dT%H:%M"), fr=args.frequency)
        os.system(cmd)
        utils.add_chi(args.event, r, args.start, args.end)
        if args.verbose: print("\n SZA estimation done")
        if args.verbose: print("\n Download done.")
        return

    inp = tqdm(pd.read_csv("config/riometers.csv")["rio"].tolist())
    processed_list = Parallel(n_jobs=8)(delayed(_run_)(i) for i in inp)
    return

def _flare_(args):
    """ Run flare models """

    def _run_(r):
        Model(r, args.event, args).run()
        return

    inp = tqdm(pd.read_csv("config/riometers.csv")["rio"].tolist())
    for i in inp:
        _run_(i)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prog", default="bgc", help="Program code [bgc/flare] (default bgc)")
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
    parser.add_argument("-sps", "--species", type=int, default=0, help="Species Type (default 0)")
    args = parser.parse_args()
    x = pd.read_csv("config/flares.csv",parse_dates=["dn","start","end"])
    if args.prog == "bgc": 
        u = x[x.bgc!="Y"]
        for i, _u in u.iterrows():
            args.start, args.end, args.event = _u["start"] - dt.timedelta(minutes=0), _u["end"] + dt.timedelta(minutes=0), _u["dn"]
            if args.verbose:
                print("\n Parameter list for simulation ")
                for k in vars(args).keys():
                    print("     " , k , "->" , str(vars(args)[k]))
            _bgc_(args)
            x.at[x.dn==args.event,"bgc"] = "Y"
        x.to_csv("config/flares.csv", header=True, index=False)
    elif args.prog == "flare": 
        u = x[x.flare!="Y"]
        for i, _u in u.iterrows():
            args.start, args.end, args.event = _u["start"] - dt.timedelta(minutes=0), _u["end"] + dt.timedelta(minutes=0), _u["dn"]
            if args.verbose:
                print("\n Parameter list for simulation ")
                for k in vars(args).keys():
                    print("     " , k , "->" , str(vars(args)[k]))
            _flare_(args)
            x.at[x.dn==args.event,"flare"] = "Y"
        x.to_csv("config/flares.csv", header=True, index=False)
    else: print("\n Program not implemented")
    print("")
    if os.path.exists("models/__pycache__"): os.system("rm -rf models/__pycache__")
