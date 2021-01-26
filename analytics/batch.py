"""batch.py: Module is used to run the codes in a batch job format to store it in """

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"



import os
import dask

import numpy as np
import datetime as dt
import argparse
import pandas as pd
from netCDF4 import Dataset, num2date
import traceback

import sys
sys.path.append("models/")
import utils

from fetch_data import Goes, Riometer, Simulation, get_session

def batch_mode_bgc_run(start, end, cls):
    command = "python bgc.py -ev {ev} -s {start} -e {end} -r {rio}"
    riometers = pd.read_csv("config/riometers.csv").rio.tolist()
    events = pd.read_csv("config/event-stats-list.csv", parse_dates=["start", "end", "peak"])
    events = events[events["class"].str.contains(cls)]
    for i, event in events.iterrows():
        st, ev, ed = event["peak"] - dt.timedelta(minutes=start), event["peak"], event["peak"] + dt.timedelta(minutes=end)
        for rio in riometers:
            cmd = command.format(ev=ev.strftime("%Y-%m-%dT%H:%M:%S"), start=st.strftime("%Y-%m-%dT%H:%M:%S"), 
                    end=ed.strftime("%Y-%m-%dT%H:%M:%S"), rio=rio)
            print(" Command - ", cmd)
            os.system(cmd)
    return

def batch_mode_flare_run(start, end, cls):
    command = "python simulate.py -ev {ev} -s {start} -e {end} -r {rio}"
    riometers = pd.read_csv("config/riometers.csv").rio.tolist()
    events = pd.read_csv("config/event-stats-list.csv", parse_dates=["start", "end", "peak"])
    events = events[events["class"].str.contains(cls)]
    for i, event in events.iterrows():
        st, ev, ed = event["peak"] - dt.timedelta(minutes=start), event["peak"], event["peak"] + dt.timedelta(minutes=end)
        for rio in riometers:
            cmd = command.format(ev=ev.strftime("%Y-%m-%dT%H:%M:%S"), start=st.strftime("%Y-%m-%dT%H:%M:%S"), 
                    end=ed.strftime("%Y-%m-%dT%H:%M:%S"), rio=rio)
            print(" Command - ", cmd)
            os.system(cmd)
    return

def batch_mode_stats_run(start, end, cls):
    riometers = pd.read_csv("config/riometers.csv").rio.tolist()
    events = pd.read_csv("config/event-stats-list.csv", parse_dates=["start", "end", "peak"])
    events = events[events["class"].str.contains(cls)]
    conn = get_session()
    for i, event in events.iterrows():
        st, ev, ed = event["peak"] - dt.timedelta(minutes=start), event["peak"], event["peak"] + dt.timedelta(minutes=end)
        Goes().get_goes_file(conn, ev)
        for rio in riometers:
            sim = Simulation(ev, rio)
            if sim.check_riometer_data_exists(conn) and sim.check_goes_exists(conn) and\
                not sim.check_flare_not_exists(conn) and sim.check_skill_not_exists(conn):
                Riometer().get_riometer_file(conn, ev, rio)
                _dir_ = "proc/outputs/{dnx}/{code}/".format(code=rio,dnx=ev.strftime("%Y.%m.%d.%H.%M"))
                skill_file = _dir_ + "skill.nc"
                if not os.path.exists(_dir_): os.system("mkdir -p " + _dir_)
                sim.get_flare_file(conn)
                try:
                    nc = Dataset(_dir_ + "flare.nc")
                    times = num2date(nc.variables["time"][:], nc.variables["time"].units, nc.variables["time"].calendar)
                    times = np.array([x._to_real_datetime() for x in times]).astype("datetime64[ns]")
                    times = [dt.datetime.utcfromtimestamp(x.astype(int) * 1e-9) for x in times]
                    print(" Time len():", len(times))
                    model = {
                        "sn": utils.smooth(utils.int_absorption(nc.variables["abs.ah.sn.o"][:], nc["alts"][:], extpoint=68), 5),
                        "avcc": utils.smooth(utils.int_absorption(nc.variables["abs.ah.av.cc.o"][:], nc["alts"][:], extpoint=68), 5),
                        "avmb": utils.smooth(utils.int_absorption(nc.variables["abs.ah.av.mb.o"][:], nc["alts"][:], extpoint=68), 5),
                        "sw": utils.smooth(utils.int_absorption(nc.variables["abs.sw.ft.x"][:], nc["alts"][:], extpoint=68), 5),
                        "dr": nc.variables["drap"][:],
                    }
                    pfm = utils.Performance(conn, rio, ev, times, model, st, ed, bar=4., alt=np.nan)
                    pfm.run()
                    pfm._to_netcdf_(skill_file, verbose=False)
                    sim.save_skill_file(conn)
                except:
                    print(" Exception in Pfm test: ", rio, ev)
                    traceback.print_exc()
                Riometer().clean_local_file(ev, rio)
                sim.clear_local_folders()
                #break
        Goes().clean_local_file(ev)
        #break
    conn.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cls", default="X", help="Event class type")
    parser.add_argument("-s", "--start", default=10, help="Start minutes delayed by some time", type=float)
    parser.add_argument("-e", "--end", default=50, help="End minutes delayed by some time", type=float)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-p", "--prog", default="stats", help="Program code [flare/bgc/stats] (default bgc)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for Bgc simulation ")
        for k in vars(args).keys():
            print("     " + k + "->" + str(vars(args)[k]))
    if args.prog=="bgc": batch_mode_bgc_run(args.start, args.end, args.cls)
    if args.prog=="flare": batch_mode_flare_run(args.start, args.end, args.cls)
    if args.prog=="stats": batch_mode_stats_run(args.start, args.end, args.cls)
