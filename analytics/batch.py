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

import datetime as dt
import argparse
import pandas as pd


def batch_mode_bgc_run(start, end, cls):
    command = "python bgc.py -ev {ev} -s {start} -e {end} -r {rio}"
    riometers = pd.read_csv("config/riometers.csv").rio.tolist()
    events = pd.read_csv("config/event-stats-list.csv", parse_dates=["start", "end", "peak"])
    events = events[events["class"].str.contains(cls)]
    for i, event in events.iterrows():
        st, ev, ed = event["peak"] - dt.timedelta(minutes=start), event["peak"], event["peak"] + dt.timedelta(minutes=end)
        for rio in riometers:
            cmd = command.format(ev=ev.strftime("%Y-%m-%dT%H:%S"), start=st.strftime("%Y-%m-%dT%H:%S"), 
                    end=ed.strftime("%Y-%m-%dT%H:%S"), rio=rio)
            print(" Command - ", cmd)
            os.system(cmd)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cls", default="X", help="Event class type")
    parser.add_argument("-s", "--start", default=10, help="Start minutes delayed by some time", type=float)
    parser.add_argument("-e", "--end", default=50, help="End minutes delayed by some time", type=float)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for Bgc simulation ")
        for k in vars(args).keys():
            print("     " + k + "->" + str(vars(args)[k]))
    batch_mode_bgc_run(args.start, args.end, args.cls)
