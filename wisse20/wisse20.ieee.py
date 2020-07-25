#!/usr/bin/env python

"""wisse20.ieee.py: WISSE 2020 simulation"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import datetime as dt

evs = [dt.datetime(2013,5,13,16,5), dt.datetime(2014,3,29,17,48), dt.datetime(2014,6,10,11,42), dt.datetime(2015,3,11,16,22), ]
starts = [dt.datetime(2013,5,13,15,40), dt.datetime(2014,3,29,17,30), dt.datetime(2014,6,10,11,20), dt.datetime(2015,3,11,15,30)]
ends = [dt.datetime(2013,5,13,17), dt.datetime(2014,3,29,18,30), dt.datetime(2014,6,10,12,30), dt.datetime(2015,3,11,17,30)]

for p in ["bgc", "flare"]:
    for ev, start, end in zip(evs, starts, ends)
        cmd = "python simulate.py -ev {ev} -s {s} -e {e} -p {p}".format(ev=ev.strftime("%Y-%m-%dT%H:%M"),
                s=start.strftime("%Y-%m-%dT%H:%M"), e=end.strftime("%Y-%m-%dT%H:%M"), p=p)
        os.system(cmd)

if os.path.exists("models/__pycache__"): os.system("rm -rf models/__pycache__")
if os.path.exists("models/experiments/__pycache__"): os.system("rm -rf models/experiments/__pycache__")
