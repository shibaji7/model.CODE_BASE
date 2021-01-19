#!/usr/bin/env python

"""fetch_data.py: Fetch data from the remote locations"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import pandas as pd
import datetime as dt
from netCDF4 import Dataset, num2date
import calendar

import paramiko
import os
from cryptography.fernet import Fernet
import json

LFS = "LFS/LFS_riometer_data/"

class Conn2Remote(object):
    
    def __init__(self, host, user, key_filename, port=22, passcode=None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
        self.passcode = passcode
        self.port = port
        self.con = False
        if passcode: self.decrypt()
        self.conn()
        return
    
    def decrypt(self):
        passcode = bytes(self.passcode, encoding="utf8")
        cipher_suite = Fernet(passcode)
        self.user = cipher_suite.decrypt(bytes(self.user, encoding="utf8")).decode("utf-8")
        self.host = cipher_suite.decrypt(bytes(self.host, encoding="utf8")).decode("utf-8")
        return
    
    def conn(self):
        if not self.con:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname=self.host, port = self.port, username=self.user, key_filename=self.key_filename)
            self.scp = paramiko.SFTPClient.from_transport(self.ssh.get_transport())
            self.con = True
        return
    
    def close(self):
        if self.con:
            self.scp.close()
            self.ssh.close()
        return
    
    def chek_remote_file_exists(self, fname):
        if self.con:
            try:
                self.scp.stat(LFS+fname)
                return True
            except FileNotFoundError:
                return False
        else: return False
        
    def to_remote_FS(self, local_file, is_local_remove=False):
        remote_file = LFS + local_file
        print(" To file:", remote_file)
        self.scp.put(local_file, remote_file)
        if is_local_remove: os.remove(local_file)
        return
    
    def from_remote_FS(self, local_file):
        remote_file = LFS + local_file
        print(" From file:", remote_file)
        self.scp.get(remote_file, local_file)
        return
    
    def create_remote_dir(self, ldir):
        rdir = LFS + ldir
        self.ssh.exec_command("mkdir -p " + rdir)
        return
    
def encrypt(host, user, filename="config/passcode.json"):
    passcode = Fernet.generate_key()
    cipher_suite = Fernet(passcode)
    host = cipher_suite.encrypt(bytes(host, encoding="utf8"))
    user = cipher_suite.encrypt(bytes(user, encoding="utf8"))
    with open(filename, "w") as f:
        f.write(json.dumps({"user": user.decode("utf-8"), "host": host.decode("utf-8"), "passcode": passcode.decode("utf-8")},
                           sort_keys=True, indent=4))
    return

def get_pubfile():
    with open("config/pub.json", "r") as f:
        obj = json.loads("".join(f.readlines()))
        pubfile = obj["pubfile"]
    return pubfile

def get_session(filename="config/passcode.json", key_filename=None, isclose=False):
    if key_filename is None: key_filename = get_pubfile()
    with open(filename, "r") as f:
        obj = json.loads("".join(f.readlines()))
        conn = Conn2Remote(obj["host"], obj["user"], 
                           key_filename=key_filename, 
                           passcode=obj["passcode"])
    if isclose: conn.close()    
    return conn


class Goes(object):
    
    def __init__(self):
        return
    
    def download_goes_data(self, conn, dn, sat=15, v=True):
        """ Download GOES data """
        def _get_month_bounds_(start_time):
            """ This method is used to get the first and last date of the month """
            month_start = start_time.replace(day = 1).strftime("%Y%m%d")
            _, month_end = calendar.monthrange(start_time.year, start_time.month)
            month_end = (start_time.replace(day = 1) + dt.timedelta(days=month_end-1)).strftime("%Y%m%d")
            return month_start, month_end
        fname = "proc/goes/{dnx}.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
        if not conn.chek_remote_file_exists(fname+".gz"):
            month_start, month_end = _get_month_bounds_(dn)
            url = "https://satdat.ngdc.noaa.gov/sem/goes/data/avg/{year}/{month}/goes{sat}/netcdf/"\
                    "g{sat}_xrs_1m_{mstart}_{mend}.nc".format(year=dn.year, month="%02d"%dn.month, sat=sat, 
                            mstart=month_start, mend=month_end)
            if v: print("\n Download file -from- " + url)
            tag_vars = ["A_AVG","B_AVG"]
            fn = fname.replace(".csv",".nc")
            os.system("wget -O {fn} {url}".format(fn=fn, url=url))
            if os.path.exists(fn):
                nc = Dataset(fn)
                tt = nc.variables["time_tag"]
                jd = np.array(num2date(tt[:],tt.units))
                data = {}
                for var in tag_vars:  data[var] = nc.variables[var][:]
                data["date"] = jd
                data_dict = pd.DataFrame(data)
                data_dict.to_csv(fname, index=False, header=True)
                os.system("gzip {fname}".format(fname=fname))
                if v: print("\n File saved  -to- " + fname)
                os.remove(fn)
                conn.to_remote_FS(fname + ".gz")
            else: print(" Unable to download file.")
        return
    
    def get_goes_file(self, conn, dn):
        close, file_name = False, None
        if conn==None: close, conn = True, get_session()
        fname = "proc/goes/{dnx}.csv.gz".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
        if conn.chek_remote_file_exists(fname):
            conn.from_remote_FS(fname)
            os.system("gzip -d {fname}".format(fname=fname))
            fname = fname.replace(".gz", "")
        else: print(" File does not exists:" + fname)
        if close: conn.close()
        return fname
    
    def clean_local_file(self, dn):
        fname = "proc/goes/{dnx}.csv".format(dnx=dn.strftime("%Y.%m.%d.%H.%M"))
        if os.path.exists(fname): os.remove(fname)
        return
    
    @staticmethod
    def run_goes_downloads():
        conn = get_session()
        goes = Goes()
        events = pd.read_csv("config/event-stats-list.csv", parse_dates=["start","end","peak"])
        for i, e in events.iterrows():
            goes.download_goes_data(conn, e["peak"])
        conn.close()
        return
    
class Riometer(object):
    
    def __init__(self):
        return
    
    def get_riometer_file(self, conn, dn, code):
        close, file_name = False, None
        if conn==None: close, conn = True, get_session()
        fname = "proc/riometer/{year}/{code}{dnx}_03.txt.gz".format(year=dn.year,code=code,dnx=dn.strftime("%Y%m%d"))
        if conn.chek_remote_file_exists(fname):
            conn.from_remote_FS(fname)
            os.system("gzip -d {fname}".format(fname=fname))
            fname = fname.replace(".gz", "")            
            file_name = "proc/riometer/{year}/{code}_{dnx}.csv".format(year=dn.year,code=code,dnx=dn.strftime("%Y.%m.%d.%H.%M"))
            self.parse_riometer(dn.strftime("%Y%m%d "), fname, file_name)
            os.remove(fname)
        else: print(" File does not exists:" + fname)
        if close: conn.close()
        return file_name
    
    def parse_riometer(self, day, fname, file_name):
        with open(fname) as c: lines = c.read().split("\n")
        data = []
        for line in lines[17:-2]:
            x = np.nan
            line = list(filter(None,line.replace("\n","").split(" ")))
            try: 
                rabs, flag = float(line[8]), int(line[9])
                data.append([dt.datetime.strptime(day+line[0]+line[1]+line[2],"%Y%m%d %H%M%S"), rabs, flag])
            except: continue
        if len(data) > 0: 
            data_dict = pd.DataFrame(data, columns=["Date", "Absorp", "Flag"])
            data_dict.to_csv(file_name, index=False, header=True)
            print(" File saved  -to- " + file_name)
        return
    
    def clean_local_file(self, dn, code):
        fname = "proc/riometer/{year}/{code}_{dnx}.csv".format(year=dn.year,code=code,dnx=dn.strftime("%Y.%m.%d.%H.%M"))
        if os.path.exists(fname): os.remove(fname)
        return
    
class Simulation(object):
    
    def __init__(self, dn, code, run_type="bgc"):
        self.dn = dn
        self.code = code
        self.run_type = run_type
        return
    
    def create_remote_local_dir(self, conn=None):
        close = False
        if conn==None: close, conn = True, get_session()
        self._dir_ = "proc/outputs/{dnx}/{code}/".format(code=self.code,dnx=self.dn.strftime("%Y.%m.%d.%H.%M"))
        if not os.path.exists(self._dir_): os.system("mkdir -p " + self._dir_)
        conn.create_remote_dir(self._dir_)
        if close: conn.close()
        return
    
    def clear_local_folders(self):
        _dir_ = "proc/outputs/{dnx}/".format(code=self.code,dnx=self.dn.strftime("%Y.%m.%d.%H.%M"))
        if os.path.exists(_dir_): os.system("rm -rf " + _dir_)
        return
    
    def save_bgc_file(self, conn=None):
        bgc_file = self._dir_ + "bgc.nc.gz"
        close = False
        if conn==None: close, conn = True, get_session()
        if os.path.exists(bgc_file): conn.to_remote_FS(bgc_file)
        if close: conn.close()
        return
    
    def get_bgc_file(self, conn=None):
        bgc_file = self._dir_ + "bgc.nc.gz"
        close = False
        if conn==None: close, conn = True, get_session()
        if conn.chek_remote_file_exists(bgc_file): conn.from_remote_FS(bgc_file)
        if close: conn.close()
        return
    
    def save_flare_file(self, conn=None):
        flare_file = self._dir_ + "flare.nc.gz"
        close = False
        if conn==None: close, conn = True, get_session()
        if os.path.exists(flare_file): conn.to_remote_FS(flare_file)
        if close: conn.close()
        return
    
    def get_flare_file(self, conn=None):
        flare_file = self._dir_ + "flare.nc.gz"
        close = False
        if conn==None: close, conn = True, get_session()
        if conn.chek_remote_file_exists(flare_file): conn.from_remote_FS(flare_file)
        if close: conn.close()
        return

if __name__ == "__main__":
    ################################
    # Fetch GOES bulk request
    ################################
    Goes.run_goes_downloads()
    ################################
    # Test GOES request
    ################################
    Goes().get_goes_file(None, dt.datetime(2015,3,11,16,22))
    ################################
    # Test Riometer request
    ################################
    Riometer().get_riometer_file(None, dt.datetime(2015,3,11,16,22), "ott")