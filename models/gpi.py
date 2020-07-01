#!/usr/bin/env python

"""gpi.py: gpi is responsible for GPI chemical reactions including recombination rates in continiuty equation."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2019, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

from scipy import interpolate
import numpy as np

import utils
from constant import *
import plot_lib as plib

class GPI(object):
    """
    This class is related to estimating primary and secondary ionization and solve the first part of the 
    continuity equation Ne + \sum{qt}. Execute method estimate production (primary + secondary) during 
    
    To calculate realistic ionization based on Schunk and Nagy 
    Ionosphere 2nd Edition 2018 book:  equation (9.24, 9.17). Based on equation 9.17 the 
    realistic_tau has been calculated.
        
    Steps of calculations:
        1. Hs (Scale height of each species) =  kT/mg
        2. EUVAC
        3. Use equation (9.24, 9.17) to estimate ionization
        4. GPI Model

    GPI Model Description: 
        1. https://doi-org.ezproxy.lib.vt.edu/10.1029/92JA01596
        2. https://core.ac.uk/download/pdf/145656035.pdf

    GPI Model incorporates the following constituents: 
        1. electron Ne, positive ions N+ = [O2+, NO2+], positive ion cluster Nx+ = [H+(H2O)n], and negetive ions N- = [O2-,CO3-,NO2-,NO3-, etc.]
        2. Charge neutrality N- + Ne = N+ + Nx+
    
    Model inputs: point_grid, EUVAC
    """
    
    def __init__(self, _o, species, _irr_model, lam_const = 1., model_type = "EUVAC+"):
        self.model_type = model_type
        self._o = _o
        self._species = species
        self._alts = _o.alts
        self._Re = pconst["Re"]
        self.gh = pconst["g"] / ( ( 1 + (self._alts/self._Re) )**2 )
        self._irr_model = _irr_model
        self._bins = len(self._irr_model.local["wave1"])
        self._exe_ionization_rate_schunk_realistic()
        self.lam_const = lam_const
        self._update_rc_()
        self._initialize_()
        return

    # ========================================================================================
    # This part is dedicated to calculate ionization rate processes
    # ========================================================================================
    def __calculate_scale_height(self):
        """
        This method is used to calculate scale heigh of different species
        """
        # ==============================================================================
        # Pressure scale height for each species - KT/mg, K=J/s,T=k, m=kg, g=m/s
        # ==============================================================================
        self._Hp = np.zeros((len(self._species), len(self._alts)))
        Tn = self._o.msis["Tn"][self._I,:]
        for I,sp in enumerate(self._species):
            self._Hp[I,:] = pconst["boltz"] * Tn / (self.gh * mass[sp] * pconst["amu"]) # in meters
            pass
        return

    def __get_tau(self, j_lam):
        """
        This method calculates realistic tau for one wavelength bin
        less than lambda ionization using equation t = sec(chi) \sum_sp (n_s * a * H_s)  (9.17)
        
        j_lam = index of the wavelength bin less than lambda ionization
        """
        tau = np.zeros(len(self._alts))
        sec_chi = 1. / np.cos(np.deg2rad(self._o.chi[self._I,:]))
        for I,sp in enumerate(self._species):
            N = self._o.msis[sp][self._I,:]
            tau = tau + (N * self._irr_model.local["xsec"]["abs"][sp][j_lam] * self._Hp[I,:])
            pass
        tau = tau * sec_chi
        return tau

    def __euvac_plus_type_data(self, j):
        """
        Get data from GOES bins
        """
        io = self._irr_model.local["current"][j]
        if j >= 35:
            j = 36-j
            io = io + self._irr_model.local["ref_GOES"][j,self._I]
            pass
        return io

    def __ffm_type_data(self, j):
        """
        Get data from FISM modified and GOES bins
        """
        j = 36 - j
        io = self._irr_model._local[j].tolist()[self._I]
        if j <= 1:
            io = self._irr_model.local["ref_GOES"][j,self._I]
            pass
        return io

    def __fdm_type_data(self, j):
        """
        Get data from FISM modified and GOES bins
        """
        io = self._irr_model.local["current"][j]
        if j >= 35:
            j = 36-j
            io = io + self._irr_model.local["ref_GOES"][j,self._I]
            pass
        return io

    def _exe_ionization_rate_schunk_realistic(self):
        """
        This method is used to calculate realistic ionization based on Schunk and Nagy 
        Ionosphere 2nd Edition 2018 book:  equation (9.24, 9.17) Based on equation 9.17 the 
        realistic_tau has been calculated.
        """

        # =========================================================
        # This is primary ionization array (photoionization)
        # =========================================================
        self.Pe = np.zeros((len(self._o.dn),len(self._species),self._bins,3,len(self._alts))) # Shape will be (time, species, wavelengths, br_ratio, heights)
        # =========================================================
        # This is secondary ionization array (impact-ionization)
        # =========================================================
        self.Se = np.zeros((len(self._o.dn),len(self._species),self._bins,3,len(self._alts))) # Shape will be (time, species, wavelengths, br_ratio, heights)


        Pe = np.zeros((len(self._o.dn),len(self._species),self._bins,3,len(self._alts)))
        Se = np.zeros((len(self._o.dn),len(self._species),self._bins,3,len(self._alts)))
        lam = self._irr_model.local["wave2"]
        for _I in range(len(self._o.dn)): # Loop over each time
            self._I = _I
            self.__calculate_scale_height()
            for i,sp in enumerate(self._species): # Loop over each species
                lam_I = self._irr_model.local["ionization_wavelength"][sp]
                N = self._o.msis[sp][self._I,:]
                br_sub_type = self._irr_model.local["xsec"]["br_ratio"][sp]["sub_type"]
                for j in range(self._bins): # Loop over each bins less than ionizing threshold wavelengths
                    if lam[j] <= lam_I:
                        if self.model_type == "EUVAC": Io = self._irr_model.local["current"][j]
                        elif self.model_type == "EUVAC+": Io = self.__euvac_plus_type_data(j)
                        elif self.model_type == "FDM": Io = self.__fdm_type_data(j)
                        elif self.model_type == "FFM": Io = self.__ffm_type_data(j)
                        ai = self._irr_model.local["xsec"]["abs"][sp][j]
                        eta = self._irr_model.local["eta"][sp][j]
                        tau = self.__get_tau(j)
                        for k,sub_type in enumerate(br_sub_type): # Loop over each energy levels
                            ps = self._irr_model.local["xsec"]["br_ratio"][sp]["values"][sub_type][j]
                            Pe[_I,i,j,k,:] = N * Io * np.exp(-tau) * ai * ps
                            Se[_I,i,j,k,:] = N * Io * np.exp(-tau) * ai * ps * eta
                            pass
                        pass
                    pass
                pass
            pass
        self.Pe = Pe
        self.Se = Se
        self.qf = None
        return

    def get_production_rate_per_minute(self, _m, type="photo", agg=True):
        """
        Helper method to fetch Pe
        Pe is in the order of species as per [.species] array
        
        _m   <int> = 0-length(dn) integer minute instance
        type <str> = photo or impact or both
        """
        ee = self.Pe[_m,:,:,:,:]
        if type == "impact": ee = self.Se[_m,:,:,:,:]
        elif type == "both": ee = ee + self.Se[_m,:,:,:,:]
        e = []
        e.append(np.sum(np.sum(ee,axis=1),axis=1)[0,:])
        e.append(np.sum(np.sum(ee,axis=1),axis=1)[1,:])
        e.append(np.sum(np.sum(ee,axis=1),axis=1)[2,:])
        e = np.array(e)
        e[np.isnan(e)] = 0.
        if agg: e = np.sum(e,axis=0)
        return e

    def q(self,t):
        """
        Provide ionization at time 't' for all heights
        """
        if self.qf is None:
            z = self.Pe + self.Se
            z = np.sum(np.sum(np.sum(z,axis=1),axis=1),axis=1)
            x,y = self._alts, np.arange(len(self._o.dn))*60
            self.qf = interpolate.interp2d(x,y,z,kind="cubic")
            pass
        qo = self.qf(self._alts,t)
        #e = self.Pe + self.Se
        #e = np.sum(np.sum(np.sum(e,axis=1),axis=1),axis=1)
        #qo = e[int(t/60),:]
        return qo

    def _update_rc_(self):
        """
        This method is used to calculate different rate constants (RC)
        Note that, all these formulas To calculate realistic RC from the above
        """
        T = self._o.iri["Te"][0,:]
        T[T < 0.] = T[60]
        T = T * 1.
        N_O2 = self._o.msis["O2"][0,:]*1e-6
        N_N2 = self._o.msis["N2"][0,:]*1e-6
        N = self._o.msis["nn"][0,:]*1e-6
        # =======================================================================
        #   lambda = Relative composition of negetive ions
        #   Taken from Rishbeth and Garriott, 1969
        # =======================================================================
        f = utils.extrap1d([45,58,69,76,83],np.log10(self.lam_const * np.array([150.,15.,1.,1e-1,1e-2])),kind="cubic")
        self.lam = 10.**f(self._alts)
        self.lam[self.lam < 0] = 0
        # =======================================================================
        #   Beta [sec^-1] = Electron attachment rate
        # =======================================================================
        self.b = 1e-31*N_O2*N_N2 + 1.4e-29*N_O2**2*(300./T)*np.exp(-600./T)
        self.b[self.b < 0.] = 0
        # =======================================================================
        #   Alpha_d [m^3 sec^-1] = Dissociative recombination rate
        # =======================================================================
        self.ad = 1e-7*1e-6
        # =======================================================================
        #   Alpha_d^c [m^3 sec^-1] = Ion cluster attachment rate
        # =======================================================================
        self.adc = 1e-6*1e-6
        # =======================================================================
        #   Alpha_i [m^3 sec^-1] = Ion ion recombination rate
        # =======================================================================
        self.ai = 1e-7*1e-6
        # =======================================================================
        #   B [sec^-1] = Effective rate of conversion +ve ion to +ve cluster ion
        # =======================================================================
        self.B = 1e-31 * N**2
        # =======================================================================
        #   # Gamma [sec^-1] = Detachment coefficient
        # =======================================================================
        self.gm = 8e-17 * N

        plib.plot_parameters(self._alts, self.lam, self.b, self.B, self.gm, self.ad, self.adc, self.ai)
        return

    # ========================================================================================
    #   This part is dedicated to solve continuity equations
    # ========================================================================================
    
    def _initialize_(self):
        self.timestamp = np.arange(len(self._o.dn)*60).astype(int)
        self.Ne = np.zeros((len(self.timestamp),len(self._alts))) # Electrons
        self.Nm = np.zeros((len(self.timestamp),len(self._alts))) # Negetive ions
        self.Np = np.zeros((len(self.timestamp),len(self._alts))) # Positive ions
        self.Nxp = np.zeros((len(self.timestamp),len(self._alts))) # Positive cluster ions

        self.Ne[0,:] = self._o.iri["Ne"][0,:]
        self.Ne[0,:][self.Ne[0,:] < 0] = 0.
        self.Nm[0,:] = self.Ne[0,:] * self.lam
        self.Np[0,:] = self.Ne[0,:]
        self.Nxp[0,:] = self.Nm[0,:]
        return

    def _nh_(self, y):
        nans, x = np.isnan(y), lambda z: z.nonzero()[0]
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y
    
    def exe(self, verbose=True):
        if verbose: print("\n Start simulation ...")
        for I in self.timestamp[1:]:
            max_err = np.max(np.abs(self.Ne[I-1,60:150] + self.Nm[I-1,60:150] - self.Np[I-1,60:150] - self.Nxp[I-1,60:150]))
            if verbose: print("\tSolving for time - %d (sec) - "%(I-1)+str(max_err))
            I0 = I-1
            self.Ne[I,:] = self.Ne[I0,:] + (self.q(I0) + self.gm*self.Nm[I0,:] - self.b*self.Ne[I0,:] - self.ad*self.Ne[I0,:]*self.Np[I0,:] - 
                            self.adc*self.Ne[I0,:]*self.Nxp[I0,:])
            self.Nm[I,:] = self.Nm[I0,:] + (self.b*self.Ne[I0,:] - self.gm*self.Nm[I0,:] - self.ai*self.Nm[I0,:]*(self.Np[I0,:]+self.Nxp[I0,:]))
            self.Np[I,:] = self.Np[I0,:] + (self.q(I0) - self.B*self.Np[I0,:] - self.ad*self.Ne[I0,:]*self.Np[I0,:] - self.ai*self.Nm[I0,:]*self.Np[I0,:])
            self.Nxp[I,:] = self.Nxp[I0,:] + (self.B*self.Np[I0,:] - self.adc*self.Ne[I0,:]*self.Nxp[I0,:] - self.ai*self.Nm[I0,:]*self.Nxp[I0,:])
            pass
        return self
    
