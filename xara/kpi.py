''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data 
    structures, using the theory described in the two following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.
    ----

    This file contains the definition of the KPI class:
    --------------------------------------------------

    an object that contains the linear model for the optical system
      of interest. Properties of this model are:
      --> name   : name of the model (HST, Keck, Annulus_19, ...)
      --> mask   : array of coordinates for pupil sample points
      --> uv     : matching array of coordinates in uv plane (baselines)
      --> RED    : vector coding the redundancy of these baselines
      --> TFM    : transfer matrix, linking pupil-phase to uv-phase
      --> KerPhi : array storing the kernel-phase relations
      -------------------------------------------------------------------- '''

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import pickle
import os
import sys
import gzip
import pdb

import core
from core import *

class KPI(object):
    ''' Fundamental kernel-phase relations

    -----------------------------------------------------------------------
    This object condenses all the knowledge about a given instrument pupil 
    geometry into a series of arrays useful for kernel-phase analysis as 
    well as for other purposes, such as wavefront sensing.
    ----------------------------------------------------------------------- '''

    name = "" # default array name. Should be descriptive of the array geometry

    # =========================================================================
    # =========================================================================

    def __init__(self, file=None, Ns=2, satur=1):
        ''' Default instantiation of a KerPhase_Relation object:

        -------------------------------------------------------------------
        Default instantiation of this KerPhase_Relation class is achieved
        by loading a pre-made file, containing all the relevant information

        Option:
        - Ns: sampling of the data, which is by default assumed to be at
              least Nyquist (for which Ns=2). If not, then the model needs 
              to be modified, and some baselines discarded, to accomodate
              for the insufficient sampling.
        -------------------------------------------------------------------'''
        try:
            # -------------------------------
            # load the pickled data structure
            # -------------------------------
            #pdb.set_trace()
            myf = gzip.GzipFile(file, "r")
            data = pickle.load(myf)
            myf.close()

            # -------------------------------
            # restore the variables for this 
            # session of Ker-phase use!
            # -------------------------------
            try:    self.name = data['name']
            except: self.name = "UNKNOWN"

            self.uv     = data['uv']
            self.mask   = data['mask']
            self.RED    = data['RED']
            self.KerPhi = data['KerPhi']
            self.TFM    = data['TFM']
        
            self.nbh   = self.mask.shape[0]
            self.nbuv  = self.uv.shape[0]
            self.nkphi = self.KerPhi.shape[0]
        
        except: 
            print("File %s isn't a valid Ker-phase data structure" % (file))
            try: self.from_coord_file(file, "", Ns, satur)
            except:
                print("Failed.")
                return None

    # =========================================================================
    # =========================================================================

    def from_coord_file(self, file, array_name="", Ns=2, satur=1):
        ''' Creation of the KerPhase_Relation object from a pupil mask file:

        ----------------------------------------------------------------
        This is the core function of this class, really...

        Input is a pupil coordinates file, containing one set of (x,y) 
        coordinates per line. Coordinates are in meters. From this, all 
        the intermediate products that lead to the kernel-phase matrix 
        KerPhi are calculated.
        ---------------------------------------------------------------- '''
        self.mask = 1.0 * np.loadtxt(file) # sub-Ap. coordinate files 
        self.nbh  = self.mask.shape[0]   # number of sub-Ap

        ndgt = 4 # number of digits of precision for rounding
        prec = 10**(-ndgt)

        # ================================================
        # Determine all the baselines in the array.
        # ================================================

        # 1. Start by doing all the possible combinations of coordinates 
        # --------------------------------------------------------------
        # in the array to calculate the baselines. The intent here, is 
        # to work with redundant arrays of course, so there will be plenty 
        # of duplicates.

        nbh = self.nbh # local representation of the class variable
        uvx = np.zeros(nbh * (nbh-1)) # prepare empty arrays to store
        uvy = np.zeros(nbh * (nbh-1)) # the baselines

        k = 0 # index for possible combinations (k = f(i,j))
        
        uvi = np.zeros(nbh * (nbh-1), dtype=int) # arrays to store the possible
        uvj = np.zeros(nbh * (nbh-1), dtype=int) # combinations k=f(i,j) !!


        for i in range(nbh):     # do all the possible combinations of
            for j in range(nbh): # sub-apertures
                if i != j:
                    uvx[k] = self.mask[i,0] - self.mask[j,0]
                    uvy[k] = self.mask[i,1] - self.mask[j,1]
                    # ---
                    uvi[k], uvj[k] = i, j
                    k+=1

        a = np.unique(np.round(uvx, ndgt)) # distinct u-component of baselines
        nbx    = a.shape[0]                # number of distinct u-components
        uv_sel = np.zeros((0,2))           # array for "selected" baselines

        for i in range(nbx):     # identify distinct v-coords and fill uv_sel
            b = np.where(np.abs(uvx - a[i]) <= prec)
            c = np.unique(np.round(uvy[b], ndgt))

            nby = np.shape(c)[0] # number of distinct v-compoments
            for j in range(nby):
                uv_sel = np.append(uv_sel, [[a[i],c[j]]], axis=0)

        self.nbuv = np.shape(uv_sel)[0]/2 # actual number of distinct uv points
        self.uv   = uv_sel[:self.nbuv,:]  # discard second half (symmetric)
        print "%d distinct baselines were identified" % (self.nbuv,)

        # 1.5. Special case for undersampled data
        # ---------------------------------------
        if (Ns < 2):
            uv_sampl = self.uv.copy()   # copy previously identified baselines
            uvm = np.abs(self.uv).max() # max baseline length
            keep = (np.abs(uv_sampl[:,0]) < (uvm*Ns/2.)) * \
                (np.abs(uv_sampl[:,1]) < (uvm*Ns/2.))
            self.uv = uv_sampl[keep]
            self.nbuv = (self.uv.shape)[0]

            print "%d baselines were kept (undersampled data)" % (self.nbuv,)

        # 1.7. Special case for saturated data
        # -------------------------------------
        if (satur < 1):
            uv_sampl = self.uv.copy()   # copy previously identified baselines
            uvm = np.abs(self.uv).max() # max baseline length

            blength = np.sqrt(np.abs(uv_sampl[:,0])**2 + 
                              np.abs(uv_sampl[:,1])**2)
            
            #pdb.set_trace()
            bmax = blength.max()
            keep = (blength < satur * bmax)
            self.uv = uv_sampl[keep]
            self.nbuv = (self.uv.shape)[0]

            print "%d baselines were kept (saturated data)" % (self.nbuv,)


        # 2. Calculate the transfer matrix and the redundancy vector
        # --------------------------------------------------------------
        self.TFM = np.zeros((self.nbuv, self.nbh), dtype=float) # matrix
        self.RED = np.zeros(self.nbuv, dtype=float)             # Redundancy


        for i in range(self.nbuv):
            a=np.where((np.abs(self.uv[i,0]-uvx) <= prec) *
                       (np.abs(self.uv[i,1]-uvy) <= prec))
            self.TFM[i, uvi[a]] +=  1.0
            self.TFM[i, uvj[a]] += -1.0
            self.RED[i]         = np.size(a)

        # 3. Determine the kernel-phase relations
        # ----------------------------------------

        # One sub-aperture is taken as reference: the corresponding
        # column of the transfer matrix is discarded. TFM is now a
        # (nbuv) x (nbh - 1) array.
        
        # The choice is up to the user... but the simplest is to
        # discard the first column, that is, use the first aperture
        # as a reference?

        self.TFM = self.TFM[:,1:] # cf. explanation
        self.TFM = np.dot(np.diag(1./self.RED), self.TFM) # experiment
        U, S, Vh = np.linalg.svd(self.TFM.T, full_matrices=1)

        S1 = np.zeros(self.nbuv)
        S1[0:nbh-1] = S

        self.nkphi  = np.size(np.where(abs(S1) < 1e-3)) # number of Ker-phases
        KPhiCol     = np.where(abs(S1) < 1e-3)[0]
        self.KerPhi = np.zeros((self.nkphi, self.nbuv)) # allocate the array

        for i in range(self.nkphi):
            self.KerPhi[i,:] = (Vh)[KPhiCol[i],:]

        print '-------------------------------'
        print 'Singular values for this array:\n', np.round(S, ndgt)
        print '\nRedundancy Vector:\n', self.RED
        self.name = array_name


    # =========================================================================
    # =========================================================================

    def summary_properties(self):
        print("Summary of properties for array: %s" % (self.name,))
        print("-------------------------------")
        print("%3d sub-apertures\n%3d distinct baselines" % \
              (self.nbh, self.nbuv))
        print("%3d Ker-phases (%.2f %% target phase information recovery)" % \
              (self.nkphi, (100.0*self.nkphi)/self.nbuv,))
        print("%d Eig-phases" % (self.nbuv-self.nkphi))

    # =========================================================================
    # =========================================================================

    def plot_pupil_and_uv(self, xymax = 8.0, plot_redun = False):
        '''Nice plot of the pupil sampling and matching uv plane.

        --------------------------------------------------------------------
        Parameters:
        ----------

        - xymax: radius of the region represented in the baseline plot (meters)
        - plot_redun: flag to add the redundancy vector information (boolean)
        - -------------------------------------------------------------------

        '''

        f0 = plt.figure(1, figsize=(14,7))
        plt.clf()
        ax0 = plt.subplot(121)
        ax0.plot(self.mask[:,0], self.mask[:,1], 'bo')
        ax0.axis([-xymax, xymax, -xymax, xymax], aspect='equal')
        #ax0.title(self.name+' pupil')

        ax1 = plt.subplot(122)
        ax1.plot(self.uv[:,0],   self.uv[:,1], 'b.') # plot baselines + symetric
        ax1.plot(-self.uv[:,0], -self.uv[:,1], 'r.') # for a "complete" feel
        #ax1.title(self.name+' uv coverage')
        ax1.axis([-2*xymax, 2*xymax, -2*xymax, 2*xymax], aspect='equal')


        # complete previous plot with redundancy of the baseline
        # -------------------------------------------------------
        dy = 0.1*abs(self.uv[0,1]-self.uv[1,1]) # to offset text in the plot.
        if plot_redun:
            for i in range(self.nbuv):
                ax1.text(self.uv[i,0]+dy, self.uv[i,1]+dy, 
                         int(self.RED[i]), ha='center')
        
        ax0.axis('equal')
        ax1.axis('equal')
        plt.draw()
        #plt.draw()

    # =========================================================================
    # =========================================================================

    def save_to_file(self, file):
        ''' Export the KerPhase_Relation data structure into a pickle
        
        ----------------------------------------------------------------
        To save on disk space, this procedure uses the gzip module.
        While there is no requirement for a specific extension for the
        file, I would recommend that one uses ".kpi.gz", so as to make
        it obvious that the file is a gzipped kpi data structure.
        ----------------------------------------------------------------  '''
        try: 
            data = {'name'   : self.name,
                    'mask'   : self.mask,
                    'uv'     : self.uv,
                    'TFM'    : self.TFM,
                    'KerPhi' : self.KerPhi,
                    'RED'    : self.RED}
        except:
            print("KerPhase_Relation data structure is incomplete")
            print("File %s wasn't saved!" % (file,))
            return None
        # -------------
        try: myf = gzip.GzipFile(file, "wb")
        except:
            print("File %s cannot be created."+
                  " KerPhase_Relation data structure wasn't saved." % (file,))
            return None
        # -------------
        pickle.dump(data, myf, -1)
        myf.close()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
