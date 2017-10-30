''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data 
    structures, using the theory described in the two following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.

    This file contains the definition of the KPO class:
    --------------------------------------------------

    an object that contains Ker-phase information (kpi), data (kpd) 
    and relevant additional information extracted from the fits header
    (hdr)
    -------------------------------------------------------------------- '''

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    import pyfits as pf
except:
    import astropy.io.fits as pf

import copy
import pickle
import os
import sys
import pdb
import glob
import gzip

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

from scipy.interpolate import griddata

import core
from core import *

import kpi
from kpi import *

class KPO():
    ''' Class used to manipulate multiple Ker-phase datasets

        -------------------------------------------------------------------
        The class is designed to handle a single or multiple frames
        that can be combined for statistics purpose into a single data
        set.
        ------------------------------------------------------------------- '''

    def __init__(self, kp_fname):
        # Default instantiation.
        self.kpi = KPI(kp_fname)

        # if the file is a complete (kpi + kpd) structure
        # additional data can be loaded.
        try:
            myf = gzip.GzipFile(kp_fname, "r")
            data = pickle.load(myf)
            myf.close()

            self.kpd = data['kpd']
            self.kpe = data['kpe']
            self.hdr = data['hdr']
        except:
            print("File %s contains KPI information only" % (kp_fname,))

    # =========================================================================
    # =========================================================================
    def extract_UVP(self, image, m2pix): #, re_center=True, wfs=False, wrad=25):
        ''' -------------------------------------------------------------------
        extract the Fourier-phase from an image (a 2D array, not a file), and
        a m2pix scaling constant.
        
        relying on an exact DFT, computed only for the coordinates of the model.
        For now, assumes that the data is centered. Subtleties shall be added
        later.

        EXPERIMENTAL: work in progress !!!!
        ------------------------------------------------------------------- '''
        (XSZ, YSZ) = image.shape

        try:
            test = self.LL # to avoid recomputing a lot of re-usable arrays!
            
        except: # do the auxilliary computations
            self.bl_v = np.unique(self.kpi.uv[:,1])
            self.bl_u = np.unique(self.kpi.uv[:,0])

            self.vstep = self.bl_v[1] - self.bl_v[0]
            self.ustep = self.bl_u[1] - self.bl_u[0]

            self.LL   = core.compute_FTM(self.bl_v, m2pix, YSZ, 0)
            self.RR   = core.compute_FTM(self.bl_u, m2pix, XSZ, 1)

            self.uv_i = np.round(self.kpi.uv[:,0] / self.ustep, 1)
            self.uv_i -= self.uv_i.min()
            self.uv_i = self.uv_i.astype('int')
            self.uv_j = np.round(self.kpi.uv[:,1] / self.vstep, 1)
            self.uv_j -= self.uv_j.min()
            self.uv_j = self.uv_j.astype('int')

            
        myft = self.LL.dot(image).dot(self.RR) # this is the DFT
        myft_v = myft[self.uv_j, self.uv_i]
        return(myft_v)
    
    # =========================================================================
    # =========================================================================
    def extract_KPD(self, path, plotim=False, ave="none",
                    re_center=True, wfs=False, wrad=25):
        ''' extract kernel-phase data from one or more files (use regexp).

        If the path leads to a fits data cube, or to multiple single frame
        files, the extracted kernel-phases are consolidated into a
        unique kpd object.      
        
        Options:
        - ave:        average procedure ("none", "median", "mean")
        - re_center:  fine-centers the frame (default = True)
        - wfs:        wavefront sensing (keeps the phase)
        - wrad:       window radius in pixels (default=25)
        '''
        fnames = glob.glob(path)
        nf = fnames.__len__()

        fits_hdr = pf.getheader(fnames[0])
        
        print "%d files will be open" % (nf,)

        hdrs = [] # empty list of kp data headers

        # =========================
        if fits_hdr['NAXIS'] < 3:
            kpds = np.zeros((nf, self.kpi.nkphi)) # empty array of Ker-phases
            uvps = np.zeros((nf, self.kpi.nbuv))  # empty array of phases
            v2s  = np.zeros((nf, self.kpi.nbuv))  # empty array of square vis

            for i, fname in enumerate(fnames):
                (data, fhdr) = pf.getdata(fname, header=True)
                try:
                    dummy = (hdr['TELESCOP'])
                except:
                    fhdr = pf.getheader(fname)
                    print("HST file, double checking header")
                (hdr, sgnl, uvp, vis2) = extract_from_array(
                    data, fhdr, self.kpi, wrad,
                    save_im=False, re_center=re_center, plotim=plotim)
                uvps[i] = uvp
                kpds[i] = sgnl
                v2s[i]  = vis2
                hdrs.append(hdr)
            if nf == 1:
                hdrs = hdr
                kpds = sgnl
                uvps = uvp
                v2s  = vis2

        # =========================
        if fits_hdr['NAXIS'] == 3:
            nslc = fits_hdr['NAXIS3'] # number of "slices"
            kpds = np.zeros((nslc, self.kpi.nkphi))# empty array of Ker-phases
            uvps = np.zeros((nslc, self.kpi.nbuv)) # empty array of phases
            v2s  = np.zeros_like(uvps)             # empty array of square vis

            dcube = pf.getdata(fnames[0])

            for i in xrange(nslc):
                sys.stdout.write(
                    "\rextracting kp from img %3d/%3d" % (i+1,nslc))
                sys.stdout.flush()
                (hdr, sgnl, uvp, vis2) = extract_from_array(
                    dcube[i], fits_hdr, self.kpi, wrad,
                    save_im=False, re_center=re_center, plotim=plotim)
                uvps[i] = uvp
                kpds[i] = sgnl
                v2s[i]  = vis2
                hdrs.append(hdr)

        self.kpe = np.std(kpds, 0)

        if ave == "median":
            print "median average"
            self.kpd  = np.median(kpds, 0)
            self.hdr  = hdrs[0]
            self.v2   = np.median(v2s, 0)

        if ave == "mean":
            print "mean average"
            self.kpd = np.mean(kpds, 0)
            self.hdr = hdrs[0]
            self.v2  = np.mean(v2s, 0)

        if ave == "none":
            print "no average"
            self.kpd = kpds
            self.hdr = hdrs
            self.v2  = v2s

        if wfs:
            self.uvph = uvps # keeps the uv phase

    # =========================================================================
    # =========================================================================
    def copy(self):
        ''' Returns a deep copy of the Multi_KPD object.
        '''
        res = copy.deepcopy(self)
        return res

    # =========================================================================
    # =========================================================================
    def calibrate(self, calib, regul="None"):
        ''' Returns a new instance of Multi_KPD object.

        Kernel-phases are calibrated by the calibrator passed as parameter.
        Assumes for now that the original object and the calibrator are
        collapsed into one single kp data set. '''

        res = copy.deepcopy(self)

        if np.size(calib.kpd.shape) == 1:
            res.kpd -= calib.kpd
            return res
        else:
            coeffs = super_cal_coeffs(self, calib, regul)
            
        return res

    # =========================================================================
    # =========================================================================
    def average_KPD(self, algo="median"):
        ''' Averages the multiple KP data into a single series.

        Default is "median". Other option is "mean".
        '''
        
        temp = np.array(self.kpd)

        if algo == "median":
            aver = np.median(self.kpd, 0)
        else:
            aver = np.mean(self.kpd, 0)


        # ----------------------
        # update data structures
        # ----------------------
        self.kpd = aver

        # -------------------------------------
        # update data header (mean orientation)
        # -------------------------------------
        nh = self.hdr.__len__()
        ori = np.zeros(nh)

        for i in xrange(nh):
            ori[i] = self.hdr[i]['orient']

        self.hdr = self.hdr[0] # only keep one header
        self.hdr['orient'] = np.mean(ori)

        return self.kpd

    # =========================================================================
    # =========================================================================
    def save_fo_file(self, fname):
        '''Saves the kpi and kpd data structures in a pickle

        --------------------------------------------------------------
        The data can then later be reloaded for additional analysis,
        without having to go through the sometimes time-consuming
        extraction from the original fits files.

        To save on disk space, this procedure uses the gzip module.
        While there is no requirement for a specific extension for the
        file, I would recommend that one uses ".kpd.gz", so as to make
        it obvious that the file is a gzipped kpd data structure.
        --------------------------------------------------------------
        '''

        try:
            data = {'name'   : self.kpi.name,
                    'mask'   : self.kpi.mask,
                    'uv'     : self.kpi.uv,
                    'TFM'    : self.kpi.TFM,
                    'KerPhi' : self.kpi.KerPhi,
                    'RED'    : self.kpi.RED}
        except:
            print("KPI data structure is incomplete")
            print("File %s was not saved to disk" % (fname,))
            return(None)

        try:
            data['hdr'] = self.hdr
            data['kpd'] = self.kpd
            data['kpe'] = self.kpe
            data['v2']  = self.v2

        except:
            print("KPD data structure is incomplete")

        try:
            data['uvph'] = self.uvph
        except:
            print("KPD data structure does not contain uv-phase")
            
        try:
            print("Savinf file %s to disk" % (fname,))
            myf = gzip.GzipFile(fname, "wb")
        except:
            print("File %s cannot be created" % (fname,))
            print("Data was not saved to disk")
            return(None)

        pickle.dump(data, myf, -1)
        myf.close()
        print("File %s was successfully written to disk" % (fname,))
        return(0)

    # =========================================================================
    # =========================================================================
    def plot_uv_phase_map(self, data=None, reso=400):

        uv = self.kpi.uv
        
        Kinv = np.linalg.pinv(self.kpi.KerPhi)

        dxy = np.max(np.abs(uv))
        xi = np.linspace(-dxy, dxy, reso)
        yi = np.linspace(-dxy, dxy, reso)

        if data == None:
            data = np.dot(Kinv, self.kpd)
        z1 = griddata((np.array([uv[:,0], -uv[:,0]]).flatten(),
                       np.array([uv[:,1], -uv[:,1]]).flatten()),
                      np.array([data, -data]).flatten(),
                      (xi[None,:], yi[:,None]), method='linear')
        
        z2 = griddata((np.array([uv[:,0], -uv[:,0]]).flatten(), 
                       np.array([uv[:,1], -uv[:,1]]).flatten()), 
                      np.array([self.kpi.RED, self.kpi.RED]).flatten(),
                      (xi[None,:], yi[:,None]), method='linear')

        plt.imshow(z1)
        return (z1, z2)

    # =========================================================================
    # =========================================================================
    def plot_uv_map(self, data=None, sym=True, reso=400):
        ''' Interpolates a uv-information vector to turn it into a 2D map.

        Parameters:
        ----------

        - data: the 1D vector of size self.kpi.nbuv
        - sym: symmetric or anti-symmetric map?
        - reso: the resolution of the plot (how many pixels across)
        ------------------------------------------------------------------ '''

        uv = self.kpi.uv
        
        Kinv = np.linalg.pinv(self.kpi.KerPhi)

        dxy = np.max(np.abs(uv))
        xi = np.linspace(-dxy, dxy, reso)
        yi = np.linspace(-dxy, dxy, reso)

        if data is None:
            data = np.dot(Kinv, self.kpd)

        if sym is True:
            data2 = data.copy()
        else:
            data2 = -data
            
        z1 = griddata((np.array([uv[:,0], -uv[:,0]]).flatten(),
                       np.array([uv[:,1], -uv[:,1]]).flatten()),
                      np.array([data, data2]).flatten(),
                      (xi[None,:], yi[:,None]), method='linear')
        f1 = plt.figure(figsize=(8,8))
        ax1 = f1.add_subplot(111)
        ax1.imshow(z1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        f1.tight_layout()
        return (z1)
