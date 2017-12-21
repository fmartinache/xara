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
    import astropy.io.fits as fits
except:
    import pyfits as fits

import copy
import pickle
import os
import sys
import glob
import gzip

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

from scipy.interpolate import griddata

import core
import kpi

import pdb

class KPO():
    ''' Class used to manipulate multiple Ker-phase datasets

        -------------------------------------------------------------------
        The class is designed to handle a single or multiple frames
        that can be combined for statistics purpose into a single data
        set.
        ------------------------------------------------------------------- '''

    def __init__(self, fname=None, array=None, ndgt=5, bfilter=None, ID=""):
        ''' Default instantiation of a KerPhase_Relation object:

        -------------------------------------------------------------------
        See the documentation of KPI class instantiation function for a
        description of the available options.
        -------------------------------------------------------------------'''

        # Default instantiation.
        self.kpi = kpi.KPI(fname=fname, array=array,
                           ndgt=ndgt, bfilter=bfilter, ID=ID)

        self.TARGET = [] # source names
        self.CVIS   = [] # complex visibilities
        self.KPDT   = [] # kernel-phase data
        self.DETPA  = [] # detector position angles
        self.MJDATE = [] # data modified Julian date
        
        # if the file is a complete (kpi + kpd) structure
        # additional data can be loaded.
        try:
            hdul = fits.open(fname)
        except:
            print("File provided is not a fits file")
            return

        # how many data sets are included?
        # -------------------------------
        nbd = 0
        for ii in range(len(hdul)):
            try:
                test = hdul[ii].header['EXTNAME']
                if 'KP-DATA' in test:
                    nbd += 1
            except:
                pass
            
        # read the data
        # -------------
        print("The file contains %d data-sets" % (nbd,))
        
        for ii in range(nbd):
            self.KPDT.append(hdul['KP-DATA%d' % (ii+1,)].data)
            self.DETPA.append(hdul['KP-INFO%d' % (ii+1,)].data['DETPA'])
            self.MJDATE.append(hdul['KP-INFO%d' % (ii+1,)].data['MJD'])
            self.TARGET.append(hdul[0].header['TARGET%d' % (ii+1,)])
        self.CWAVEL = hdul[0].header['CWAVEL']
        # end
        # ---
        hdul.close()
        
    # =========================================================================
    # =========================================================================
    def copy(self):
        ''' -----------------------------------------------------------------
        Returns a deep copy of the Multi_KPD object.
        ----------------------------------------------------------------- '''
        res = copy.deepcopy(self)
        return res

    # =========================================================================
    # =========================================================================
    def extract_cvis_from_img(self, image, m2pix, method="LDFT1"):
        ''' -----------------------------------------------------------------
        Extracts the complex visibility vector of a 2D image for the KPI.

        Parameters:
        ----------
        - m2pix: meter to pixel scaling parameter

        - method: a string describing the extraction method. Default="LDFT1"
          + "LDFT1": one sided DFT (recommended: most flexible)
          + "LDFT2": two-sided DFT (FASTER, but for cartesian grids only)
          + "FFT"  : the old fashioned way - not as accurate !!
        ----------------------------------------------------------------- '''
        if   method == "LDFT2": res = self.__extract_cvis_ldft2(image, m2pix)
        elif method == "LDFT1": res = self.__extract_cvis_ldft1(image, m2pix)
        elif method == "FFT":   res = self.__extract_cvis_fft(image, m2pix)
        else:
            res = None
            print("Requested method %s does not exist" % (method,))
        return res
    
    # =========================================================================
    # =========================================================================
    def __extract_cvis_ldft2(self, image, m2pix):
        ''' -------------------------------------------------------------------
        extracts complex visibility from an image (using DFT = LL . image . RR)

        Assumes that the data is cleaned and centered.

        Parameters:
        ----------
        - image: a 2D image array
        - m2pix: a scaling constant

        Returns a vector of DFT computed only for the coordinates of the model.
        This method, based on a (LL . image . RR) computation is suited to 
        models based on a cartesian grid aligned with the image pixels. It 
        becomes less efficient when models do not follow such a grid (like for
        an hexagonal grid) or when the grid is tilted at a non trivial angle
        (to match specific pupil features).
        
        For an image of size (SZ x SZ), the computation requires two small 
        matrices of size (N_U x SZ) and (N_V x SZ).

        The alternative method is *extract_cvis_ldft1*
        ------------------------------------------------------------------- '''
        (XSZ, YSZ) = image.shape

        try:
            test = self.LL # check to avoid recomputing auxilliary arrays!
            
        except: # do the auxilliary computations
            self.bl_v = np.unique(self.kpi.UVC[:,1])
            self.bl_u = np.unique(self.kpi.UVC[:,0])

            self.vstep = self.bl_v[1] - self.bl_v[0]
            self.ustep = self.bl_u[1] - self.bl_u[0]

            self.LL   = core.compute_DFTM2(self.bl_v, m2pix, YSZ, 0)
            self.RR   = core.compute_DFTM2(self.bl_u, m2pix, XSZ, 1)

            self.uv_i = np.round(self.kpi.UVC[:,0] / self.ustep, 1)
            self.uv_i -= self.uv_i.min()
            self.uv_i = self.uv_i.astype('int')
            self.uv_j = np.round(self.kpi.UVC[:,1] / self.vstep, 1)
            self.uv_j -= self.uv_j.min()
            self.uv_j = self.uv_j.astype('int')
            
        myft = self.LL.dot(image).dot(self.RR) # this is the DFT
        myft_v = myft[self.uv_j, self.uv_i]
        return(myft_v)
    
    # =========================================================================
    # =========================================================================
    def __extract_cvis_ldft1(self, image, m2pix):
        ''' -------------------------------------------------------------------
        extracts complex visibility from an image (using DFT = FF . image)

        Assumes that the data is cleaned and centered.

        Parameters:
        ----------
        - image: a 2D image array
        - m2pix: a scaling constant

        Returns a 1D vector representation of the image DFT computed only for
        the uv coordinates of the model associated to this KPO object.

        This method is based on a single dot product between a matrix FF and a
        flattened representation of the image.

        For an image of size (SZ x SZ), the computation requires what can be a
        fairly large (N_UV x SZ^2) auxilliary matrix.

        The alternative method is *extract_cvis_ldft2*
        ------------------------------------------------------------------- '''
        ISZ = image.shape[0]
        try:
            test = self.FF # check to avoid recomputing auxilliary arrays!

        except:
            self.FF = core.compute_DFTM1(self.kpi.UVC, m2pix, ISZ)

        myft_v = self.FF.dot(image.flatten())
        return(myft_v)
    
    # =========================================================================
    # =========================================================================
    def __extract_cvis_fft(self, image, m2pix):
        ''' -------------------------------------------------------------------
        extracts complex visibility from a square image (using old school FFT)

        Assumes that you know what you do and have provided sufficient 
        zero-padding to avoid Fourier aliasing for the spatial frequencies the 
        model tries to get access to.

        The alternative methods use LDFT (local discrete Fourier transform)
        ------------------------------------------------------------------- '''
        ISZ, DZ = image.shape[0], image.shape[0]/2
        uv_samp = self.kpi.UVC * m2pix + DZ # uv sample coordinates in F pixels

        # calculate and normalize the Fourier transform
        ac = shift(fft(shift(image)))
        ac /= (np.abs(ac)).max() / kpi.nbap

        xx = np.cast['int'](np.round(uv_samp[:,1]))
        yy = np.cast['int'](np.round(uv_samp[:,0]))
        myft_v = ac[xx, yy]
        return(myft_v)
    
    # =========================================================================
    # =========================================================================
    def extract_KPD(self, path, target=None,
                    recenter=True, wrad=None, method="LDFT1"):
        ''' extract kernel-phase data from one or more files (use regexp).

        ---------------------------------------------------------------------
        If the path leads to a fits data cube, or to multiple single frame
        files, the extracted kernel-phases are consolidated into a unique
        KPD array.
        
        Parameters:
        ----------
        - path     : path to one or more data fits files
        - target   : a 8 character string ID (default: get from fits file)

        - recenter : fine-centers the frame (default = True)
        - wrad     : window radius in pixels (default is None)
        - method   : string describing the extraction method. Default="LDFT1"
          + "LDFT1" : one sided DFT (recommended: most flexible)
          + "LDFT2" : two-sided DFT (FASTER, but for cartesian grids only)
          + "FFT"   : the old fashioned way - not as accurate !!
        ---------------------------------------------------------------------
        '''
        fnames   = sorted(glob.glob(path))
        hdul     = fits.open(fnames[0])
        tel_name = hdul[0].header['TELESCOP']
        # ------------------------------------------------------------
        if 'Keck II' in tel_name:
            print("The data comes from Keck")
            
            tgt, wl, cvis, kpd, detpa, mjdate = self.__extract_KPD_Keck(
                fnames, target=target, recenter=recenter, wrad=wrad,
                method=method)
        # ------------------------------------------------------------
        elif 'HST' in tel_name:
            print("The data comes from HST")

            tgt, wl, cvis, kpd, detpa, mjdate = self.__extract_KPD_HST(
                fnames, target=target, recenter=recenter, wrad=wrad,
                method=method)
        # ------------------------------------------------------------
        elif 'VLT' in tel_name:
            if 'SPHERE' in hdul[0].header['INSTRUME']:
                print("The data comes from VLT - SPHERE")
                
        # ------------------------------------------------------------
        else:
            print("Extraction for %s not implemented." % (tel_name,))
            return
        # ------------------------------------------------------------
        hdul.close()
        
        self.CWAVEL = wl
        self.TARGET.append(tgt)
        self.CVIS.append(np.array(cvis))
        self.KPDT.append(np.array(kpd))
        self.DETPA.append(np.array(detpa).flatten())
        self.MJDATE.append(np.array(mjdate))
        return
    
    # =========================================================================
    # =========================================================================
    def __extract_KPD_HST(self, fnames, target=None,
                          recenter=True, wrad=None, method="LDFT1"):
        
        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))

        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date

        hdul   = fits.open(fnames[0])
        xsz    = hdul['SCI'].header['NAXIS1']       # image x-size
        ysz    = hdul['SCI'].header['NAXIS2']       # image y-size
        pscale = 43.1                               # plate scale (mas)
        cwavel = hdul[0].header['PHOTPLAM']*1e-10   # central wavelength
        imsize = 128                                # chop image
        m2pix  = core.mas2rad(pscale)*imsize/cwavel # Fourier scaling
            
        if target is None:
            target = hdul[0].header['TARGNAME']    # Target name

        hdul.close()

        for ii in range(nf):
            hdul = fits.open(fnames[ii])

        nslice = 1
        if hdul['SCI'].header['NAXIS'] == 3:
            nslice = hdul['SCI'].header['NAXIS3']

        data = hdul['SCI'].data.reshape((nslice, ysz, xsz))
                
        # ---- extract the Fourier data ----
        for jj in range(nslice):
            img = core.recenter(data[jj], sg_rad=50, verbose=False)
            img = img[192:320,192:320] # from 512x512 -> 128x128
            temp = self.extract_cvis_from_img(img, m2pix, method)
            cvis.append(temp)
            kpdata.append(self.kpi.KPM.dot(np.angle(temp)))
            print("File %s, slice %2d" % (fnames[ii], jj+1))
            
            mjdate.append(hdul['SCI'].header['ROUTTIME'])
            detpa.append(hdul[0].header['ORIENTAT'])
        hdul.close()

        return target, cwavel, cvis, kpdata, detpa, mjdate

    # =========================================================================
    # =========================================================================
    def __extract_KPD_VLT_SPHERE(self, fnames, target=None,
                                 recenter=False, wrad=None, method="LDFT1"):
        
        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))
        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date
        wavel  = 0.0
        
        if target is None:
            target = hdul[0].header['OBJECT']      # Target name

        return target, cwavel, cvis, kpdata, detpa, mjdate

    # =========================================================================
    # =========================================================================
    def __extract_KPD_Keck(self, fnames, target=None,
                           recenter=False, wrad=None, method="LDFT1"):
        
        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))
        
        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date

        hdul   = fits.open(fnames[0])
        xsz    = hdul[0].header['NAXIS1']           # image x-size
        ysz    = hdul[0].header['NAXIS2']           # image y-size
        pscale = 10.0                               # plate scale (mas)
        cwavel = hdul[0].header['CENWAVE'] * 1e-6   # central wavelength
        imsize = hdul[0].header['NAXIS1']           # image size (pixls)
        m2pix  = core.mas2rad(pscale)*imsize/cwavel # Fourier scaling
        if target is None:
            target = hdul[0].header['OBJECT']      # Target name
        hdul.close()

        index = 0
        for ii in range(nf):
            hdul = fits.open(fnames[ii])

            nslice = 1
            if hdul[0].header['NAXIS'] == 3:
                nslice = hdul[0].header['NAXIS3']
            data = hdul[0].data.reshape((nslice, ysz, xsz))

            # ---- extract the Fourier data ----
            for jj in range(nslice):
                if recenter:
                    img = core.recenter(data[jj], sg_rad=50, verbose=False)
                else:
                    img = data[jj]

                index += 1
                temp = self.extract_cvis_from_img(img, m2pix, method)
                cvis.append(temp)
                kpdata.append(self.kpi.KPM.dot(np.angle(temp)))
                print("File %s, slice %2d" % (fnames[ii], jj+1))
                
                mjdate.append(hdul[0].header['MJD-OBS'])
            # --- detector position angle read globally ---
            detpa.append(hdul[1].data['pa'])

            hdul.close()
                
        return target, cwavel, cvis, kpdata, detpa, mjdate
    
    # =========================================================================
    # =========================================================================
    def save_as_fits(self, fname):
        ''' ------------------------------------------------------------------
        Exports the KPO data structure (KPI + KPDT) as a multi-extension FITS
        file, and writes it to disk.

        Parameters:
        ----------
        - fname: a file name (required)
        ------------------------------------------------------------------ '''
        self.hdul = self.kpi.package_as_fits() # KPI data

        # KPI information only?
        # ---------------------
        try:
            test = self.TARGET[0]
        except:
            print("No kernel-phase data was included")
            self.hdul.writeto(fname, overwrite=True)
            return

        self.hdul[0].header['CWAVEL'] = (self.CWAVEL,
                                         "Central wavelength (in meters)")
        # KPD information available?
        # --------------------------
        for ii in range(len(self.TARGET)): # loop over the different datasets

            # KP-DATA HDU
            # -----------
            kpd_hdu = fits.ImageHDU(self.KPDT[ii].astype(np.float64))
            kpd_hdu.header.add_comment("Kernel-phase data")
            kpd_hdu.header['EXTNAME'] = 'KP-DATA%d' % (ii+1,)
            kpd_hdu.header['TARGET'] = self.TARGET[ii]
            self.hdul.append(kpd_hdu)

            self.hdul[0].header['TARGET%d' % (ii+1,)] = (self.TARGET[ii],
                                                         "Target name from fits")

            # KP-INFO HDU
            # -----------
            detpa  = fits.Column(name="DETPA", format='D', array=self.DETPA[ii])
            mjdate = fits.Column(name="MJD",   format='D', array=self.MJDATE[ii])
            kpi_hdu = fits.BinTableHDU.from_columns([mjdate, detpa])
            kpi_hdu.header['TTYPE1'] = ('MJD', 'Obs. Modified Julian Date')
            kpi_hdu.header['TTYPE2'] = ('DETPA', 'Detector P.A. (degrees)')
            kpi_hdu.header['EXTNAME'] = 'KP-INFO%d' % (ii+1,)
            self.hdul.append(kpi_hdu)

        self.hdul.writeto(fname, overwrite=True)

    # =========================================================================
    # =========================================================================
    def kpd_binary_model(self, params, index=0, obs="KERNEL"):
        ''' Produces an observable model of a binary target matching the data
        Takes into account the possibly variable detector position angle.
        ----------------------------------------------------------------

        Parameters:
        ----------
        - params: 3-component vector (+2 optional), the binary "parameters":
          - params[0] = sep (mas)
          - params[1] = PA (deg) E of N.
          - params[2] = contrast ratio (primary/secondary)
        
          optional:
          - params[3] = angular size of primary (mas)
          - params[4] = angular size of secondary (mas)

        - index: the index of the dataset in the current data structure
                 default value is 0 (first dataset available)

        - obs: a string that precises the type of model produced:
          - "KERNEL": Kernel-phase data (default)
          - "PHASE": Fourier-phase
          - "AMPLI": Fourier-amplitude
          - "REAL": the real part of the Fourier-transform
          - "IMAG": the imaginary part of the Fourier-transform
        ---------------------------------------------------------------- '''

        try:
            test = self.TARGET[index]
        except:
            print("Requested dataset (index=%d) does not exist" % (index,))
            print("No data-matching binary model can be built.")
            print("For generic binary model, use xara.core.cvis_binary()")
            return

        nbd = self.KPDT[index].shape[0]

        sim = []

        # compute binary complex visibilities at multiple DETPA
        for ii in range(nbd):
            temp = self.__cvis_binary_model(params, self.DETPA[index][ii])
            if "KERNEL" in obs.upper():
                sim.append(self.kpi.KPM.dot(np.angle(temp)))
            elif "PHASE" in obs.upper():
                sim.append(np.angle(temp))
            elif "AMPLI" in obs.upper():
                sim.append(np.abs(temp))
            elif "REAL" in obs.upper():
                sim.append(temp.real)
            elif "IMAG" in obs.upper():
                sim.append(temp.imag)
            else:
                sim.append(temp)
        return np.array(sim)
        
    # =========================================================================
    # =========================================================================
    def __cvis_binary_model(self, params, detpa):
        ''' ------------------------------------------------------------------
        Private call to xara.core.cvis_binary(), using KPO object properties.
        ------------------------------------------------------------------ '''
        u = self.kpi.UVC[:,0]
        v = self.kpi.UVC[:,1]
        wl = self.CWAVEL
        return(core.cvis_binary(u,v,wl, params, detpa))

    # =========================================================================
    # =========================================================================
    def plot_uv_map(self, data=None, sym=True, reso=400, fsize=8):
        ''' ------------------------------------------------------------------
        Interpolates a uv-information vector to turn it into a 2D map.

        Parameters:
        ----------

        - data: the 1D vector of size self.kpi.nbuv
        - sym: symmetric or anti-symmetric map?
        - reso: the resolution of the plot (how many pixels across)
        - fsize: the size of the figure (in inches)
        ------------------------------------------------------------------ '''

        uv = self.kpi.UVC
        
        Kinv = np.linalg.pinv(self.kpi.KPM)

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
        f1 = plt.figure(figsize=(fsize, fsize))
        ax1 = f1.add_subplot(111)
        ax1.imshow(z1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        f1.tight_layout()
        return (z1)

    # =========================================================================
    # =========================================================================
    def plot_KPD(self, ii=0):
        ''' 1D Kernel-phase plot from one of the available datasets

        Parameters:
        ----------
        - ii: the index of the data set to plot (default = 0)
        ----------------------------------------------------------------- '''
        
        plt.errorbar(np.arange(self.kpi.nbkp),
                     np.median(self.KPDT[ii], axis=0),
                     yerr=np.std(self.KPDT[ii], axis=0),
                     label="%s" % (self.TARGET[ii]))
        plt.xlabel("Kernel-phase index")
        plt.ylabel("Kernel-phase (in radians)")
        plt.legend()

        # ideally, I'd like to have a little plot of the pupil+uv model
        return
