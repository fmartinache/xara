#!/usr/bin/env python

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import leastsq
from scipy.sparse import diags
from scipy.interpolate import griddata

import astropy.io.fits as fits
from astropy.time import Time
import copy

from . import core
from . import kpi

shift = np.fft.fftshift
fft = np.fft.fft2
ifft = np.fft.ifft2

i2pi = 1j*2*np.pi


class KPO():
    ''' Class used to manipulate multiple Ker-phase datasets

        -------------------------------------------------------------------
        The class is designed to handle a single or multiple frames
        that can be combined for statistics purpose into a single data
        set.
        ------------------------------------------------------------------- '''

    def __init__(self, fname=None, array=None, ndgt=5, bmax=None, ID=""):
        ''' Default instantiation of a KerPhase_Relation object:

        -------------------------------------------------------------------
        See the documentation of KPI class instantiation function for a
        description of the available options.
        -------------------------------------------------------------------'''

        # Default instantiation.
        self.kpi = kpi.KPI(fname=fname, array=array,
                           ndgt=ndgt, bmax=bmax, ID=ID)

        self.CWAVEL = []  # image/cube central wavelength
        self.PSCALE = []  # image/cube plate scale
        self.WRAD = []    # data apodization function radius
        self.WTYPE = []   # data apodization function type
        self.DETPA = []   # detector position angles
        self.MJDATE = []  # data modified Julian date
        self.TARGET = []  # source names
        self.CVIS = []    # complex visibilities
        self.KPDT = []    # kernel-phase data
        self.M2PIX = -1   # used to save time in later computations

        self._between_pix = False  # assumption for data centering

        self._in_cube = False     # to order the data structure
        self._cube_index = 0      # to order the data structure

        # if the file is a complete (kpi + kpd) structure
        # additional data can be loaded.
        if fname is None:
            print("No KPO data included")
            return

        try:
            hdul = fits.open(fname)
        except OSError:
            print("File provided is not a fits file")
            print("No KPO data included")
            return

        # how many data sets are included?
        # -------------------------------
        nbd = 0
        for ii in range(1, len(hdul)):
            try:
                test = hdul[ii].header['EXTNAME']
                if 'KP-DATA' in test:
                    nbd += 1
            except KeyError:
                pass

        # read the data
        # -------------
        print("The file contains %d data-sets" % (nbd,))

        for ii in range(nbd):
            self.KPDT.append(hdul['KP-DATA%d' % (ii+1,)].data)
            self.DETPA.append(hdul['KP-INFO%d' % (ii+1,)].data['DETPA'])
            self.MJDATE.append(hdul['KP-INFO%d' % (ii+1,)].data['MJD'])
            self.TARGET.append(hdul[0].header['TARGET%d' % (ii+1,)])
        try:
            self.CWAVEL = hdul[0].header['CWAVEL']
        except KeyError:
            print("CWAVEL was not set")

        # covariance?
        # -----------
        try:
            test = hdul['KP_COV']
            self.kp_cov = test.data
            print("Covariance data available and loaded")
        except KeyError:
            print("No covariance data available")
        # end
        # ---
        hdul.close()

    # =========================================================================
    def __del__(self):
        print("%s deleted" % (repr(self),))

    # =========================================================================
    def __str__(self):
        msg = "%s KPO data structure\n" % (repr(self),)
        msg += self.kpi.__str__()

        nset = len(self.KPDT)
        msg += "\n%d datasets present\n" % (nset,)

        for ii in range(nset):
            msg += "-" * 40 + "\n"
            msg += "DATA-%02d:\n" % (ii,)
            msg += "-" * 7 + "\n"
            msg += "-> %d frames\n" % (self.KPDT[ii].shape[0])
            msg += "-> TARGET = %s\n" % (self.TARGET[ii],)
            msg += "-> CWAVEL = %.2f microns\n" % (self.CWAVEL[ii] * 1e6,)
            msg += "-> PSCALE = %.2f mas/pixel\n" % (self.PSCALE[ii],)
            msg += "-> DETPA = %.2f (degrees)\n" % (self.DETPA[ii])
            if self.MJDATE[ii][0] != 0.0:
                myd = Time(val=self.MJDATE[ii][0], format="mjd")
                msg += "-> MJDATE = %s\n" % myd.to_value("iso")

        msg += "-" * 40 + "\n"
        return msg

    # =========================================================================
    def copy(self):
        ''' -----------------------------------------------------------------
        Returns a deep copy of the Multi_KPD object.
        ----------------------------------------------------------------- '''
        res = copy.deepcopy(self)
        return res

    # =========================================================================
    def KP_filter_img(self, image, pscale, cwavel):
        ''' -----------------------------------------------------------------
        !!EXPERIMENTAL!!

        Kernel-phase filtering of an image.

        One image is Fourier-transformed, its phase filtered by kernel and
        inverse Fourier-transformed, so that an "image" is returned.

        The function currently returns a complex array. First few tryouts
        suggest one should look at the real part of that "image".


        Parameters:
        ----------
        - image: 2D image to be cleaned
        - pscale: the plate scale of that image        (in mas/pixel)
        - cwavel: the central wavelength of that image (in meters)

        ----------------------------------------------------------------- '''

        ISZ = image.shape[0]
        m2pix = core.mas2rad(pscale) * ISZ / cwavel

        print(m2pix)

        try:
            _ = self.kpi.iKPM
        except AttributeError:
            self.kpi.iKPM = np.linalg.pinv(self.kpi.KPM)
            self.kpi.KPFILT = self.kpi.iKPM.dot(self.kpi.KPM)

        cvis = self.extract_cvis_from_img(image, m2pix)
        kkphi = self.kpi.KPFILT.dot(np.angle(cvis))
        cvis2 = np.abs(cvis)*np.exp(1j*kkphi)

        try:
            _ = self.iFF
        except AttributeError:
            self.iFF = core.compute_DFTM1(self.kpi.UVC, m2pix, ISZ, True)
        img1 = (self.iFF.dot(cvis2)).reshape(ISZ, ISZ)
        return(img1)

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
        if method == "LDFT2":
            res = self.__extract_cvis_ldft2(image, m2pix)
        elif method == "LDFT1":
            res = self.__extract_cvis_ldft1(image, m2pix)
        elif method == "FFT":
            res = self.__extract_cvis_fft(image, m2pix)
        else:
            res = None
            print("Requested method %s does not exist" % (method,))
        self.M2PIX = m2pix  # to check the validity of aux data next time !
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
        (YSZ, XSZ) = image.shape

        if m2pix != self.M2PIX:
            print("\nFirst time for m2pix = %.2f: " % (m2pix,))
            print("LDFT2: Computing new auxilliary data...")

            self.bl_v = np.unique(self.kpi.UVC[:, 1])
            self.bl_u = np.unique(self.kpi.UVC[:, 0])

            self.vstep = self.bl_v[1] - self.bl_v[0]
            self.ustep = self.bl_u[1] - self.bl_u[0]

            self.LL = core.compute_DFTM2(self.bl_v, m2pix, YSZ, 0)
            self.RR = core.compute_DFTM2(self.bl_u, m2pix, XSZ, 1)

            self.uv_i = np.round(self.kpi.UVC[:, 0] / self.ustep, 1)
            self.uv_i -= self.uv_i.min()
            self.uv_i = self.uv_i.astype('int')
            self.uv_j = np.round(self.kpi.UVC[:, 1] / self.vstep, 1)
            self.uv_j -= self.uv_j.min()
            self.uv_j = self.uv_j.astype('int')
            print("Done!")

        myft = self.LL.dot(image).dot(self.RR)  # this is the DFT
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

        if m2pix != self.M2PIX:
            print("\nFirst time for m2pix = %.2f: " % (m2pix,))
            print("LDFT1: Computing new Fourier matrix...")
            self.FF = core.compute_DFTM1(self.kpi.UVC, m2pix, ISZ)
            print("Done!")

        myft_v = self.FF.dot(image.flatten())
        myft_v *= self.kpi.TRM.sum() / image.sum()
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
        are the recommended way of computing the Fourier Transform.
        ------------------------------------------------------------------- '''
        DZ = image.shape[0]//2
        uv_samp = self.kpi.UVC * m2pix + DZ  # uv sample coords in F pixels

        # calculate and normalize the Fourier transform
        ac = shift(fft(shift(image)))
        ac /= (np.abs(ac)).max() / self.kpi.nbap

        xx = np.cast['int'](np.round(uv_samp[:, 1]))
        yy = np.cast['int'](np.round(uv_samp[:, 0]))
        myft_v = ac[xx, yy]
        return(myft_v)

    # =========================================================================
    # =========================================================================
    def create_UVP_cov_matrix(self, var_img, option="ABS", m2pix=None):
        ''' -------------------------------------------------------------------
        generate the covariance matrix for the UV phase.

        For independent image noises, the covariance matrix of the imaginary
        part of the Fourier transform can be computed explicitly. To go from
        that to the covariance of the phase is possible in the high-Strehl
        regime where Arctan(imag/real) ~ imag/real.

        Possibilities for the real part:
        - model redundancy -> option="RED"
        - "real part"      -> option="REAL"
        - "modulus"        -> option="ABS"

        Parameters:
        ----------
        - var_img: a 2D image with variance per pixel

        Further remark:
        ------
        A more sophisticated implementation could include the computation
        of cross-terms between imaginary and real parts to be more exact?

        Note: Covariance matrix can also be computed via MC simulations, if
        you are unhappy with this one.

        Note: This algorithm was developed in part by Romain Laugier
        ------------------------------------------------------------------- '''

        ISZ = var_img.shape[0]
        try:
            _ = self.FF  # check to avoid recomputing existing arrays!

        except AttributeError:
            if m2pix is not None:
                self.FF = core.compute_DFTM1(self.kpi.UVC, m2pix, ISZ)
            else:
                print("Fourier matrix and/or m2pix are not available.")
                print("Please compute Fourier matrix.")
                return

        cov_img = diags(var_img.flat)  # image covariance matrix

        if option == "RED":
            BB = self.FF.imag / self.kpi.RED[:, None]
            BB *= self.kpi.TRM.sum() / var_img.sum()
            print("Covariance Matrix computed using model redundancy!")

        if option == "REAL":
            ft = self.FF.dot(var_img.flat)
            BB = self.FF.imag / ft.real[:, None]
            print("Covariance Matrix computed using the real part of FT!")

        if option == "ABS":
            ft = self.FF.dot(var_img.flat)
            BB = self.FF.imag / np.abs(ft)[:, None]
            print("Covariance Matrix computed using the modulus of FT!")

        # fourier phase covariance added to the KPO data structure
        self.phi_cov = BB.dot(cov_img.dot(BB.T))
        # kernel phase covariance added to the KPO data structure
        self.kp_cov = self.kpi.KPM.dot(self.phi_cov.dot(self.kpi.KPM.T))
        return self.phi_cov

    # =========================================================================
    # =========================================================================
    def extract_KPD_single_cube(self, cube, pscale, cwavel,
                                detpa=0.0, mjdate=0.0, target="NO_NAME",
                                recenter=False, wrad=None, wtype="sgauss",
                                method="LDFT1"):
        """ ----------------------------------------------------------------
        Handles the kernel processing of a cube of square frames

        Parameters:
        ----------
        - cube  : the 2D array image to process
        - pscale: the image plate scale (in mas/pixels)
        - cwavel: the central wavelength (in meters)
        ---------------------------------------------------------------- """
        nfrm = cube.shape[0]  # number of frames in the cube

        self._in_cube = True  # signaling we are processing a cube!
        _cvis = []  # temporary storage
        _kpdt = []  # temporary storage

        for ii, img in enumerate(cube):
            print(f"\rCube slice {ii+1:3d} / {nfrm:3d}", end="", flush=True)

            cwl = cwavel[ii] if type(cwavel) is np.ndarray else cwavel
            dpa = detpa[ii] if type(detpa) is np.ndarray else detpa
            mjd = mjdate[ii] if type(mjdate) is np.ndarray else mjdate
            rad = wrad[ii] if type(wrad) is np.ndarray else wrad

            cvis = self.extract_KPD_single_frame(
                img, pscale, cwl,
                detpa=dpa, mjdate=mjd, target=target,
                recenter=recenter, wrad=rad, wtype=wtype, method=method)

            _cvis.append(cvis)
            _kpdt.append(self.kpi.KPM.dot(np.angle(cvis)))
        print()

        self._in_cube = False

        _cvis = np.array(_cvis)
        _kpdt = np.array(_kpdt)

        self.CVIS.append(_cvis)
        self.KPDT.append(_kpdt)

        self.CWAVEL.append(cwavel)
        self.PSCALE.append(pscale)
        self.WRAD.append(wrad)
        self.WTYPE.append(wtype)
        self.DETPA.append(detpa)
        self.TARGET.append(target)
        self.MJDATE.append(mjd)
        print()
        return

    # =========================================================================
    # =========================================================================
    def extract_KPD_single_frame(self, frame, pscale, cwavel,
                                 detpa=0.0, mjdate=0.0, target="NO_NAME",
                                 recenter=False, wrad=None, wtype="sgauss",
                                 method="LDFT1"):
        """Handles the kernel processing of a single square frame

        Parameters
        ----------
        - frame    : the 2D array image to process
        - pscale   : the image plate scale (in mas/pixels)
        - cwavel   : the central wavelength (in meters)
        - detpa    : detector position angle (in degrees)
        - mjdate   : mean julian day of observation (float)
        - target   : the target name (string)
        - recenter : data needs to be recentered (boolean)
        - wrad     : window radius (float - number of pixels)
        - wtype    : type of window ("sgauss" or "tophat")
        - method   : type of Fourier transform ("LDFT1", "LDFT2" or "FFT")

        """

        ysz, xsz = frame.shape                       # image size
        m2pix = core.mas2rad(pscale) * xsz / cwavel  # Fourier scaling

        # TODO: Flag when wrad is not None and wtype is None
        # to prevent users from trying to disable with wtype
        # and accidentally get "wmask".
        # Or change behaviour so that wtype=None is None
        if wrad is not None:
            if "hat" in wtype.lower():
                self.wmask = core.uniform_disk(
                    ysz, xsz, wrad, between_pix=self._between_pix)
            else:  # default super-gaussian window assumption
                self.wmask = core.super_gauss(
                    ysz, xsz, wrad, between_pix=self._between_pix)
        else:
            self.wmask = None

        self._tmp_img = frame.copy()
        if recenter is True:
            (x0, y0) = core.determine_origin(self._tmp_img, # mask=self.sgmask,
                                             algo="BCEN", verbose=False)
            dy, dx = (y0-ysz/2), (x0-xsz/2)

        if self.wmask is not None:  # apply window mask before extraction
            if recenter is True:
                self.wmask = np.roll(
                    self.wmask, np.round((dy, dx)).astype(int), axis=(0, 1))
            self._tmp_img *= self.wmask

        # ----- complex visibility extraction -----
        cvis = self.extract_cvis_from_img(self._tmp_img, m2pix, method)

        # ---- sub-pixel recentering correction -----
        if recenter is True:
            uvc = self.kpi.UVC * self.M2PIX
            corr = np.exp(i2pi * uvc.dot(np.array([dx, dy])/float(ysz)))
            cvis *= corr

        if self._in_cube:  # analysis of a data cube returns data
            return cvis

        else:  # analysis of an isolated frame => feed data structure
            self.CWAVEL.append(cwavel)
            self.PSCALE.append(pscale)
            self.WRAD.append(wrad)
            self.WTYPE.append(wtype)
            self.DETPA.append(detpa)
            self.MJDATE.append(mjdate)
            self.TARGET.append(target)
            self.CVIS.append(cvis)
            self.KPDT.append(self.kpi.KPM.dot(np.angle(cvis)))
            return

    # =========================================================================
    # =========================================================================
    def update_cov_matrix(self, kp_cov=None, phi_cov=None):
        '''------------------------------------------------------------------
        Updates the covariance matrices of the KPO data-structure.

        Parameters: (optional)
        ----------
        - kp_cov : a kernel-phase covariance matrix (ndarray)
        - phi_cov: a Fourier-phase covariance matrix (ndarray)

        Remarks:
        -------

        By default: computes an experimental covariance matrix using the
        first available dataset in the kpo datastructure for the Fourier
        phase and the kernel-phase. If there is not enough data to compute
        a sensible covariance matrix, one has to resort to MC simulations.
        The result of such computations can be passed as optional arguments.

        See create_UVP_cov_matrix() for a way to compute the covariance
        matrix based on an analytical model.
        ------------------------------------------------------------------ '''
        if phi_cov is not None:
            self.phi_cov = kp_cov
        else:
            try:
                _ = self.CVIS[0]
            except IndexError:
                print("Extract data before computing a covariance")
                return
            self.phi_cov = np.cov(np.angle(self.CVIS[0]).T)

        if kp_cov is not None:
            self.kp_cov = kp_cov
        else:
            try:
                _ = self.KPDT[0]
            except IndexError:
                print("Extract data before computing a covariance")
                return
            self.kp_cov = np.cov(self.KPDT[0].T)

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
        self.hdul = self.kpi.package_as_fits()  # KPI data

        # KPI information only?
        # ---------------------
        try:
            _ = self.TARGET[0]
        except IndexError:
            print("No kernel-phase data was included")
            self.hdul.writeto(fname, overwrite=True)
            return

        self.hdul[0].header['CWAVEL'] = (self.CWAVEL,
                                         "Central wavelength (in meters)")
        # KPD information available?
        # --------------------------
        for ii in range(len(self.TARGET)):  # loop over the different datasets

            # KP-DATA HDU
            # -----------
            kpd_hdu = fits.ImageHDU(self.KPDT[ii].astype(np.float64))
            kpd_hdu.header.add_comment("Kernel-phase data")
            kpd_hdu.header['EXTNAME'] = 'KP-DATA%d' % (ii+1,)
            kpd_hdu.header['TARGET'] = self.TARGET[ii]
            self.hdul.append(kpd_hdu)

            self.hdul[0].header['TARGET%d' % (ii+1,)] = (
                self.TARGET[ii], "Target name from fits")

            # KP-INFO HDU
            # -----------
            detpa = fits.Column(
                name="DETPA", format='D', array=self.DETPA[ii])
            mjdate = fits.Column(
                name="MJD",   format='D', array=self.MJDATE[ii])
            kpi_hdu = fits.BinTableHDU.from_columns([mjdate, detpa])
            kpi_hdu.header['TTYPE1'] = ('MJD', 'Obs. Modified Julian Date')
            kpi_hdu.header['TTYPE2'] = ('DETPA', 'Detector P.A. (degrees)')
            kpi_hdu.header['EXTNAME'] = 'KP-INFO%d' % (ii+1,)
            self.hdul.append(kpi_hdu)

        # include additional information in fits header
        # ---------------------------------------------
        if self.WRAD is not None:
            self.hdul[0].header.add_comment(
                "Super Gaussian apodization radius used: %d pixels" % (
                    self.WRAD))
        else:
            self.hdul[0].header.add_comment("Data was not apodized")

        try:
            _ = self.kp_cov
            kcv_hdu = fits.ImageHDU(self.kp_cov.astype(np.float64))
            kcv_hdu.header.add_comment(
                "Kernel-phase covariance matrix")
            kcv_hdu.header['EXTNAME'] = 'KP-COV'
            self.hdul.append(kcv_hdu)
            self.hdul[0].header.add_comment(
                "KP covariance extension included")
        except AttributeError:
            print("No covariance added")
            self.hdul[0].header.add_comment(
                "No covariance information provided")

        # ------------------------
        self.hdul.writeto(fname, overwrite=True)

    # =========================================================================
    # =========================================================================
    def kpd_binary_match_map(
            self, gsz, gstep, kp_signal, cwavel, cref=0.01, norm=False):
        """ ---------------------------------------------------------------
        Produces a 2D-map showing where the best binary fit occurs for
        the kp_signal vector provided as an argument

        Computes the dot product between the kp_signal and a (x,y) grid of
        possible positions for the companion, for a pre-set contrast.

        Parameters:
        ----------
        - gsz       : grid size (gsz x gsz)
        - gstep     : grid step in mas
        - kp_signal : the kernel-phase vector
        - cwavel    : wavelength value (float).
                      Should be the element of `KPO.CWAVEL` corresponding to the data if available.
        - cref      : reference contrast (optional, default = 0.01)
        - norm      : normalizes the map (boolean, default = False)

        Remarks:
        -------
        In the high-contrast regime, the amplitude is proportional to the
        companion brightness ratio.

        With the map normalized, an estimate of the contrast at the
        brightest correlation peak can directly be read.
        --------------------------------------------------------------- """
        mgrid = np.zeros((gsz, gsz))
        cvis = 1.0 + cref * core.grid_precalc_aux_cvis(
            self.kpi.UVC[:, 0],
            self.kpi.UVC[:, 1],
            cwavel, mgrid, gstep)

        kpmap = self.kpi.KPM.dot(np.angle(cvis))
        crit = kpmap.T.dot(kp_signal)

        if norm is not False:
            crit /= kp_signal.dot(kp_signal) * cref
        return(crit.reshape(gsz, gsz))

    # =========================================================================
    def kpd_colinearity_map(self, gsz, gstep, index=0, cref=0.01, norm=True):
        """ ---------------------------------------------------------------
        Produces a 2D grid map showing where the best binary fit occurs
        for a KP dataset that is part of the current instance of KPO.

        Computes the dot product between the signal and a (x,y) grid of
        possible positions for the companion, for a pre-set contrast.

        The purpose of this version of the kpd_binary_match_map() is to
        handle field rotation. It is (for now) more CPU intensive and slower!

        Parameters:
        ----------
        - gsz   : grid size (gsz x gsz)
        - gstep : grid step in mas
        - index : the index of the dataset in the current data structure
                  default value is 0 (first dataset available)
        - cref  : reference contrast (optional, default = 0.01)
        - norm  : normalizes the map (boolean, default = False)
        --------------------------------------------------------------- """

        try:
            _ = self.TARGET[index]
        except IndexError:
            print("Requested dataset (index=%d) does not exist" % (index,))
            print("No data-matching binary model can be built.")
            print("For generic binary model, use xara.core.cvis_binary()")
            return

        xx, yy = np.meshgrid(np.arange(gsz)-gsz//2, np.arange(gsz)-gsz//2)
        azim = - np.arctan2(xx, yy) * 180.0 / np.pi
        dist = np.hypot(xx, yy) * gstep
        cmap = np.zeros_like(dist)

        azim_flat = azim.flatten()
        dist_flat = dist.flatten()
        cmap_flat = cmap.flatten()
        kpdata_flat = self.KPDT[index].flatten()

        ng = gsz**2
        nd = int(1 + np.log10(ng))
        for ii in range(ng):
            params = [dist_flat[ii], azim_flat[ii], 1/cref]
            cmap_flat[ii] = kpdata_flat.dot(
                self.kpd_binary_model(params).flatten())
            print("\rColinearity: %0{:d}d/%0{:d}d".format(nd, nd) % (ii+1, ng),
                  end="", flush=True)

        print()
        cmap_flat /= kpdata_flat.dot(kpdata_flat) * cref
        return cmap_flat.reshape(gsz, gsz)

    # =========================================================================
    # =========================================================================
    def kpd_binary_cdet_map(self, gsz, gstep, kp_var, cref=0.01):
        """Produces a 2D 1-sigma contrast detection limit map.
        ---------------------------------------------------------------

        Uses kernel-phase variance (vector) or covariance (matrix) kp_var
        and two parameters that define a grid.

        In the high-contrast regime, the kernel signal is proportional to the
        contrast (secondary/primary convention).

        Parameters:
        ----------
        - gsz     : grid size (gsz x gsz)
        - gstep   : grid step in mas
        - kp_var  : the kernel-phase variance (vector) or covariance (matrix)
        - cref    : reference contrast (optional, default = 0.01)

        Returns:
        -------
        A 2D map of attainable contrast (in magnitudes)
        --------------------------------------------------------------- """
        mgrid = np.zeros((gsz, gsz))
        cvis = 1.0 + cref * core.grid_precalc_aux_cvis(
            self.kpi.UVC[:, 0], self.kpi.UVC[:, 1],
            self.CWAVEL, mgrid, gstep)

        kpmap = self.kpi.KPM.dot(np.angle(cvis)) / cref
        # covariance matrix case!
        if len(kp_var.shape) == 2:
            cov_inv = np.linalg.pinv(kp_var)
            ng = gsz*gsz  # number of grid points
            clim = np.zeros((ng))
            for ii in range(ng):
                k0 = kpmap[:, ii]
                tmp = np.dot(np.dot(k0, cov_inv), k0)
                if tmp < 1e-20:
                    tmp = 1e-20
                clim[ii] = tmp**-0.5
        # only variance is available
        else:
            clim = 1.0 / np.sqrt((1.0/kp_var).T.dot(kpmap**2))
        cmag = -2.5 * np.log10(clim)
        return(cmag.reshape(gsz, gsz))

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
            _ = self.TARGET[index]
        except IndexError:
            print("Requested dataset (index=%d) does not exist" % (index,))
            print("No data-matching binary model can be built.")
            print("For generic binary model, use xara.core.cvis_binary()")
            return

        nbd = self.KPDT[index].shape[0]
        detpa_i = self.DETPA[index]
        if isinstance(detpa_i, float):
            detpa_i = np.full(nbd, detpa_i)

        sim = []

        # compute binary complex visibilities at multiple DETPA
        for ii in range(nbd):
            temp = self.__cvis_binary_model(params, detpa_i[ii])
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
        if nbd == 1:
            return np.array(sim)[0]
        else:
            return np.array(sim)

    # =========================================================================
    # =========================================================================
    def binary_model_fit_residuals(
            self, params, index=0, calib=None, obs="KERNEL"):
        model = self.kpd_binary_model(params, index, obs)

        # ----------------
        # calibrator first
        # ----------------
        if calib is not None:
            if isinstance(calib, int):
                if self.KPDT[calib].ndim == 2:
                    clbrt = np.mean(self.KPDT[calib], axis=0)
                    clerr = np.std(self.KPDT[calib], axis=0)
                else:
                    clbrt = self.KPDT[calib]
                    clerr = np.zeros(self.kpi.nbkp)

            else:  # assumes that the calib is another kpo
                if calib.KPDT[0].ndim == 2:
                    clbrt = np.mean(calib.KPDT[0], axis=0)
                    clerr = np.std(calib.KPDT[0], axis=0)
                else:
                    clbrt = calib.KPDT[0]
                    clerr = np.zeros(self.kpi.nbkp)
        else:
            clbrt = np.zeros(self.kpi.nbkp)
            clerr = np.zeros(self.kpi.nbkp)

        # ------------------
        # target of interest
        # ------------------
        if (self.KPDT[index].ndim == 2) and (np.shape(self.KPDT[index])[0] > 1):
            model = np.mean(model, axis=0)
            error = np.mean(self.KPDT[index], axis=0) - clbrt - model
            uncrt = np.sqrt(np.var(self.KPDT[index], axis=0) + clerr**2)
        else:
            error = self.KPDT[index][0] - clbrt - model
            uncrt = None

        if uncrt is not None:
            error /= uncrt
        return(error)

    # =========================================================================
    # =========================================================================
    def binary_model_fit(self, p0, index=0, calib=None, obs="KERNEL"):
        ''' ------------------------------------------------------------------
        Least square fit a binary to the data.

        Parameters:
        ----------
        - p0: initial guess (3 parameter vector) eg. [100.0, 0.0, 5.0]

        optional:
        - index: the index of the dataset in the current data structure
                 default value is 0 (first dataset available)
        - calib: another kpo structure containing a calibrator, or a
                 different index, corresponding to a calibrator dataset
                 in the current data structure
        - obs: just an idea for now?
        ------------------------------------------------------------------ '''
        try:
            _ = self.TARGET[index]
        except IndexError:
            print("Requested dataset (index=%d) does not exist" % (index,))
            print("No data-matching binary model can be built.")
            return

        soluce = leastsq(self.binary_model_fit_residuals,
                         p0, args=((index, calib, obs,)), full_output=1)

        return(soluce)

    # =========================================================================
    # =========================================================================
    def __cvis_binary_model(self, params, detpa):
        ''' ------------------------------------------------------------------
        Private call to xara.core.cvis_binary(), using KPO object properties.
        ------------------------------------------------------------------ '''
        u = self.kpi.UVC[:, 0]
        v = self.kpi.UVC[:, 1]
        wl = self.CWAVEL
        return(core.cvis_binary(u, v, wl, params, detpa % 360))

    # =========================================================================
    # =========================================================================
    def scatter_uv_map(self, data, sym=True, title="", cbar=True, fsize=5,
                       cmap=cm.rainbow, marker='o', ssize=12, lw=0):
        ''' ------------------------------------------------------------------
        Produce a 2D scatter map of data in the Fourier plane.

        Note:
        ----

        About the *sym* parameter: depending on what you want to display, you
        need to set this flag to the appropriate state: displaying *phase*
        requires *sym=False*, since the phase is antisymmetric!

        Parameters:
        ----------

        - data   : the 1D vector of size self.kpi.nbuv
        - sym    : symmetric or anti-symmetric map      (default=True)
        - title  : string title to add to the main plot (default="")
        - cbar   : add a colorbar                       (default=True)
        - fsize  : figure size in inches                (default=5)
        - cmap   : matplotlib colormap                  (default=cm.rainbow)
        - marker : matplotlib marker                    (default='o')
        - ssize  : marker size                          (default=12)
        - lw     : line width for symbol outline        (default=0)
        ------------------------------------------------------------------ '''
        xx, yy = self.kpi.UVC.T
        xx2, yy2 = np.append(xx, -xx), np.append(yy, -yy)
        data2 = np.append(data, data) if sym else np.append(data, -data)

        ssz = ssize**2  # symbol size
        dtitle = "Fourier map" if title == "" else title

        if not cbar:
            fig, ax = plt.subplots()
            fig.set_size_inches(fsize, fsize, forward=True)
            ax.scatter(xx2, yy2, c=data2, cmap=cmap, s=ssz, marker=marker, lw=lw)
            ax.set_title(dtitle)
            ax.set_xlabel("Fourier u-coordinate (meters)")
            ax.set_ylabel("Fourier v-coordinate (meters)")
            ax.axis('equal')

        else:
            fsize2 = np.array([1, 0.1]) * fsize
            fig, axes = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': fsize2.tolist()})
            axes[0].scatter(xx2, yy2, c=data2, cmap=cmap,
                            s=ssz, marker=marker, lw=lw)
            axes[0].set_title(dtitle)
            axes[0].set_xlabel("Fourier u-coordinate (meters)")
            axes[0].set_ylabel("Fourier v-coordinate (meters)")
            axes[0].axis('equal')
            axes[0].set_title(dtitle)
            foo = cm.ScalarMappable(cmap=cmap)     # for the colorbar
            foo.set_array(np.append(data, data2))  # prepping the range
            fig.colorbar(foo, cax=axes[1], orientation='vertical')

        fig.set_tight_layout(True)
        return fig

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
