#!/usr/bin/env python

''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data
    structures, using the theory described in the following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.
    - Martinache et al, 2020, A&A, 636, A72

    This file contains the definition of the IWFS class
    -------------------------------------------------------------------- '''

import numpy as np
from . import kpi
from . import core

shift = np.fft.fftshift
fft = np.fft.fft2
ifft = np.fft.ifft2

i2pi = 1j*2*np.pi


def fuzzy_intersect1(arr1, arr2, rr=2, single=True):
    ''' -----------------------------------------------------------------------
    Computes the fuzzy intersection between two 1D arrays of floating point
    numbers with a floating point rounding error *rr*

    Parameters:
    ----------
    - arr1   : the first 1D array of floats
    - arr2   : the second 1D array of floats
    - rr     : the rounding error (default=2 -> two digits)
    - single : if True, returns only the first value found

    Returns:
    -------
    the values that checks abs(v1 - v2) < 10**-rr or []
    if single is True returns only the first value found
    ----------------------------------------------------------------------- '''
    res = []
    for ii, val in enumerate(arr1):
        test = list(filter(lambda x: np.abs(x - val) <= 10**(-rr), arr2))

        if test != []:
            res.append(test[0])
            if single:
                return test[0]  # return the first value that matches

    return res


# =============================================================================
# =============================================================================
class IWFS():
    ''' -----------------------------------------------------------------------
    IWFS stands for interferometric wavefront sensor

    This class is designed to facilitate the use of the Fourier-phase framework
    for wavefront sensing purposes.
    ----------------------------------------------------------------------- '''

    def __init__(self, fname=None, array=None, ID="IWFS", nl=1):
        ''' -------------------------------------------------------------------
        Instantiation of an interferometric wavefront sensor.

        This object relies primarily on the KPI data structure that it keeps
        as a class attribute.

        If nl > 1, multiple Fourier matrices will be stored

        Parameters:
        ----------
        - fname : text of fits file storing model information
        - array : array of coordinates for a pupil model
        - ID    : a string to identifiy the PIWFS
        - nl    : number of wavelength (integer, default=1)
        ------------------------------------------------------------------- '''
        self.kpi = kpi.KPI(fname=fname, array=array, ID=ID)
        self.update_wfs_properties()

        self.ID = ID                  # name of this IWFS
        self.nl = nl                  # number of bandpasses
        self.nwmax = 5                # expected OPD range (+/- nwmax waves)

        if self.nl == 1:
            print("Monochromatic fringe tracker")

        self.M2PIX = -1 * np.ones(self.nl)
        self.cwl = np.zeros(self.nl)
        self.ISZ = np.zeros(self.nl, dtype=int)
        self.pscale = np.zeros(self.nl)
        self.FF = [None] * self.nl    # Fourier transform arrays
        self.cvis = [None] * self.nl  # complex visibilities
        self.opde = [None] * self.nl  # monochromatic OPD estimates

        # if the file is a complete (kpi + wfd) structure
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

    # =========================================================================
    # =========================================================================
    def svd(self,):
        ''' -------------------------------------------------------------------
        Computes and stores the SVD of the model phase transfer matrix
        ------------------------------------------------------------------- '''
        self.U, self.S, self.Vt = np.linalg.svd(self.kpi.TFM, full_matrices=0)

    # =========================================================================
    # =========================================================================
    def update_wfs_properties(self,):
        ''' -------------------------------------------------------------------
        Computes pseudo-inverse of the model phase transfer matrix
        ------------------------------------------------------------------- '''
        self.PINV = np.linalg.pinv(self.kpi.TFM)

    # =========================================================================
    # =========================================================================
    def update_img_properties(self, isz=None, wl=None, pscale=None, ii=0):
        ''' -------------------------------------------------------------------
        Provide or update the properties of the images fed to the IWFS

        Parameters:
        ----------
        - isz    : image size in pixels (integer: assumes square images!)
        - wl     : central wavelength (float: in meters)
        - pscale : the detector plate scale (float: in mas/pixels)
        - ii     : image index (between 0 and self.nl - 1)
        ------------------------------------------------------------------- '''
        print(f"Updating properties for image channel #{ii}:")

        if wl is not None:
            self.cwl[ii] = wl
            prm = 'central wlength'
            print(f"Updated {prm:<24} -> {self.cwl[ii]*1e6 :.2f} microns")

        if isz is not None:
            self.ISZ[ii] = int(isz)
            prm = 'image size'
            print(f"Updated {prm:<24} -> {self.ISZ[ii]} pixels (square)")

        if pscale is not None:
            self.pscale[ii] = pscale
            prm = 'plate scale'
            print(f"Updated {prm:<24} -> {self.pscale[ii] :.2f} mas/pixel")

        try:
            m2pix = core.mas2rad(self.pscale[ii]) * self.ISZ[ii] / self.cwl[ii]
            if self.M2PIX[ii] != m2pix:
                self.M2PIX[ii] = m2pix
                self.FF[ii] = None  # erasing the FF matrix!
            prm = 'm2pix parameter'
            print(f"Updated {prm:<24} -> {self.M2PIX[ii]:.2f} meters to pixel")
        except NameError:
            print("Missing image information to compute m2pix!")
            self.M2PIX[ii] = -1

    # =========================================================================
    # =========================================================================
    def extract_data(self, image, ii=0):
        ''' -------------------------------------------------------------------
        Updates internal structure:
        - cvis: complex visibility
        - opde: monochromatic OPD estimate (in microns)
        ------------------------------------------------------------------- '''
        self.cvis[ii] = self.extract_cvis_from_img(image, ii)
        phi = np.angle(self.cvis[ii]) 
        self.opde[ii] = phi * self.cwl[ii] * 1e6 / (2*np.pi)

    # =========================================================================
    # =========================================================================
    def extract_cvis_from_img(self, image, ii=0):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        if self.M2PIX[ii] == -1:
            print("M2PIX is not set: please run update_img_properties(...)")
            print("Can't extract complex visibilities image with index #{ii}")
            return None
        return self.__extract_cvis_ldft1(image, ii)

    # =========================================================================
    # =========================================================================
    def __extract_cvis_ldft1(self, image, ii=0):
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
        ------------------------------------------------------------------- '''
        try:
            test = self.FF[ii].shape
        except AttributeError:
            print("LDFT1: Computing new Fourier matrix...")
            self.FF[ii] = core.compute_DFTM1(
                self.kpi.UVC, self.M2PIX[ii], self.ISZ[ii])
            print("Done!")

        myft_v = self.FF[ii].dot(image.flatten())
        myft_v *= self.kpi.TRM.sum() / image.sum()
        return(myft_v)

    # =========================================================================
    # =========================================================================
    def get_opd(self):
        ''' -------------------------------------------------------------------
        Computes the optical path difference in microns for the provided
        set of complex visibilities measured at the different wlengths.

        This algorithm uses the fuzzy intersection function to find matches
        across possible OPDs identified for the different spectral channels.

        Currently set to operate with one or two filters but no more! More
        channels will require some kind of recursion.
        ------------------------------------------------------------------- '''

        if self.nl == 1:  # monochromatic analysis (simple)
            opd = - self.PINV.dot(self.opde[0])

        else:  # polychromatic phase unwrapping
            if self.nl > 2:
                print("Can't do that just yet!")
                return

            nw = self.nwmax              # local short-hand
            nuv = self.kpi.nbuv          # local short-hand
            delta = np.array(self.opde)  # match my earlier notations
            klambda = np.outer(np.arange(-nw, nw+1), self.cwl) * 1e6

            DELTA = np.zeros(nuv)  # True OPDs here

            for ii in range(nuv):
                pval = np.outer(np.ones(2*nw+1), delta[:, ii]) + klambda
                DELTA[ii] = fuzzy_intersect1(pval[:, 0], pval[:, 1], rr=2, single=True)
            self.test = DELTA
            opd = - self.PINV.dot(DELTA)
        return opd

    # =========================================================================
    # =========================================================================
    def get_opd2(self):
        ''' -------------------------------------------------------------------
        Computes the optical path difference in microns for the provided
        set of complex visibilities measured at the different wlengths.

        This algorithm uses a pre-computed set of cvis for the different
        spectral channels as a function of OPD value.

        This is the first approach used for the lab demo of Heimdallr!
        ------------------------------------------------------------------- '''

        if self.nl == 1:  # monochromatic analysis (simple)
            opd = - self.PINV.dot(self.opde[0])

        else:  # polychromatic phase unwrapping
            print("Not implemented yet!")
        return opd
