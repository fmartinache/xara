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
      --> name : name of the model (HST, Keck, Annulus_19, ...)
      --> VAC  : virtual array coordinates
      --> TRM  : transmission of virtual array coordinates
      --> UVC  : matching array of coordinates in uv plane (baselines)
      --> RED  : vector coding the redundancy of these baselines
      --> BLM  : baseline mapping model
      --> TFM  : transfer matrix, linking pupil-phase to uv-phase
      --> KPM  : array storing the kernel-phase relations

    Aliases:
    -------
    - mask   = VAC
    - KerPhi = KPM
    - uv     = UVC

    Note:
    ----

    - TFM = RED^{-1} * BLM (i.e. TFM = R^{-1} * A)
    - KPM = Ker(TFM)
    -------------------------------------------------------------------- '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import pickle
import gzip

from .core import hexagon


class KPI(object):
    ''' Fundamental kernel-phase relations

    -----------------------------------------------------------------------
    This object condenses all the knowledge about a given instrument pupil
    geometry into a series of arrays useful for kernel-phase analysis as
    well as for wavefront sensing.
    ----------------------------------------------------------------------- '''

    # =========================================================================
    # =========================================================================

    def __init__(self, fname=None, array=None, ndgt=5,
                 bmax=None, hexa=False, ID=""):
        ''' Default instantiation of a KerPhase_Relation object:

        -------------------------------------------------------------------
        Default instantiation of this KerPhase_Relation class is achieved
        by loading a pre-made file, containing all the relevant information

        Parameters:
        ----------
        - fname: a valid file name, expected to be either a multi-extension
                 FITS file respecting a definition agreed upon in Nov 2017
                 or a text file containing (x,y) coordinates.

        - array: If no file name is provided, a 2- or 3-column array
                 containing the (x,y) coordinates or the (x,y,t) coordinates
                 (t being the transmission) of a virtual interferometric
                 aperture, in meters is required. Not used if a valid file
                 name was provided.

        Option:
        ------
        - ndgt: (integer) number of digits when rounding x,y baselines
        - bmax: length of the max baseline kept in the model (in meters)
        - ID  : (string) give the KPI structure a human readable ID

        Remarks:
        -------

        Coordinates in file or array are in meters. For the baseline search
        algorithm to take place smoothly when building up the discrete model,
        the coordinates should be given with a reasonably good number of
        digits, typically 5 or 6, especially if the coordinates are located
        on a grid that is rotated by a non-trivial angle.

        The bmax option makes it possible to filter out very noisy baselines
        by eliminating them from the model. This results in less kernels but
        improves the overall performance in the presence of noisy data.
        -------------------------------------------------------------------'''

        if fname is not None:
            print("Attempting to load file %s" % (fname,))
            if '.fits' in fname:
                try:
                    self.load_from_fits(fname)
                except OSError:
                    raise Exception("Invalid KERNEL FITS data structure")

            elif '.kpi.gz' in fname:
                try:
                    self.load_from_pickle(fname)
                except OSError:
                    raise Exception("Invalid KERNEL pickled data structure")

            else:
                try:
                    self.load_aperture_model(fname=fname)
                except OSError:
                    raise Exception("Not a valid coordinate file")
                self.rebuild_model(ndgt=ndgt, bmax=bmax, hexa=hexa)

            print("KPI data successfully loaded")

        elif array is not None:
            print("Attempting to build KPI from array")
            try:
                self.load_aperture_model(data=array)
                self.rebuild_model(ndgt=ndgt, bmax=bmax, hexa=hexa)
                self.name = ID
            except:
                print("Problem using array %s" % (array,))
                print("Array cannot be used to create KPI structure")
            print("KPI data successfully created")
        else:
            print("KPI constructor requires startup data! Use either:")
            print("1. KPI(fname='my_model.fits')   # (if KERNEL   FITS)")
            print("2. KPI(fname='my_model.kpi.gz') # (if KERNEL pickle)")
            print("3. KPI(fname='my_coords.txt)    # (if (x,y)  coords)")
            print("4. KPI(array=[[x1, y1], [x2, y2], ..., [xn, yn]]) ")
            return None

    # =========================================================================
    def __del__(self):
        print("%s deleted " % (repr(self),))

    # =========================================================================

    def load_aperture_model(self, fname=None, data=None):
        ''' ------------------------------------------------------------------
        Create a virtual aperture model from the coordinates provided in a
        text file. Expected format is a two or three column array.

        - col #1: virtual aperture x-coordinate in meters
        - col #2: virtual aperture y-coordinate in meters
        - col #3: virtual aperture transmission (0 < t <= 1) (optional)
        ------------------------------------------------------------------ '''
        if fname is not None:
            self.VAC = 1.0 * np.loadtxt(fname)  # sub-Ap. coordinate files
            self.name = fname.split('/')[-1]    # use file name for KPI name
        else:
            if data is not None:
                self.VAC = data
                self.name = "no_name"
            else:
                print("Data is unavailable")
                return None

        self.nbap = self.VAC.shape[0]      # number of sub-Ap

        self.TRM = np.ones(self.nbap, dtype=float)

        if self.VAC.shape[1] == 3:
            self.TRM = self.VAC[:, 2]
        self.mask = self.VAC             # backward compatibility
        if self.VAC.shape[1] == 2:
            self.VAC = np.hstack((self.VAC, np.ones((self.nbap, 1))))

    # =========================================================================
    # =========================================================================

    def rebuild_model(self, ndgt=5, bmax=None, hexa=False):
        ''' ------------------------------------------------------------------
        Builds or rebuilds the Fourier-phase model, from the information
        provided by the virtual aperture coordinate (VAC) table.

        Parameters:
        ----------

        - ndgt: the number of digits to round baselines (in meters)

        - bmax: a baseline filtering parameter (in meters): if not None,
          baselines larger than bmax are eliminated from the discrete model

        - hexa: if True, masks out noisy baselines using a hexagonal shape

        ------------------------------------------------------------------ '''

        prec = 10**(-ndgt)

        # 1. Determine all possible combinations of virtual coordinates
        # if the array is redundant, there will be plenty of duplicates.
        # --------------------------------------------------------------

        nbap = self.nbap                 # shorthand
        uvx = np.zeros(nbap * (nbap-1))  # prepare empty arrays to store
        uvy = np.zeros(nbap * (nbap-1))  # the baselines
        bgn = np.zeros(nbap * (nbap-1))  # and their "gain"

        k = 0  # index for possible combinations (k = f(i,j))

        uvi = np.zeros(nbap * (nbap-1), dtype=int)  # arrays to store possible
        uvj = np.zeros(nbap * (nbap-1), dtype=int)  # combinations k=f(i,j) !!

        for i in range(nbap):      # do all the possible combinations of
            for j in range(nbap):  # sub-apertures
                if i != j:
                    uvx[k] = self.VAC[i, 0] - self.VAC[j, 0]
                    uvy[k] = self.VAC[i, 1] - self.VAC[j, 1]
                    bgn[k] = (self.VAC[i, 2] * self.VAC[j, 2])
                    # ---
                    uvi[k], uvj[k] = i, j
                    k += 1

        # search for distinct uv components along the u-axis
        temp = np.sort(uvx)
        mark = np.append(True, np.diff(temp))
        a = temp[mark > prec]
        nbx = a.shape[0]               # number of distinct u-components
        uv_sel = np.zeros((0, 2))      # array for "selected" baselines

        for i in range(nbx):     # identify distinct v-coords and fill uv_sel
            b = np.where(np.abs(uvx - a[i]) <= prec)
            c = np.unique(np.round(uvy[b], ndgt))

            nby = np.shape(c)[0]  # number of distinct v-compoments
            for j in range(nby):
                uv_sel = np.append(uv_sel, [[a[i], c[j]]], axis=0)

        self.nbuv = np.shape(uv_sel)[0]//2  # actual # of distinct uv points
        self.UVC = uv_sel[:self.nbuv, :]  # discard second half (symmetric)
        print("%d distinct baselines were identified" % (self.nbuv,))
        self.BLEN = np.hypot(self.UVC[:, 0], self.UVC[:, 1])

        # 1.5. Special case: baseline filtering
        # -------------------------------------
        if bmax is not None:
            if hexa:
                flat_to_flat = 2.*6.5 # m, size is doubled in uv-plane
                mask = hexagon(1024, 512*2.*float(bmax)/flat_to_flat, interp_edge=False) # center is (512, 512)
                uv_sampl = self.UVC.copy()   # copy previously identified baselines
                # uvm = np.abs(self.UVC).max() # max baseline length

                xx = uv_sampl[:, 0]/flat_to_flat*512.+512.
                yy = uv_sampl[:, 1]/flat_to_flat*512.+512.

                keep = np.array([mask[int(round(yy[ii])), int(round(xx[ii]))] for ii in range(uv_sampl.shape[0])]) > 0.5
                self.UVC = uv_sampl[keep]
                self.nbuv = (self.UVC.shape)[0]

                print("%d baselines were preserved after filtering" % (self.nbuv,))
                self.BMAX = bmax
                self.BLEN = np.hypot(self.UVC[:, 0], self.UVC[:, 1])
            else:
                uv_sampl = self.UVC.copy()   # copy previously identified baselines
                # uvm = np.abs(self.UVC).max() # max baseline length

                blength = np.sqrt(np.abs(uv_sampl[:, 0])**2 +
                                  np.abs(uv_sampl[:, 1])**2)

                keep = (blength < bmax)
                self.UVC = uv_sampl[keep]
                self.nbuv = (self.UVC.shape)[0]

                print("%d baselines were preserved after filtering" % (self.nbuv,))
                self.BMAX = bmax
                self.BLEN = np.hypot(self.UVC[:, 0], self.UVC[:, 1])

        # 2. compute baseline mapping model + redundancy
        # ----------------------------------------------
        self.BLM = np.zeros((self.nbuv, self.nbap), dtype=float)  # matrix
        self.RED = np.zeros(self.nbuv, dtype=float)               # Redundancy

        for i in range(self.nbuv):
            a = np.where((np.abs(self.UVC[i, 0] - uvx) <= prec) *
                         (np.abs(self.UVC[i, 1] - uvy) <= prec))

            self.BLM[i, uvi[a]] += bgn[a]
            self.BLM[i, uvj[a]] -= bgn[a]
            self.RED[i] = np.sum(bgn[a])
            '''
            self.BLM[i, uvi[a]] +=  1.0
            self.BLM[i, uvj[a]] += -1.0
            self.RED[i]          = np.size(a)
            '''

        # 3. Determine the kernel-phase relations
        # ----------------------------------------

        self.compute_KPM()
        # # backward compatibility
        # # ----------------------
        self.backward_compatibility_patch()

    # =========================================================================
    # =========================================================================

    def filter_baselines(self, criterion=None):
        ''' ------------------------------------------------------------------
        Filter (ie eliminate) baseline based on a criterion mask
        The criterion must be met for the baseline to be preserved!

        Parameters:
        ----------
        - criterion: an array of boolean of size *nbuv* (before filtering)

        Example of criterion excluding all baselines for which R < 1:

        criterion = kpo.kpi.RED > 1
        ------------------------------------------------------------------ '''
        if criterion is not None:
            uv_sampl = self.UVC.copy()
            self.UVC = uv_sampl[criterion]
            self.BLM = self.BLM[criterion, :]
            self.RED = self.RED[criterion]
            self.nbuv = self.RED.size
            self.BLEN = np.hypot(self.UVC[:, 0], self.UVC[:, 1])
            self.compute_KPM()
            self.backward_compatibility_patch()

    # =========================================================================
    # =========================================================================

    def compute_KPM(self,):
        ''' ------------------------------------------------------------------
        Compute the kernel-phase matrix for the model

        Takes the BLM + RED to build TFM and then computes the KPM.

        Remarks:
        -------
        - Was isolated from rebuild_model() to make it easier to maintain.
        - the computation of KPM can be written in a more concise manner!
        ------------------------------------------------------------------ '''
        # One sub-aperture is taken as reference: the corresponding
        # column of the transfer matrix is discarded. TFM is now a
        # (nbuv) x (nbap - 1) array.

        # the first column of TFM is discarded: the first aperture
        # of the model is used as a reference.

        self.TFM = self.BLM.copy()
        self.TFM = self.TFM[:, 1:]                         # cf. explanation
        self.TFM = np.dot(np.diag(1./self.RED), self.TFM)  # redundancy

        # U, S, Vh = np.linalg.svd(self.TFM.T, full_matrices=1)
        U, S, Vh = np.linalg.svd(self.BLM[:, 1:].T, full_matrices=1)

        S1 = np.zeros(self.nbuv)
        S1[0:self.nbap-1] = S

        self.nbkp = np.size(np.where(abs(S1) < 1e-3))  # number of Ker-phases
        KPhiCol = np.where(abs(S1) < 1e-3)[0]
        self.KPM = np.zeros((self.nbkp, self.nbuv))  # allocate the array

        for i in range(self.nbkp):
            self.KPM[i, :] = (Vh)[KPhiCol[i], :]

        self.KPM = self.KPM.dot(np.diag(self.RED))
        norm = np.linalg.norm(self.KPM, axis=1)
        self.KPM = np.divide(self.KPM.T, norm).T

        print('first %d singular values for this array:' % (
            min(10, self.nbap-1)))
        print(np.round(S[:10], 5))
        print(self)

    # =========================================================================
    # =========================================================================

    def backward_compatibility_patch(self,):
        ''' ------------------------------------------------------------------
        Some aliases are defined here to provide compatibility with
        software using an earlier version of this library.
        ------------------------------------------------------------------ '''
        self.mask = self.VAC
        self.KerPhi = self.KPM
        self.uv = self.UVC

    # =========================================================================
    # =========================================================================

    def load_from_fits(self, fname):
        ''' ------------------------------------------------------------------
        Load KPI data from a multi-extension FITS file.

        Parameters:
        ----------
        - fname: the file name. e.g.: "my_data.fits"
        ------------------------------------------------------------------ '''
        try:
            hdulist = fits.open(fname)
        except OSError:
            print("%s is not a valid FITS file" % (fname,))
            raise OSError
        try:
            self.name = hdulist['PRIMARY'].header['KPI-ID']
        except KeyError:
            self.name = fname.split('/')[-1]

        # ------------------------------------
        #    APERTURE is an OPTIONAL HDU
        # ------------------------------------
        ap_flag = True
        try:
            tmp = hdulist['APERTURE'].data
            self.VAC = np.array([tmp['XXC'], tmp['YYC']]).T
            self.nbap = self.VAC.shape[0]
            try:
                self.TRM = np.array(tmp['TRM'])
            except KeyError:
                self.TRM = np.ones(self.nbap, dtype=float)
            self.VAC = np.array([tmp['XXC'], tmp['YYC'], tmp['TRM']]).T
            self.mask = self.VAC
        except KeyError:
            print('APERTURE HDU not available')
            ap_flag = False

        # ------------------------------------
        #    UV-PLANE is an REQUIRED HDU
        # ------------------------------------
        tmp = hdulist['UV-PLANE'].data
        self.UVC = np.array([tmp['UUC'], tmp['VVC']]).T
        self.RED = np.array(tmp['RED']).astype(float)

        self.KPM = hdulist['KER-MAT'].data
        self.BLM = hdulist['BLM-MAT'].data
        self.BLEN = np.hypot(self.UVC[:, 0], self.UVC[:, 1])

        if ap_flag:
            self.TFM = np.diag(1./self.RED).dot(self.BLM[:, 1:])
            # self.TFM = np.dot(np.diag(1./self.RED), self.TFM)  # redundancy

        self.nbuv = self.UVC.shape[0]
        self.nbkp = self.KPM.shape[0]

        self.KerPhi = self.KPM
        self.uv = self.UVC

        # ---------------------------------
        # ---------------------------------
        hdulist.close()

    # =========================================================================
    # =========================================================================

    def load_from_pickle(self, fname):
        try:
            # -------------------------------
            # load the pickled data structure
            # -------------------------------
            myf = gzip.GzipFile(fname, "r")
            data = pickle.load(myf)
            myf.close()

            # -------------------------------
            # restore the variables for this
            # session of Ker-phase use!
            # -------------------------------
            try:
                self.name = data['name']
            except KeyError:
                self.name = "UNKNOWN"

            self.VAC = data['mask']
            self.UVC = data['uv']
            self.RED = data['RED']
            self.KPM = data['KerPhi']
            self.TFM = data['TFM']

            self.nbap = self.VAC.shape[0]
            self.nbuv = self.UVC.shape[0]
            self.nbkp = self.KPM.shape[0]

            self.backward_compatibility_patch()

            self.BLM = np.diag(1.0/self.RED).dot(self.TFM)
        except:
            print("File %s isn't a valid Ker-phase data structure" % (fname,))
            return None

    # =========================================================================
    # =========================================================================

    def __str__(self):
        msg = "%s KPI data structure\n" % (repr(self),)
        msg += "-" * 40 + "\n"
        msg += "-> %3d sub-apertures\n" % (self.nbap)
        msg += "-> %3d distinct baselines\n" % (self.nbuv)
        msg += "-> %3d Ker-phases (%5.1f %% target phase)\n" % (
            self.nbkp, (100.0*self.nbkp)/self.nbuv,)
        msg += "-> %3d Eig-phases (%5.1f %% wavefront phase)\n" % (
            self.nbuv-self.nbkp, 100.0*(self.nbuv-self.nbkp)/(self.nbap-1))
        msg += "-" * 40 + "\n"
        return msg

    # =========================================================================
    # =========================================================================

    def plot_pupil_and_uv(self, xymax=4.0, figsize=(12, 6), plot_redun=False,
                          cmap=cm.gray, ssize=12, lw=0, alpha=1.0, marker='s'):
        '''Nice plot of the pupil sampling and matching uv plane.

        --------------------------------------------------------------------
        Options:
        ----------

        - xymax: radius of pupil plot in meters           (default=4.0)
        - figsize: matplotlib figure size                 (default=(12,6))
        - plot_redun: bool add the redundancy information (default=False)
        - cmap: matplotlib colormap                       (default:cm.gray)
        - ssize: symbol size                              (default=12)
        - lw:  line width for symbol outline              (default=0)
        - alpha: gamma (transparency)                     (default=1)
        - maker: matplotlib marker for sub-aperture       (default='s')
        - -------------------------------------------------------------------
        '''

        f0 = plt.figure(figsize=figsize)
        plt.clf()
        ax0 = plt.subplot(121)

        s1, s2 = ssize**2, (ssize/2)**2
        p0 = ax0.scatter(self.VAC[:, 0], self.VAC[:, 1], s=s1, c=self.VAC[:, 2],
                         cmap=cmap, alpha=alpha, marker=marker, lw=lw)
        ax0.axis([-xymax, xymax, -xymax, xymax])
        ax0.set_aspect('equal')
        ax0.set_xlabel("Aperture x-coordinate (meters)")
        ax0.set_ylabel("Aperture y-coordinate (meters)")
        plt.colorbar(p0, ax=ax0)

        ax1 = plt.subplot(122)
        p1 = ax1.scatter(-self.UVC[:, 0], -self.UVC[:, 1], s=s2, c=self.RED,
                    cmap=cmap, alpha=alpha, marker=marker, lw=lw)
        ax1.scatter(self.UVC[:, 0], self.UVC[:, 1], s=s2, c=self.RED,
                    cmap=cmap, alpha=alpha, marker=marker, lw=lw)

        ax1.axis([-2*xymax, 2*xymax, -2*xymax, 2*xymax])
        ax1.set_aspect('equal')
        ax1.set_xlabel("Fourier u-coordinate (meters)")
        ax1.set_ylabel("Fourier v-coordinate (meters)")
        plt.colorbar(p1, ax=ax1)

        # complete previous plot with redundancy of the baseline
        # -------------------------------------------------------
        dy = 0.1*abs(self.uv[0, 1] - self.uv[1, 1])  # plot text offset
        if plot_redun:
            for i in range(self.nbuv):
                ax1.text(self.uv[i, 0]+dy, self.uv[i, 1]+dy,
                         int(self.RED[i]), ha='center')
            ax1.axis('equal')
        plt.draw()
        f0.set_tight_layout(True)
        return f0

    # =========================================================================
    # =========================================================================

    def package_as_fits(self, fname=None):
        ''' ---------------------------------------------------------------
        Packages the KPI data structure into a multi-extension FITS,
        that may be written to disk. Returns a hdu list.

        Parameters:
        ----------

        - fname: a file name. If provided, the hdulist is saved as a
        fits file.
        --------------------------------------------------------------- '''

        # prepare the data for fits table format
        # --------------------------------------
        xy1 = fits.Column(name='XXC', format='D', array=self.VAC[:, 0])
        xy2 = fits.Column(name='YYC', format='D', array=self.VAC[:, 1])
        trm = fits.Column(name='TRM', format='D', array=self.TRM)

        uv1 = fits.Column(name='UUC', format='D', array=self.UVC[:, 0])
        uv2 = fits.Column(name='VVC', format='D', array=self.UVC[:, 1])
        uv3 = fits.Column(name='RED', format='D', array=self.RED)

        # make up a primary HDU
        # ---------------------
        hdr = fits.Header()
        hdr['SOFTWARE'] = 'XARA'
        hdr['KPI-ID'] = self.name[:8]
        hdr['GRID'] = (False, "True for integer grid mode")
        hdr['G-STEP'] = (0.0, "Used for integer grid mode")
        hdr.add_comment("File created by the XARA python pipeline")
        try:
            _ = self.BMAX
            hdr.add_comment(
                "Model filtering baselines > %.1f meters" % (self.BMAX))
        except AttributeError:
            hdr.add_comment("Discrete model is complete")
        pri_hdu = fits.PrimaryHDU(header=hdr)

        # APERTURE HDU
        # ------------
        tb1_hdu = fits.BinTableHDU.from_columns([xy1, xy2, trm])
        tb1_hdu.header['EXTNAME'] = 'APERTURE'
        tb1_hdu.header['TTYPE1'] = (
            'XXC', 'Virtual aperture x-coord (in meters)')
        tb1_hdu.header['TTYPE2'] = (
            'YYC', 'Virtual aperture y-coord (in meters)')
        tb1_hdu.header['TTYPE3'] = (
            'TRM', 'Virtual aperture transmission (0 < t <=1)')

        # UV-PLANE HDU
        # ------------
        tb2_hdu = fits.BinTableHDU.from_columns([uv1, uv2, uv3])
        tb2_hdu.header['TTYPE1'] = ('UUC', 'Baseline u coordinate (in meters)')
        tb2_hdu.header['TTYPE2'] = ('VVC', 'Baseline v coordinate (in meters)')
        tb2_hdu.header['TTYPE3'] = ('RED', 'Baseline redundancy (float)')
        tb2_hdu.header['EXTNAME'] = 'UV-PLANE'

        # KER-MAT HDU
        # -----------
        kpm_hdu = fits.ImageHDU(self.KerPhi)
        kpm_hdu.header.add_comment("Kernel-phase Matrix")
        kpm_hdu.header['EXTNAME'] = 'KER-MAT'

        # BLM-MAT HDU
        # -----------
        # BLM = (np.diag(self.RED).dot(self.TFM))#.astype(np.int)
        blm_hdu = fits.ImageHDU(self.BLM)
        blm_hdu.header.add_comment("Baseline Mapping Matrix")
        blm_hdu.header['EXTNAME'] = 'BLM-MAT'

        # compile HDU list and save
        # -------------------------

        self.hdul = fits.HDUList([pri_hdu, tb1_hdu, tb2_hdu, kpm_hdu, blm_hdu])

        if fname is not None:
            self.hdul.writeto(fname, overwrite=True)
        return(self.hdul)

    # =========================================================================
    # =========================================================================

    def save_to_file(self, fname):
        ''' Export the KPI data structure as an external file

        ----------------------------------------------------------------
        Default export mode is as a multi-extension fits file. While this
        is the prefered option, for backward compatibility purposes, it is
        also possible to save the structure as a python pickle data
        structure, if the option 'pickle' is set to True.

        To save on disk space, this procedure uses the gzip module.
        While there is no requirement for a specific extension for the
        file, I would recommend that one uses ".kpi.gz", so as to make
        it obvious that the file is a gzipped kpi data structure.
        ----------------------------------------------------------------  '''
        if '.fits' in fname:
            self.package_as_fits(fname)
            return 0
        else:

            try:
                data = {'name': self.name,
                        'mask': self.VAC,
                        'uv': self.UVC,
                        'TFM': self.TFM,
                        'KerPhi': self.KPM,
                        'RED': self.RED}
            except AttributeError:
                print("KPI data structure is incomplete")
                print("File %s wasn't saved!" % (fname,))
                return None
            # -------------
            try:
                myf = gzip.GzipFile(fname, "wb")
            except:
                print("File %s cannot be created." % (fname,))
                print("KPI data structure wasn't saved.")
                return None
            # -------------
            pickle.dump(data, myf, -1)
            myf.close()
            return 0

###############################################################################
###############################################################################
###############################################################################
###############################################################################
