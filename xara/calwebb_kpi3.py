from __future__ import division
"""
JWST stage 3 pipeline for kernel-phase imaging.

Authors: Jens Kammerer
Supported instruments: NIRCam, NIRISS
"""

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

import matplotlib.patheffects as PathEffects

from scipy.ndimage import median_filter

from xara import core
from xara import kpo
from xara import calwebb_utils as ut

# http://svo2.cab.inta-csic.es/theory/fps/
wave_nircam = {'F212N': 2.121193}  # micron
weff_nircam = {'F212N': 0.027427}  # micron
wave_niriss = {'F277W': 2.739519,
               'F380M': 3.826384,
               'F430M': 4.282976,
               'F480M': 4.813019}  # micron
weff_niriss = {'F277W': 0.644830,
               'F380M': 0.201962,
               'F430M': 0.203914,
               'F480M': 0.297379}  # micron
# https://jwst-reffiles.stsci.edu/source/data_quality.html
pxdq_flags = {'DO_NOT_USE': 1,
              'SATURATED': 2,
              'JUMP_DET': 4}
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview
pscale = {'NIRCAM_SHORT': 31.,
          'NIRCAM_LONG': 63.,
          'NIRISS': 65.6}  # mas
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview/niriss-detector-performance
gain = {'NIRCAM_SHORT': 2.05,
        'NIRCAM_LONG': 1.82,
        'NIRISS': 1.61}  # e-/ADU


# =============================================================================
# CLASSES
# =============================================================================

class KPI3Pipeline():
    """
    JWST stage 3 pipeline for kernel-phase imaging.

    ..Notes:: AMI skips ipc, photom, and resample steps in stage 1 & 2
              pipelines. Kernel-phase should also skip these steps.
    """

    def __init__(self):
        """
        Initialize the pipeline.
        """

        # Initialize the pipeline steps.
        self.fix_bad_pixels = fix_bad_pixels()
        self.recenter_frames = recenter_frames()
        self.window_frames = window_frames()
        self.extract_kerphase = extract_kerphase()
        self.empirical_uncertainties = empirical_uncertainties()

        # Initialize the pipeline parameters.
        self.output_dir = None
        self.show_plots = False

    def run(self,
            file):
        """
        Run the pipeline.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        """

        # Make the output directory if it does not exist.
        if (self.output_dir is not None):
            if (not os.path.exists(self.output_dir)):
                os.makedirs(self.output_dir)

        # Run the pipeline steps if they are not skipped. For the kernel-phase
        # extraction, run the re-centering and the windowing internally in
        # complex visibility space.
        suffix = ''
        if not self.fix_bad_pixels.skip:
            suffix = self.fix_bad_pixels.step(file,
                                              suffix,
                                              self.output_dir,
                                              show_plots=self.show_plots)
        if not self.extract_kerphase.skip:
            suffix = self.extract_kerphase.step(file,
                                                suffix,
                                                self.output_dir,
                                                self.recenter_frames,
                                                self.window_frames,
                                                show_plots=self.show_plots)
        else:
            if not self.recenter_frames.skip:
                suffix = self.recenter_frames.step(file,
                                                   suffix,
                                                   self.output_dir,
                                                   show_plots=self.show_plots)
            if not self.window_frames.skip:
                suffix = self.window_frames.step(file,
                                                 suffix,
                                                 self.output_dir,
                                                 show_plots=self.show_plots)
        if not self.empirical_uncertainties.skip:
            suffix = self.empirical_uncertainties.step(file,
                                                       suffix,
                                                       self.output_dir,
                                                       show_plots=self.show_plots)


class fix_bad_pixels():
    """
    Fix bad pixels.

    References for the KI method:
        https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
        https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.bad_bits = ['DO_NOT_USE']
        self.bad_bits_allowed = pxdq_flags.keys()
        self.method = 'medfilt'
        self.method_allowed = ['medfilt', 'KI']

    def step(self,
             file,
             suffix,
             output_dir,
             show_plots=False):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory.
        """

        print('--> Running fix bad pixels step...')

        # Open FITS file.
        if (suffix != ''):
            raise ValueError('Requires pipeline output')
        hdul = ut.open_fits(file)
        data = hdul['SCI'].data
        erro = hdul['ERR'].data
        pxdq = hdul['DQ'].data
        if (data.ndim not in [2, 3]):
            raise UserWarning('Only implemented for 2D image/3D data cube')
        sy, sx = data.shape[-2:]

        # Suffix for the file path from the current step.
        suffix_out = '_bpfixed'

        # Make bad pixel map.
        mask = pxdq < 0
        for i in range(len(self.bad_bits)):
            if (self.bad_bits[i] not in self.bad_bits_allowed):
                raise UserWarning('Unknown data quality flag')
            else:
                value = pxdq_flags[self.bad_bits[i]]
                mask = mask | (pxdq & value == value)
                if (i == 0):
                    bb = self.bad_bits[i]
                else:
                    bb += ', '+self.bad_bits[i]

        print('Found %.0f bad pixels (%.2f%%)' % (np.sum(mask), np.sum(mask)/np.prod(mask.shape)*100.))

        # Fix bad pixels.
        data_bpfixed = data.copy()
        erro_bpfixed = erro.copy()
        if (self.method not in self.method_allowed):
            raise UserWarning('Unknown bad pixel cleaning method')
        else:
            if (self.method == 'medfilt'):
                if (data.ndim == 2):
                    data_bpfixed[mask] = median_filter(data_bpfixed, size=5)[mask]
                    erro_bpfixed[mask] = median_filter(erro_bpfixed, size=5)[mask]
                else:
                    for i in range(data.shape[0]):
                        data_bpfixed[i][mask[i]] = median_filter(data_bpfixed[i], size=5)[mask[i]]
                        erro_bpfixed[i][mask[i]] = median_filter(erro_bpfixed[i], size=5)[mask[i]]
            elif (self.method == 'KI'):
                raise UserWarning('Not implemented yet')

        # Find output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(1, 3, figsize=(2.25*6.4, 0.75*4.8))
            if (data.ndim == 2):
                p0 = ax[0].imshow(mask, origin='lower')
            else:
                p0 = ax[0].imshow(mask[0], origin='lower')
            plt.colorbar(p0, ax=ax[0])
            t0 = ax[0].text(0.01, 0.01, bb, color='white', ha='left', va='bottom', transform=ax[0].transAxes, size=12)
            t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[0].set_title('Bad pixel map', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                p1 = ax[1].imshow(np.log10(np.abs(data)), origin='lower')
            else:
                p1 = ax[1].imshow(np.log10(np.abs(data[0])), origin='lower')
            plt.colorbar(p1, ax=ax[1])
            ax[1].set_title('Full frame (log-scale)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                p2 = ax[2].imshow(np.log10(np.abs(data_bpfixed)), origin='lower')
            else:
                p2 = ax[2].imshow(np.log10(np.abs(data_bpfixed[0])), origin='lower')
            plt.colorbar(p2, ax=ax[2])
            t2 = ax[2].text(0.01, 0.01, 'method = '+self.method, color='white', ha='left', va='bottom', transform=ax[2].transAxes, size=12)
            t2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[2].set_title('Full frame (log-scale, fixed)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Fix bad pixels step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if show_plots:
                plt.show()
            plt.close()

        # Save FITS file.
        hdu_sci_mod = pyfits.ImageHDU(data_bpfixed)
        hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
        hdu_sci_mod.header['METHOD'] = self.method
        hdu_err_mod = pyfits.ImageHDU(erro_bpfixed)
        hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
        hdu_err_mod.header['METHOD'] = self.method
        hdu_dq_mod = pyfits.ImageHDU(mask.astype('uint32'))
        hdu_dq_mod.header['EXTNAME'] = 'DQ-MOD'
        hdu_dq_mod.header['BAD_BITS'] = bb
        hdul += [hdu_sci_mod, hdu_err_mod, hdu_dq_mod]
        hdul.writeto(path+suffix_out+'.fits', output_verify='fix', overwrite=True)
        hdul.close()

        print('Done')

        return suffix_out


class recenter_frames():
    """
    Re-center the individual frames.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.method = 'FPNM'
        self.method_allowed = ['BCEN', 'COGI', 'FPNM']
        self.instrume_allowed = ['NIRCAM', 'NIRISS']
        self.crop = False
        self.bmax = 6.  # m

    def step(self,
             file,
             suffix,
             output_dir,
             show_plots=False):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory.
        """

        print('--> Running recenter frames step...')

        # Open FITS file.
        hdul = ut.open_fits(file, suffix=suffix, dirpath=output_dir)
        data, erro, sx, sy = ut.get_data(hdul)

        INSTRUME = hdul[0].header['INSTRUME']
        FILTER = hdul[0].header['FILTER']

        if self.crop:
            if INSTRUME != "NIRISS":
                raise NotImplementedError("Cropping is implemented for NIRISS only.")
            # Copied from NIRISS bad pixel correction script
            # TODO: Handle 2d data
            ww_max = []
            for j in range(data.shape[0]):
                ww_max += [
                    np.unravel_index(
                        np.argmax(median_filter(data[j], size=3)), data[j].shape
                    )
                ]
            ww_max = np.array(ww_max)
            # the bottom 4 rows are reference pixels
            yh = min(sy - np.max(ww_max[:, 0]), np.min(ww_max[:, 0]) - 4)
            xh = min(sx - np.max(ww_max[:, 1]), np.min(ww_max[:, 1]) - 0)
            sh = min(xh, yh)

            print('Cutting all frames to %.0fx%.0f pixels' % (2 * sh, 2 * sh))
            data = data[:, ww_max[j, 0] - sh:ww_max[j, 0] + sh, ww_max[j, 1] - sh:ww_max[j, 1] + sh].copy()
            erro = erro[:, ww_max[j, 0] - sh:ww_max[j, 0] + sh, ww_max[j, 1] - sh:ww_max[j, 1] + sh].copy()
            sy, sx = data.shape[-2:]

        # PSCALE = np.sqrt(hdul['SCI'].header['PIXAR_A2'])*1000. # mas
        if (INSTRUME == 'NIRCAM'):
            CHANNEL = hdul[0].header['CHANNEL']
            PSCALE = pscale[INSTRUME+'_'+CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = -hdul['SCI'].header['V3I_YANG']*hdul['SCI'].header['VPARITY']  # deg, counter-clockwise
        sh = 10  # pix

        # Suffix for the file path from the current step.
        suffix_out = '_recentered'

        if ('FPNM' in self.method):

            # Check if instrument and filter are known.
            if (INSTRUME not in self.instrume_allowed):
                raise UserWarning('Unknown instrument')
            else:
                if (INSTRUME == 'NIRCAM'):
                    filter_allowed = wave_nircam.keys()
                elif (INSTRUME == 'NIRISS'):
                    filter_allowed = wave_niriss.keys()
            if (FILTER not in filter_allowed):
                raise UserWarning('Unknown filter')

            # Get pupil model path and filter effective wavelength and width.
            if (INSTRUME == 'NIRCAM'):
                fname = os.path.join(sys.prefix, "xara_jwst_pupils", 'nircam_clear_pupil.fits')
                if not os.path.exists(fname):
                    path = os.path.realpath(__file__)
                    temp = path.rfind('/')
                    fname = path[:temp]+'/../jwst/nircam_clear_pupil.fits'
                wave = wave_nircam[FILTER]*1e-6  # m
                weff = weff_nircam[FILTER]*1e-6  # m
            elif (INSTRUME == 'NIRISS'):
                fname = os.path.join(sys.prefix, "xara_jwst_pupils", 'niriss_clear_pupil.fits')
                if not os.path.exists(fname):
                    path = os.path.realpath(__file__)
                    temp = path.rfind('/')
                    fname = path[:temp]+'/../jwst/nircam_clear_pupil.fits'
                wave = wave_niriss[FILTER]*1e-6  # m
                weff = weff_niriss[FILTER]*1e-6  # m

            print('Rotating pupil model by %.2f deg (counter-clockwise)' % V3I_YANG)

            # # Rotate pupil model.
            # theta = np.deg2rad(V3I_YANG)  # rad
            # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # hdul_pup = pyfits.open(fname)
            # xxc = hdul_pup['APERTURE'].data['XXC']
            # yyc = hdul_pup['APERTURE'].data['YYC']
            # trm = hdul_pup['APERTURE'].data['TRM']
            # hdul_pup.close()
            # txt = ''
            # for i in range(len(trm)):
            #     temp = rot.dot(np.array([xxc[i], yyc[i]]))
            #     txt += '%+.5f %+.5f %.5f\n' % (temp[0], temp[1], trm[i])
            # txtfile = open('pupil_model.txt', 'w')
            # txtfile.write(txt)
            # txtfile.close()

            # Load pupil model.
            KPO = kpo.KPO(fname=fname,
                          array=None,
                          ndgt=5,
                          bmax=self.bmax,
                          ID='')
            m2pix = core.mas2rad(PSCALE)*sx/wave

        else:
            KPO = None
            m2pix = None

        # Re-center.
        if (self.method not in self.method_allowed):
            raise UserWarning('Unknown re-centering method')
        else:

            if (data.ndim == 2):
                data_recentered, dx, dy = core.recenter(data,
                                                        algo=self.method,
                                                        subpix=True,
                                                        between=False,
                                                        verbose=False,
                                                        return_center=True,
                                                        dxdy=None,
                                                        mykpo=KPO,
                                                        m2pix=m2pix,
                                                        bmax=self.bmax)
                erro_recentered = core.recenter(erro,
                                                algo=self.method,
                                                subpix=True,
                                                between=False,
                                                verbose=False,
                                                return_center=False,
                                                dxdy=(dx, dy),
                                                mykpo=KPO,
                                                m2pix=m2pix,
                                                bmax=self.bmax)

                print('Image shift = (%.2f, %.2f)' % (dx, dy))
            else:
                data_recentered = []
                dx = []
                dy = []
                erro_recentered = []
                for i in range(data.shape[0]):
                    temp = core.recenter(data[i],
                                         algo=self.method,
                                         subpix=True,
                                         between=False,
                                         verbose=False,
                                         return_center=True,
                                         dxdy=None,
                                         mykpo=KPO,
                                         m2pix=m2pix,
                                         bmax=self.bmax)
                    data_recentered += [temp[0]]
                    dx += [temp[1]]
                    dy += [temp[2]]
                    temp = core.recenter(erro[i],
                                         algo=self.method,
                                         subpix=True,
                                         between=False,
                                         verbose=False,
                                         return_center=False,
                                         dxdy=(dx[-1], dy[-1]),
                                         mykpo=KPO,
                                         m2pix=m2pix,
                                         bmax=self.bmax)
                    erro_recentered += [temp]

                    print('Image shift = (%.2f, %.2f)' % (dx[-1], dy[-1]))
                data_recentered = np.array(data_recentered)
                erro_recentered = np.array(erro_recentered)

        # Find output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(2, 2, figsize=(1.5*6.4, 1.5*4.8))
            if (data.ndim == 2):
                p00 = ax[0, 0].imshow(data, origin='lower')
            else:
                p00 = ax[0, 0].imshow(data[0], origin='lower')
            plt.colorbar(p00, ax=ax[0, 0])
            if (data.ndim == 2):
                ax[0, 0].axhline(sy//2+dy, color='red')
                ax[0, 0].axvline(sx//2+dx, color='red')
                t00 = ax[0, 0].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2+dx, sy//2+dy), color='white', ha='left', va='bottom', transform=ax[0, 0].transAxes, size=12)
            else:
                ax[0, 0].axhline(sy//2+dy[0], color='red')
                ax[0, 0].axvline(sx//2+dx[0], color='red')
                t00 = ax[0, 0].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2+dx[0], sy//2+dy[0]), color='white', ha='left', va='bottom', transform=ax[0, 0].transAxes, size=12)
            t00.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[0, 0].set_title('Full frame (original)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                p01 = ax[0, 1].imshow(data_recentered, origin='lower')
            else:
                p01 = ax[0, 1].imshow(data_recentered[0], origin='lower')
            plt.colorbar(p01, ax=ax[0, 1])
            ax[0, 1].axhline(sy//2, color='red')
            ax[0, 1].axvline(sx//2, color='red')
            t01 = ax[0, 1].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2, sy//2), color='white', ha='left', va='bottom', transform=ax[0, 1].transAxes, size=12)
            t01.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[0, 1].set_title('Full frame (recentered)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                yl = int(sy//2+dy-sh)
                yu = yl+2*sh+1
                xl = int(sx//2+dx-sh)
                xu = xl+2*sh+1
            else:
                yl = int(sy//2+dy[0]-sh)
                yu = yl+2*sh+1
                xl = int(sx//2+dx[0]-sh)
                xu = xl+2*sh+1
            if (data.ndim == 2):
                p10 = ax[1, 0].imshow(data[yl:yu, xl:xu], origin='lower')
            else:
                p10 = ax[1, 0].imshow(data[0][yl:yu, xl:xu], origin='lower')
            plt.colorbar(p10, ax=ax[1, 0])
            if (data.ndim == 2):
                ax[1, 0].axhline(sy//2+dy-yl, color='red')
                ax[1, 0].axvline(sx//2+dx-xl, color='red')
            else:
                ax[1, 0].axhline(sy//2+dy[0]-yl, color='red')
                ax[1, 0].axvline(sx//2+dx[0]-xl, color='red')
            ax[1, 0].set_title('PSF (original)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            yl = int(sy//2-sh)
            yu = yl+2*sh+1
            xl = int(sx//2-sh)
            xu = xl+2*sh+1
            if (data.ndim == 2):
                p11 = ax[1, 1].imshow(data_recentered[yl:yu, xl:xu], origin='lower')
            else:
                p11 = ax[1, 1].imshow(data_recentered[0][yl:yu, xl:xu], origin='lower')
            plt.colorbar(p11, ax=ax[1, 1])
            ax[1, 1].axhline(sy//2-yl, color='red')
            ax[1, 1].axvline(sx//2-xl, color='red')
            ax[1, 1].set_title('PSF (recentered)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Recenter frames step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if show_plots:
                plt.show()
            plt.close()

        # Save FITS file.
        try:
            hdul['SCI-MOD'].data = data_recentered
            hdul['ERR-MOD'].data = erro_recentered
        except Exception:
            hdu_sci_mod = pyfits.ImageHDU(data_recentered)
            hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
            hdu_err_mod = pyfits.ImageHDU(erro_recentered)
            hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
            hdul += [hdu_sci_mod, hdu_err_mod]
        if (data.ndim == 2):
            xsh = pyfits.Column(name='XSHIFT', format='D', array=np.array([dx]))  # pix
            ysh = pyfits.Column(name='YSHIFT', format='D', array=np.array([dy]))  # pix
        else:
            xsh = pyfits.Column(name='XSHIFT', format='D', array=np.array(dx))  # pix
            ysh = pyfits.Column(name='YSHIFT', format='D', array=np.array(dy))  # pix
        hdu_ims = pyfits.BinTableHDU.from_columns([xsh, ysh])
        hdu_ims.header['EXTNAME'] = 'IMSHIFT'
        hdul += [hdu_ims]
        hdul.writeto(path+suffix_out+'.fits', output_verify='fix', overwrite=True)
        hdul.close()

        print('Done')

        return suffix_out


class window_frames():
    """
    Window the individual frames.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.wrad = None  # pix

    def step(self,
             file,
             suffix,
             output_dir,
             show_plots=False):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory.
        """

        print('--> Running window frames step...')

        # Open FITS file.
        hdul = ut.open_fits(file, suffix=suffix, dirpath=output_dir)
        data, erro, sx, sy = ut.get_data(hdul)

        # Suffix for the file path from the current step.
        suffix_out = '_windowed'

        # Window.
        if (self.wrad is None):
            wrad = min(40, sx//4)  # pix
        else:
            wrad = self.wrad  # pix
        data_windowed = data.copy()
        erro_windowed = erro.copy()
        sgmask = core.super_gauss(sy,
                                  sx,
                                  wrad,
                                  between_pix=False)
        data_windowed *= sgmask
        erro_windowed *= sgmask

        # Find output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(1, 2, figsize=(1.5*6.4, 0.75*4.8))
            if (data.ndim == 2):
                vmin = np.min(np.log10(np.abs(data)))
                vmax = np.max(np.log10(np.abs(data)))
            else:
                vmin = np.min(np.log10(np.abs(data[0])))
                vmax = np.max(np.log10(np.abs(data[0])))
            if (data.ndim == 2):
                p0 = ax[0].imshow(np.log10(np.abs(data)), origin='lower', vmin=vmin, vmax=vmax)
            else:
                p0 = ax[0].imshow(np.log10(np.abs(data[0])), origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(p0, ax=ax[0])
            ax[0].set_title('Full frame (log-scale)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                p1 = ax[1].imshow(np.log10(np.abs(data_windowed)), origin='lower', vmin=vmin, vmax=vmax)
            else:
                p1 = ax[1].imshow(np.log10(np.abs(data_windowed[0])), origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(p1, ax=ax[1])
            t1 = ax[1].text(0.01, 0.01, 'wrad = %.0f pix' % wrad, color='white', ha='left', va='bottom', transform=ax[1].transAxes, size=12)
            t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[1].set_title('Full frame (log-scale, windowed)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Window frames step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if show_plots:
                plt.show()
            plt.close()

        # Save FITS file.
        try:
            hdul['SCI-MOD'].data = data_windowed
            hdul['SCI-MOD'].header['WRAD'] = wrad
            hdul['ERR-MOD'].data = erro_windowed
            hdul['ERR-MOD'].header['WRAD'] = wrad
        except Exception:
            hdu_sci_mod = pyfits.ImageHDU(data_windowed)
            hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
            hdu_sci_mod.header['WRAD'] = wrad
            hdu_err_mod = pyfits.ImageHDU(erro_windowed)
            hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
            hdu_err_mod.header['WRAD'] = wrad
            hdul += [hdu_sci_mod, hdu_err_mod]
        hdu_win = pyfits.ImageHDU(sgmask)
        hdu_win.header['EXTNAME'] = 'WINMASK'
        hdu_win.header['WRAD'] = wrad
        hdul += [hdu_win]
        hdul.writeto(path+suffix_out+'.fits', output_verify='fix', overwrite=True)
        hdul.close()

        print('Done')

        return suffix_out


class extract_kerphase():
    """
    Extract the kernel-phase while re-centering in complex visibility space.

    The KPFITS file structure has been agreed upon by the participants of
    Steph Sallum's masking & kernel-phase hackathon in 2021 and is defined
    here:
        https://docs.google.com/document/d/1iBbcCYiq9J2PpLSr21-xB4AXP8X_6tSszxnHY1VDGXg/edit?usp=sharing
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.instrume_allowed = ['NIRCAM', 'NIRISS']
        self.bmax = None  # m

    def step(self,
             file,
             suffix,
             output_dir,
             recenter_frames_obj,
             window_frames_obj,
             show_plots=False):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory.
        recenter_frames_obj: obj
            Object of recenter_frames class.
        window_frames_obj: obj
            Object of window_frames class.
        """

        print('--> Running extract kerphase step...')

        # Open FITS file.
        hdul = ut.open_fits(file, suffix=suffix, dirpath=output_dir)
        data, erro, sx, sy = ut.get_data(hdul)

        INSTRUME = hdul[0].header['INSTRUME']
        FILTER = hdul[0].header['FILTER']

        if recenter_frames_obj.crop:
            if INSTRUME != "NIRISS":
                raise NotImplementedError("Cropping is implemented for NIRISS only.")
            # Copied from NIRISS bad pixel correction script
            # TODO: Handle 2d data
            ww_max = []
            for j in range(data.shape[0]):
                ww_max += [
                    np.unravel_index(
                        np.argmax(median_filter(data[j], size=3)), data[j].shape
                    )
                ]
            ww_max = np.array(ww_max)
            # the bottom 4 rows are reference pixels
            yh = min(sy - np.max(ww_max[:, 0]), np.min(ww_max[:, 0]) - 4)
            xh = min(sx - np.max(ww_max[:, 1]), np.min(ww_max[:, 1]) - 0)
            sh = min(xh, yh)

            print('Cutting all frames to %.0fx%.0f pixels' % (2 * sh, 2 * sh))
            data = data[:, ww_max[j, 0] - sh:ww_max[j, 0] + sh, ww_max[j, 1] - sh:ww_max[j, 1] + sh].copy()
            erro = erro[:, ww_max[j, 0] - sh:ww_max[j, 0] + sh, ww_max[j, 1] - sh:ww_max[j, 1] + sh].copy()
            sy, sx = data.shape[-2:]

        # PSCALE = np.sqrt(hdul['SCI'].header['PIXAR_A2'])*1000. # mas
        if (INSTRUME == 'NIRCAM'):
            CHANNEL = hdul[0].header['CHANNEL']
            PSCALE = pscale[INSTRUME+'_'+CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = -hdul['SCI'].header['V3I_YANG']*hdul['SCI'].header['VPARITY']  # deg, counter-clockwise
        sh = 10  # pix

        # Suffix for the file path from the current step.
        suffix_out = '_kpfits'

        # Check if instrument and filter are known.
        if (INSTRUME not in self.instrume_allowed):
            raise UserWarning('Unknown instrument')
        else:
            if (INSTRUME == 'NIRCAM'):
                filter_allowed = wave_nircam.keys()
            elif (INSTRUME == 'NIRISS'):
                filter_allowed = wave_niriss.keys()
        if (FILTER not in filter_allowed):
            raise UserWarning('Unknown filter')

        # Get pupil model path and filter effective wavelength and width.
        if (INSTRUME == 'NIRCAM'):
            fname = os.path.join(sys.prefix, "xara_jwst_pupils", 'nircam_clear_pupil.fits')
            if not os.path.exists(fname):
                path = os.path.realpath(__file__)
                temp = path.rfind('/')
                fname = path[:temp]+'/../jwst/nircam_clear_pupil.fits'
            wave = wave_nircam[FILTER]*1e-6  # m
            weff = weff_nircam[FILTER]*1e-6  # m
        elif (INSTRUME == 'NIRISS'):
            fname = os.path.join(sys.prefix, "xara_jwst_pupils", 'niriss_clear_pupil.fits')
            if not os.path.exists(fname):
                path = os.path.realpath(__file__)
                temp = path.rfind('/')
                fname = path[:temp]+'/../jwst/niriss_clear_pupil.fits'
            wave = wave_niriss[FILTER]*1e-6  # m
            weff = weff_niriss[FILTER]*1e-6  # m

        print('Rotating pupil model by %.2f deg (counter-clockwise)' % V3I_YANG)

        # # Rotate pupil model.
        # theta = np.deg2rad(V3I_YANG)  # rad
        # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # hdul_pup = pyfits.open(fname)
        # xxc = hdul_pup['APERTURE'].data['XXC']
        # yyc = hdul_pup['APERTURE'].data['YYC']
        # trm = hdul_pup['APERTURE'].data['TRM']
        # hdul_pup.close()
        # txt = ''
        # for i in range(len(trm)):
        #     temp = rot.dot(np.array([xxc[i], yyc[i]]))
        #     txt += '%+.5f %+.5f %.5f\n' % (temp[0], temp[1], trm[i])
        # txtfile = open('pupil_model.txt', 'w')
        # txtfile.write(txt)
        # txtfile.close()

        # Load pupil model.
        KPO = kpo.KPO(fname=fname,
                      array=None,
                      ndgt=5,
                      bmax=self.bmax,
                      ID='')

        # Re-center, window, and extract kernel-phase.
        if not window_frames_obj.skip:
            wrad = window_frames_obj.wrad  # pix
            if (wrad is None):
                wrad = min(40, sx//4)  # pix
        else:
            wrad = None  # pix
        if not recenter_frames_obj.skip:
            if (recenter_frames_obj.method not in recenter_frames_obj.method_allowed):
                raise UserWarning('Unknown re-centering method')
            else:
                if (data.ndim == 2):
                    dx, dy = KPO.extract_KPD_single_frame(data,
                                                          PSCALE,
                                                          wave,
                                                          target=None,
                                                          recenter=True,
                                                          wrad=wrad,
                                                          method='LDFT1',
                                                          algo_cent=recenter_frames_obj.method,
                                                          bmax_cent=recenter_frames_obj.bmax)
                else:
                    dx = []
                    dy = []
                    for i in range(data.shape[0]):
                        temp = KPO.extract_KPD_single_frame(data[i],
                                                            PSCALE,
                                                            wave,
                                                            target=None,
                                                            recenter=True,
                                                            wrad=wrad,
                                                            method='LDFT1',
                                                            algo_cent=recenter_frames_obj.method,
                                                            bmax_cent=recenter_frames_obj.bmax)
                        dx += [temp[0]]
                        dy += [temp[1]]
        else:
            if (data.ndim == 2):
                KPO.extract_KPD_single_frame(data,
                                             PSCALE,
                                             wave,
                                             target=None,
                                             recenter=False,
                                             wrad=wrad,
                                             method='LDFT1')
            else:
                for i in range(data.shape[0]):
                    KPO.extract_KPD_single_frame(data[i],
                                                 PSCALE,
                                                 wave,
                                                 target=None,
                                                 recenter=False,
                                                 wrad=wrad,
                                                 method='LDFT1')

        # Find output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        if not recenter_frames_obj.skip:
            if ('FPNM' in recenter_frames_obj.method):
                m2pix = core.mas2rad(PSCALE)*sx/wave
            else:
                m2pix = None

            # Re-center.
            if (data.ndim == 2):
                data_recentered = core.recenter(data,
                                                algo=recenter_frames_obj.method,
                                                subpix=True,
                                                between=False,
                                                verbose=False,
                                                return_center=False,
                                                dxdy=(dx, dy),
                                                mykpo=KPO,
                                                m2pix=m2pix,
                                                bmax=recenter_frames_obj.bmax)
                erro_recentered = core.recenter(erro,
                                                algo=recenter_frames_obj.method,
                                                subpix=True,
                                                between=False,
                                                verbose=False,
                                                return_center=False,
                                                dxdy=(dx, dy),
                                                mykpo=KPO,
                                                m2pix=m2pix,
                                                bmax=recenter_frames_obj.bmax)
            else:
                data_recentered = []
                erro_recentered = []
                for i in range(data.shape[0]):
                    temp = core.recenter(data[i],
                                         algo=recenter_frames_obj.method,
                                         subpix=True,
                                         between=False,
                                         verbose=False,
                                         return_center=False,
                                         dxdy=(dx[i], dy[i]),
                                         mykpo=KPO,
                                         m2pix=m2pix,
                                         bmax=recenter_frames_obj.bmax)
                    data_recentered += [temp]
                    temp = core.recenter(erro[i],
                                         algo=recenter_frames_obj.method,
                                         subpix=True,
                                         between=False,
                                         verbose=False,
                                         return_center=False,
                                         dxdy=(dx[i], dy[i]),
                                         mykpo=KPO,
                                         m2pix=m2pix,
                                         bmax=recenter_frames_obj.bmax)
                    erro_recentered += [temp]

                    print('Image shift = (%.2f, %.2f)' % (dx[i], dy[i]))
                data_recentered = np.array(data_recentered)
                erro_recentered = np.array(erro_recentered)

            # Plot.
            if recenter_frames_obj.plot:
                suffix_tmp = '_recentered'
                plt.ioff()
                f, ax = plt.subplots(2, 2, figsize=(1.5*6.4, 1.5*4.8))
                if (data.ndim == 2):
                    p00 = ax[0, 0].imshow(data, origin='lower')
                else:
                    p00 = ax[0, 0].imshow(data[0], origin='lower')
                plt.colorbar(p00, ax=ax[0, 0])
                if (data.ndim == 2):
                    ax[0, 0].axhline(sy//2+dy, color='red')
                    ax[0, 0].axvline(sx//2+dx, color='red')
                    t00 = ax[0, 0].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2+dx, sy//2+dy), color='white', ha='left', va='bottom', transform=ax[0, 0].transAxes, size=12)
                else:
                    ax[0, 0].axhline(sy//2+dy[0], color='red')
                    ax[0, 0].axvline(sx//2+dx[0], color='red')
                    t00 = ax[0, 0].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2+dx[0], sy//2+dy[0]), color='white', ha='left', va='bottom', transform=ax[0, 0].transAxes, size=12)
                t00.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                ax[0, 0].set_title('Full frame (original)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
                if (data.ndim == 2):
                    p01 = ax[0, 1].imshow(data_recentered, origin='lower')
                else:
                    p01 = ax[0, 1].imshow(data_recentered[0], origin='lower')
                plt.colorbar(p01, ax=ax[0, 1])
                ax[0, 1].axhline(sy//2, color='red')
                ax[0, 1].axvline(sx//2, color='red')
                t01 = ax[0, 1].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2, sy//2), color='white', ha='left', va='bottom', transform=ax[0, 1].transAxes, size=12)
                t01.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                ax[0, 1].set_title('Full frame (recentered)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
                if (data.ndim == 2):
                    yl = int(sy//2+dy-sh)
                    yu = yl+2*sh+1
                    xl = int(sx//2+dx-sh)
                    xu = xl+2*sh+1
                else:
                    yl = int(sy//2+dy[0]-sh)
                    yu = yl+2*sh+1
                    xl = int(sx//2+dx[0]-sh)
                    xu = xl+2*sh+1
                if (data.ndim == 2):
                    p10 = ax[1, 0].imshow(data[yl:yu, xl:xu], origin='lower')
                else:
                    p10 = ax[1, 0].imshow(data[0][yl:yu, xl:xu], origin='lower')
                plt.colorbar(p10, ax=ax[1, 0])
                if (data.ndim == 2):
                    ax[1, 0].axhline(sy//2+dy-yl, color='red')
                    ax[1, 0].axvline(sx//2+dx-xl, color='red')
                else:
                    ax[1, 0].axhline(sy//2+dy[0]-yl, color='red')
                    ax[1, 0].axvline(sx//2+dx[0]-xl, color='red')
                ax[1, 0].set_title('PSF (original)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
                yl = int(sy//2-sh)
                yu = yl+2*sh+1
                xl = int(sx//2-sh)
                xu = xl+2*sh+1
                if (data.ndim == 2):
                    p11 = ax[1, 1].imshow(data_recentered[yl:yu, xl:xu], origin='lower')
                else:
                    p11 = ax[1, 1].imshow(data_recentered[0][yl:yu, xl:xu], origin='lower')
                plt.colorbar(p11, ax=ax[1, 1])
                ax[1, 1].axhline(sy//2-yl, color='red')
                ax[1, 1].axvline(sx//2-xl, color='red')
                ax[1, 1].set_title('PSF (recentered)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
                plt.suptitle('Recenter frames step', size=18)
                plt.tight_layout()
                plt.savefig(path+suffix_tmp+'.pdf')
                if show_plots:
                    plt.show()
                plt.close()

            # Save FITS file.
            try:
                hdul['SCI-MOD'].data = data_recentered
                hdul['ERR-MOD'].data = erro_recentered
            except Exception:
                hdu_sci_mod = pyfits.ImageHDU(data_recentered)
                hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
                hdu_err_mod = pyfits.ImageHDU(erro_recentered)
                hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
                hdul += [hdu_sci_mod, hdu_err_mod]
            if (data.ndim == 2):
                xsh = pyfits.Column(name='XSHIFT', format='D', array=np.array([dx]))  # pix
                ysh = pyfits.Column(name='YSHIFT', format='D', array=np.array([dy]))  # pix
            else:
                xsh = pyfits.Column(name='XSHIFT', format='D', array=np.array(dx))  # pix
                ysh = pyfits.Column(name='YSHIFT', format='D', array=np.array(dy))  # pix
            hdu_ims = pyfits.BinTableHDU.from_columns([xsh, ysh])
            hdu_ims.header['EXTNAME'] = 'IMSHIFT'
            hdul += [hdu_ims]

        else:
            data_recentered = data.copy()
            erro_recentered = erro.copy()

        if not window_frames_obj.skip:

            # Window.
            data_windowed = data_recentered.copy()
            erro_windowed = erro_recentered.copy()
            sgmask = core.super_gauss(sy,
                                      sx,
                                      wrad,
                                      between_pix=False)
            data_windowed *= sgmask
            erro_windowed *= sgmask

            # Plot.
            if window_frames_obj.plot:
                suffix_tmp = '_windowed'
                plt.ioff()
                f, ax = plt.subplots(1, 2, figsize=(1.5*6.4, 0.75*4.8))
                if (data.ndim == 2):
                    vmin = np.min(np.log10(np.abs(data_recentered)))
                    vmax = np.max(np.log10(np.abs(data_recentered)))
                else:
                    vmin = np.min(np.log10(np.abs(data_recentered[0])))
                    vmax = np.max(np.log10(np.abs(data_recentered[0])))
                if (data.ndim == 2):
                    p0 = ax[0].imshow(np.log10(np.abs(data_recentered)), origin='lower', vmin=vmin, vmax=vmax)
                else:
                    p0 = ax[0].imshow(np.log10(np.abs(data_recentered[0])), origin='lower', vmin=vmin, vmax=vmax)
                plt.colorbar(p0, ax=ax[0])
                ax[0].set_title('Full frame (log-scale)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
                if (data.ndim == 2):
                    p1 = ax[1].imshow(np.log10(np.abs(data_windowed)), origin='lower', vmin=vmin, vmax=vmax)
                else:
                    p1 = ax[1].imshow(np.log10(np.abs(data_windowed[0])), origin='lower', vmin=vmin, vmax=vmax)
                plt.colorbar(p1, ax=ax[1])
                t1 = ax[1].text(0.01, 0.01, 'wrad = %.0f pix' % wrad, color='white', ha='left', va='bottom', transform=ax[1].transAxes, size=12)
                t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                ax[1].set_title('Full frame (log-scale, windowed)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
                plt.suptitle('Window frames step', size=18)
                plt.tight_layout()
                plt.savefig(path+suffix_tmp+'.pdf')
                if show_plots:
                    plt.show()
                plt.close()

            # Save FITS file.
            try:
                hdul['SCI-MOD'].data = data_windowed
                hdul['SCI-MOD'].header['WRAD'] = wrad
                hdul['ERR-MOD'].data = erro_windowed
                hdul['ERR-MOD'].header['WRAD'] = wrad
            except Exception:
                hdu_sci_mod = pyfits.ImageHDU(data_windowed)
                hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
                hdu_sci_mod.header['WRAD'] = wrad
                hdu_err_mod = pyfits.ImageHDU(erro_windowed)
                hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
                hdu_err_mod.header['WRAD'] = wrad
                hdul += [hdu_sci_mod, hdu_err_mod]
            hdu_win = pyfits.ImageHDU(sgmask)
            hdu_win.header['EXTNAME'] = 'WINMASK'
            hdu_win.header['WRAD'] = wrad
            hdul += [hdu_win]

        else:
            data_windowed = data_recentered.copy()
            erro_windowed = erro_recentered.copy()

        # Extract kernel-phase covariance.
        if (data.ndim == 2):
            frame = data_windowed.copy()
            varframe = erro_windowed.copy()**2
            B = KPO.kpi.KPM.dot(np.divide(KPO.FF.imag.T, np.abs(KPO.FF.dot(frame.flatten()))).T)
            kpcov = np.multiply(B, varframe.flatten()).dot(B.T)
            kpsig = np.sqrt(np.diag(kpcov))
            kpcor = np.true_divide(kpcov, kpsig[:, None]*kpsig[None, :])
        else:
            kpcov = []
            kpsig = []
            kpcor = []
            for i in range(data.shape[0]):
                frame = data_windowed[i].copy()
                varframe = erro_windowed[i].copy()**2
                B = KPO.kpi.KPM.dot(np.divide(KPO.FF.imag.T, np.abs(KPO.FF.dot(frame.flatten()))).T)
                kpcov += [np.multiply(B, varframe.flatten()).dot(B.T)]
                kpsig += [np.sqrt(np.diag(kpcov[-1]))]
                kpcor += [np.true_divide(kpcov[-1], kpsig[-1][:, None]*kpsig[-1][None, :])]
            kpcov = np.array(kpcov)
            kpsig = np.array(kpsig)
            kpcor = np.array(kpcor)

        # Plot.
        if self.plot:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.ioff()
            f, ax = plt.subplots(2, 2, figsize=(1.5*6.4, 1.5*4.8))
            if (data.ndim == 2):
                d00 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data_windowed)))
            else:
                d00 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data_windowed[0])))
            p00 = ax[0, 0].imshow(np.angle(d00), origin='lower', vmin=-np.pi, vmax=np.pi)
            c00 = plt.colorbar(p00, ax=ax[0, 0])
            c00.set_label('Fourier phase [rad]', rotation=270, labelpad=20)
            m2pix = core.mas2rad(PSCALE)*sx/wave
            xx = KPO.kpi.UVC[:, 0]*m2pix+sx//2
            yy = KPO.kpi.UVC[:, 1]*m2pix+sy//2
            ax[0, 0].scatter(xx, yy, s=0.2, c='red')
            xx = -KPO.kpi.UVC[:, 0]*m2pix+sx//2
            yy = -KPO.kpi.UVC[:, 1]*m2pix+sy//2
            ax[0, 0].scatter(xx, yy, s=0.2, c='red')
            ax[0, 0].set_title('Fourier phase', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                p01 = ax[0, 1].imshow(kpcor, origin='lower', cmap='RdBu', vmin=-1., vmax=1.)
            else:
                p01 = ax[0, 1].imshow(kpcor[0], origin='lower', cmap='RdBu', vmin=-1., vmax=1.)
            c01 = plt.colorbar(p01, ax=ax[0, 1])
            c01.set_label('Correlation', rotation=270, labelpad=20)
            ax[0, 1].set_title('Kernel-phase correlation', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            ww = np.argsort(KPO.kpi.BLEN)
            ax[1, 0].plot(np.angle(KPO.CVIS[0][0, ww]))
            ax[1, 0].axhline(0., ls='--', color='black')
            ax[1, 0].set_ylim([-np.pi, np.pi])
            ax[1, 0].grid(axis='y')
            ax[1, 0].set_xlabel('Index sorted by baseline length')
            ax[1, 0].set_ylabel('Fourier phase [rad]')
            ax[1, 0].set_title('Fourier phase', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            if (data.ndim == 2):
                ax[1, 1].errorbar(np.arange(KPO.KPDT[0][0, :].shape[0]), KPO.KPDT[0][0, :], yerr=kpsig, color=colors[0], alpha=1./3.)
            else:
                ax[1, 1].errorbar(np.arange(KPO.KPDT[0][0, :].shape[0]), KPO.KPDT[0][0, :], yerr=kpsig[0], color=colors[0], alpha=1./3.)
            ax[1, 1].plot(np.arange(KPO.KPDT[0][0, :].shape[0]), KPO.KPDT[0][0, :], color=colors[0])
            ax[1, 1].axhline(0., ls='--', color='black')
            ylim = ax[1, 1].get_ylim()
            ax[1, 1].set_ylim([-np.max(ylim), np.max(ylim)])
            ax[1, 1].grid(axis='y')
            ax[1, 1].set_xlabel('Index')
            ax[1, 1].set_ylabel('Kernel-phase [rad]')
            ax[1, 1].set_title('Kernel-phase', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Extract kerphase step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if show_plots:
                plt.show()
            plt.close()

        # Save FITS file.
        hdul[0].header['PSCALE'] = PSCALE  # mas
        if ('NIRCAM' in hdul[0].header['INSTRUME']):
            hdul[0].header['GAIN'] = gain[hdul[0].header['INSTRUME']+'_'+hdul[0].header['CHANNEL']]  # e-/ADU
        else:
            hdul[0].header['GAIN'] = gain[hdul[0].header['INSTRUME']]  # e-/ADU
        hdul[0].header['DIAM'] = 6.559348  # m (flat-to-flat)
        hdul[0].header['EXPTIME'] = hdul[0].header['EFFINTTM']  # s
        hdul[0].header['DATEOBS'] = hdul[0].header['DATE-OBS']+'T'+hdul[0].header['TIME-OBS']  # YYYY-MM-DDTHH:MM:SS.MMM
        hdul[0].header['PROCSOFT'] = 'CALWEBB_KPI3'
        try:
            hdul[0].header['WRAD'] = hdul['WINMASK'].header['WRAD']  # pix
        except Exception:
            hdul[0].header['WRAD'] = 'NONE'  # pix
        hdul[0].header['CALFLAG'] = 'False'
        hdul[0].header['CONTENT'] = 'KPFITS1'
        xy1 = pyfits.Column(name='XXC', format='D', array=KPO.kpi.VAC[:, 0])  # m
        xy2 = pyfits.Column(name='YYC', format='D', array=KPO.kpi.VAC[:, 1])  # m
        trm = pyfits.Column(name='TRM', format='D', array=KPO.kpi.TRM)
        hdu_ape = pyfits.BinTableHDU.from_columns([xy1, xy2, trm])
        hdu_ape.header['EXTNAME'] = 'APERTURE'
        hdu_ape.header['TTYPE1'] = ('XXC', 'Virtual aperture x-coord (meters)')
        hdu_ape.header['TTYPE2'] = ('YYC', 'Virtual aperture y-coord (meters)')
        hdu_ape.header['TTYPE3'] = ('TRM', 'Virtual aperture transmission (0 < t <= 1)')
        hdul += [hdu_ape]
        uv1 = pyfits.Column(name='UUC', format='D', array=KPO.kpi.UVC[:, 0])  # m
        uv2 = pyfits.Column(name='VVC', format='D', array=KPO.kpi.UVC[:, 1])  # m
        red = pyfits.Column(name='RED', format='I', array=KPO.kpi.RED)
        hdu_uvp = pyfits.BinTableHDU.from_columns([uv1, uv2, red])
        hdu_uvp.header['EXTNAME'] = 'UV-PLANE'
        hdu_uvp.header['TTYPE1'] = ('UUC', 'Baseline u coordinate (meters)')
        hdu_uvp.header['TTYPE2'] = ('VVC', 'Baseline v coordinate (meters)')
        hdu_uvp.header['TTYPE3'] = ('RED', 'Baseline redundancy (int)')
        hdul += [hdu_uvp]
        hdu_kpm = pyfits.ImageHDU(KPO.kpi.KPM)
        hdu_kpm.header['EXTNAME'] = 'KER-MAT'
        hdul += [hdu_kpm]
        hdu_blm = pyfits.ImageHDU(np.diag(KPO.kpi.RED).dot(KPO.kpi.TFM))
        hdu_blm.header['EXTNAME'] = 'BLM-MAT'
        hdul += [hdu_blm]
        if (data.ndim == 2):
            hdu_kpd = pyfits.ImageHDU(KPO.KPDT[0][np.newaxis, :])  # rad
        else:
            hdu_kpd = pyfits.ImageHDU(np.array(KPO.KPDT))  # rad
        hdu_kpd.header['EXTNAME'] = 'KP-DATA'
        hdul += [hdu_kpd]
        if (data.ndim == 2):
            hdu_kpe = pyfits.ImageHDU(kpsig[np.newaxis, np.newaxis, :])  # rad
        else:
            hdu_kpe = pyfits.ImageHDU(kpsig[:, np.newaxis, :])  # rad
        hdu_kpe.header['EXTNAME'] = 'KP-SIGM'
        hdul += [hdu_kpe]
        if (data.ndim == 2):
            hdu_kpc = pyfits.ImageHDU(kpcov[np.newaxis, np.newaxis, :])  # rad^2
        else:
            hdu_kpc = pyfits.ImageHDU(kpcov[:, np.newaxis, :])  # rad^2
        hdu_kpc.header['EXTNAME'] = 'KP-COV'
        hdul += [hdu_kpc]
        cwavel = pyfits.Column(name='CWAVEL', format='D', array=np.array([wave]))  # m
        dwavel = pyfits.Column(name='DWAVEL', format='D', array=np.array([weff/2.]))  # m
        hdu_lam = pyfits.BinTableHDU.from_columns([cwavel, dwavel])
        hdu_lam.header['EXTNAME'] = 'CWAVEL'
        hdul += [hdu_lam]
        if (data.ndim == 2):
            hdu_ang = pyfits.ImageHDU(np.array([hdul['SCI'].header['ROLL_REF']]))  # deg
        else:
            hdu_ang = pyfits.ImageHDU(np.array([hdul['SCI'].header['ROLL_REF']]*data.shape[0]))  # deg
        hdu_ang.header['EXTNAME'] = 'DETPA'
        hdul += [hdu_ang]
        if (data.ndim == 2):
            hdu_vis = pyfits.ImageHDU(np.vstack((np.real(KPO.CVIS[0])[np.newaxis, np.newaxis, :], np.imag(KPO.CVIS[0])[np.newaxis, np.newaxis, :])))
        else:
            temp = np.zeros((data.shape[0], 2, hdul['KER-MAT'].data.shape[1], hdul['KER-MAT'].data.shape[1]))
            temp[:, 0, :, :] = np.real(np.array(KPO.CVIS))
            temp[:, 1, :, :] = np.imag(np.array(KPO.CVIS))
            hdu_vis = pyfits.ImageHDU(temp)
        hdu_vis.header['EXTNAME'] = 'CVIS-DATA'
        hdul += [hdu_vis]
        hdul.writeto(path+suffix_out+'.fits', output_verify='fix', overwrite=True)
        hdul.close()

        print('Done')

        return suffix_out


class empirical_uncertainties():
    """
    Compute empirical uncertainties for the kernel-phase.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.get_emp_err = True

    def step(self,
             file,
             suffix,
             output_dir,
             show_plots=False):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory.
        """

        print('--> Running empirical uncertainties step...')

        # Open FITS file.
        hdul = ut.open_fits(file, suffix=suffix, dirpath=output_dir)
        kpdat = hdul['KP-DATA'].data
        kpsig = hdul['KP-SIGM'].data
        kpcov = hdul['KP-COV'].data
        if ((kpdat.ndim != 3) or (kpsig.ndim != 3) or (kpcov.ndim != 4)):
            raise UserWarning('Input is not a valid KPFITS file')
        if (kpdat.shape[0] < 3):
            print('Not enough frames to estimate uncertainties empirically, skipping step!')
            return suffix

        # Suffix for the file path from the current step.
        suffix_out = '_emp_kpfits'

        #
        nframe = kpdat.shape[0]
        nwave = kpdat.shape[1]
        nkp = kpdat.shape[2]
        wmdat = np.zeros((1, nwave, nkp))
        wmsig = np.zeros((1, nwave, nkp))
        wmcov = np.zeros((1, nwave, nkp, nkp))
        emsig = np.zeros((1, nwave, nkp))  # empirical uncertainties
        emcov = np.zeros((1, nwave, nkp, nkp))  # empirical uncertainties
        for i in range(nwave):
            invcov = []
            invcovdat = []
            for j in range(nframe):
                invcov += [np.linalg.inv(kpcov[j, i])]
                invcovdat += [invcov[-1].dot(kpdat[j, i])]
            wmcov[0, i] = np.linalg.inv(np.sum(np.array(invcov), axis=0))
            wmsig[0, i] = np.sqrt(np.diag(wmcov[0, i]))
            wmdat[0, i] = wmcov[0, i].dot(np.sum(np.array(invcovdat), axis=0))
            emsig[0, i] = np.std(kpdat[:, i], axis=0)
            wmcor = np.true_divide(wmcov[0, i], wmsig[0, i][:, None]*wmsig[0, i][None, :])
            emcov[0, i] = np.multiply(wmcor, emsig[0, i][:, None]*emsig[0, i][None, :])

        # Find output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.ioff()
            f = plt.figure()
            ax = plt.gca()
            ax.fill_between(np.arange(nkp), np.min(kpdat[:, 0], axis=0), np.max(kpdat[:, 0], axis=0), ec='None', fc=colors[0], alpha=1./3., label='Min/max range')
            if self.get_emp_err:
                ax.errorbar(np.arange(nkp), wmdat[0, 0], yerr=emsig[0, 0], color=colors[0], label='Weighted mean')
            else:
                ax.errorbar(np.arange(nkp), wmdat[0, 0], yerr=wmsig[0, 0], color=colors[0], label='Weighted mean')
            ax.axhline(0., ls='--', color='black')
            ylim = ax.get_ylim()
            ax.set_ylim([-np.max(ylim), np.max(ylim)])
            ax.grid(axis='y')
            ax.set_xlabel('Index')
            ax.set_ylabel('Kernel-phase [rad]')
            ax.legend(loc='upper right')
            plt.suptitle('Empirical uncertainties step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if show_plots:
                plt.show()
            plt.close()

        # Save FITS file.
        hdul['KP-DATA'].data = wmdat
        if self.get_emp_err:
            hdul['KP-SIGM'].data = emsig
            hdul['KP-COV'].data = emcov
        else:
            hdul['KP-SIGM'].data = wmsig
            hdul['KP-COV'].data = wmcov
        hdul.writeto(path+suffix_out+'.fits', output_verify='fix', overwrite=True)
        hdul.close()

        print('Done')

        return suffix_out
