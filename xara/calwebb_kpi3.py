from __future__ import division
"""
JWST stage 3 pipeline for kernel-phase imaging.

Authors: Jens Kammerer
Supported instruments: NIRCam, NIRISS
"""

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import matplotlib.patheffects as PathEffects

from scipy.ndimage import median_filter

import core
import kpi
import kpo

show_plots = True

# http://svo2.cab.inta-csic.es/theory/fps/
wave_nircam = {'F212N': 2.121193} # micron
wave_niriss = {'F277W': 2.739519,
               'F380M': 3.826384,
               'F430M': 4.282976,
               'F480M': 4.813019} # micron
# https://jwst-reffiles.stsci.edu/source/data_quality.html
pxdq_flags = {'DO_NOT_USE': 1,
              'SATURATED': 2,
              'JUMP_DET': 4}


# =============================================================================
# CLASSES
# =============================================================================

class KPI3Pipeline():
    """
    JWST stage 3 pipeline for kernel-phase imaging.
    
    ..Notes:: NRM skips ipc, photom, and resample steps in stage 1 & 2
              pipelines. Kernel-phase should also skip these steps.
    """
    
    def __init__(self):
        """
        """
        
        self.fix_bad_pixels = fix_bad_pixels()
        self.recenter_frames = recenter_frames()
        self.window_frames = window_frames()
        self.extract_kerphase = extract_kerphase()
        self.empirical_uncertainties = empirical_uncertainties()
        
        self.output_dir = None
        
        pass
    
    def run(self,
            file):
        """
        Function to run the pipeline.
        """
        
        if (self.output_dir is not None):
            if (not os.path.exists(self.output_dir)):
                os.makedirs(self.output_dir)
        
        suffix = ''
        if (self.fix_bad_pixels.skip == False):
            suffix = self.fix_bad_pixels.step(file,
                                              suffix,
                                              self.output_dir)
        if (self.recenter_frames.skip == False):
            suffix = self.recenter_frames.step(file,
                                               suffix,
                                               self.output_dir)
        if (self.window_frames.skip == False):
            suffix = self.window_frames.step(file,
                                             suffix,
                                             self.output_dir)
        if (self.extract_kerphase.skip == False):
            suffix = self.extract_kerphase.step(file,
                                                suffix,
                                                self.output_dir)
        if (self.empirical_uncertainties.skip == False):
            suffix = self.empirical_uncertainties.step(file,
                                                       suffix,
                                                       self.output_dir)
        
        pass

class fix_bad_pixels():
    """
    Fix bad pixels using the Kammerer & Ireland method.
    
    The corresponding references are https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
    and https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        self.plot = True
        self.bad_bits = ['SATURATED', 'JUMP_DET']
        self.bad_bits_allowed = pxdq_flags.keys()
        self.method = 'medfilt'
        self.method_allowed = ['medfilt', 'KI']
        
        pass
    
    def step(self,
             file,
             suffix,
             output_dir):
        """
        Function to run the step.
        """
        
        print('--> Running fix bad pixels step...')
        
        hdul = pyfits.open(file[:-5]+suffix+'.fits')
        if (suffix != ''):
            raise UserWarning('Requires pipeline output')
        data = hdul['SCI'].data
        erro = hdul['ERR'].data
        pxdq = hdul['DQ'].data
        if (data.ndim != 2):
            raise UserWarning('Only implemented for 2D image')
        sy, sx = data.shape
        suffix_out = '_bpfixed'
        
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
        
        data_bpfixed = data.copy()
        erro_bpfixed = erro.copy()
        if (self.method not in self.method_allowed):
            raise UserWarning('Unknown bad pixel cleaning method')
        else:
            if (self.method == 'medfilt'):
                data_bpfixed[mask] = median_filter(data_bpfixed, size=5)[mask]
                erro_bpfixed[mask] = median_filter(erro_bpfixed, size=5)[mask]
            elif (self.method == 'KI'):
                raise UserWarning('Not implemented yet')
        
        if (output_dir is None):
            path = file[:-5]
        else:
            temp = file.rfind('/')
            if (temp == -1):
                path = output_dir+file[:-5]
            else:
                path = output_dir+file[temp+1:-5]
        
        if (self.plot == True):
            f, ax = plt.subplots(1, 3, figsize=(2.25*6.4, 0.75*4.8))
            p0 = ax[0].imshow(mask, origin='lower')
            plt.colorbar(p0, ax=ax[0])
            t0 = ax[0].text(0.01, 0.01, bb, color='white', ha='left', va='bottom', transform=ax[0].transAxes, size=12)
            t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[0].set_title('Bad pixel map', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            p1 = ax[1].imshow(np.log10(np.abs(data)), origin='lower')
            plt.colorbar(p1, ax=ax[1])
            ax[1].set_title('Full frame (log-scale)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            p2 = ax[2].imshow(np.log10(np.abs(data_bpfixed)), origin='lower')
            plt.colorbar(p2, ax=ax[2])
            t2 = ax[2].text(0.01, 0.01, 'method = '+self.method, color='white', ha='left', va='bottom', transform=ax[2].transAxes, size=12)
            t2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[2].set_title('Full frame (log-scale, fixed)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Fix bad pixels step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if (show_plots == True):
                plt.show()
            plt.close()
        
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
        """
        
        self.skip = False
        self.plot = True
        
        pass
    
    def step(self,
             file,
             suffix,
             output_dir):
        """
        Function to run the step.
        """
        
        print('--> Running recenter frames step...')
        
        if (suffix == ''):
            hdul = pyfits.open(file[:-5]+suffix+'.fits')
        else:
            if (output_dir is None):
                hdul = pyfits.open(file[:-5]+suffix+'.fits')
            else:
                temp = file.rfind('/')
                if (temp == -1):
                    hdul = pyfits.open(output_dir+file[:-5]+suffix+'.fits')
                else:
                    hdul = pyfits.open(output_dir+file[temp+1:-5]+suffix+'.fits')
        try:
            data = hdul['SCI-MOD'].data
            erro = hdul['ERR-MOD'].data
        except:
            data = hdul['SCI'].data
            erro = hdul['ERR'].data
        if (data.ndim != 2):
            raise UserWarning('Only implemented for 2D image')
        sy, sx = data.shape
        sh = 10 # pix
        suffix_out = '_recentered'
        
        data_recentered, dx, dy = core.recenter(data,
                                                algo='BCEN',
                                                subpix=True,
                                                between=False,
                                                verbose=False,
                                                return_center=True,
                                                centroid=None)
        erro_recentered = core.recenter(erro,
                                        algo='BCEN',
                                        subpix=True,
                                        between=False,
                                        verbose=False,
                                        return_center=False,
                                        centroid=(dx, dy))
        
        print('PSF shift = (%.2f, %.2f)' % (dx, dy))
        
        if (output_dir is None):
            path = file[:-5]
        else:
            temp = file.rfind('/')
            if (temp == -1):
                path = output_dir+file[:-5]
            else:
                path = output_dir+file[temp+1:-5]
        
        if (self.plot == True):
            f, ax = plt.subplots(2, 2, figsize=(1.5*6.4, 1.5*4.8))
            p00 = ax[0, 0].imshow(data, origin='lower')
            plt.colorbar(p00, ax=ax[0, 0])
            ax[0, 0].axhline(sy//2+dy, color='red')
            ax[0, 0].axvline(sx//2+dx, color='red')
            t00 = ax[0, 0].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2+dx, sy//2+dy), color='white', ha='left', va='bottom', transform=ax[0, 0].transAxes, size=12)
            t00.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[0, 0].set_title('Full frame (original)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            p01 = ax[0, 1].imshow(data_recentered, origin='lower')
            plt.colorbar(p01, ax=ax[0, 1])
            ax[0, 1].axhline(sy//2, color='red')
            ax[0, 1].axvline(sx//2, color='red')
            t01 = ax[0, 1].text(0.01, 0.01, 'center = %.2f, %.2f' % (sx//2, sy//2), color='white', ha='left', va='bottom', transform=ax[0, 1].transAxes, size=12)
            t01.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[0, 1].set_title('Full frame (recentered)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            yl = int(sy//2+dy-sh)
            yu = yl+2*sh+1
            xl = int(sx//2+dx-sh)
            xu = xl+2*sh+1
            p10 = ax[1, 0].imshow(data[yl:yu, xl:xu], origin='lower')
            plt.colorbar(p10, ax=ax[1, 0])
            ax[1, 0].axhline(sy//2+dy-yl, color='red')
            ax[1, 0].axvline(sx//2+dx-xl, color='red')
            ax[1, 0].set_title('PSF (original)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            yl = int(sy//2-sh)
            yu = yl+2*sh+1
            xl = int(sx//2-sh)
            xu = xl+2*sh+1
            p11 = ax[1, 1].imshow(data_recentered[yl:yu, xl:xu], origin='lower')
            plt.colorbar(p11, ax=ax[1, 1])
            ax[1, 1].axhline(sy//2-yl, color='red')
            ax[1, 1].axvline(sx//2-xl, color='red')
            ax[1, 1].set_title('PSF (recentered)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Recenter frames step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if (show_plots == True):
                plt.show()
            plt.close()
        
        try:
            hdul['SCI-MOD'].data = data_recentered
            hdul['SCI-MOD'].header['DX'] = dx
            hdul['SCI-MOD'].header['DY'] = dy
            hdul['ERR-MOD'].data = erro_recentered
            hdul['ERR-MOD'].header['DX'] = dx
            hdul['ERR-MOD'].header['DY'] = dy
        except:
            hdu_sci_mod = pyfits.ImageHDU(data_recentered)
            hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
            hdu_sci_mod.header['DX'] = dx
            hdu_sci_mod.header['DY'] = dy
            hdu_err_mod = pyfits.ImageHDU(erro_recentered)
            hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
            hdu_err_mod.header['DX'] = dx
            hdu_err_mod.header['DY'] = dy
            hdul += [hdu_sci_mod, hdu_err_mod]
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
        """
        
        self.skip = False
        self.plot = True
        self.wrad = 40 # pix
        
        pass
    
    def step(self,
             file,
             suffix,
             output_dir):
        """
        Function to run the step.
        """
        
        print('--> Running window frames step...')
        
        if (suffix == ''):
            hdul = pyfits.open(file[:-5]+suffix+'.fits')
        else:
            if (output_dir is None):
                hdul = pyfits.open(file[:-5]+suffix+'.fits')
            else:
                temp = file.rfind('/')
                if (temp == -1):
                    hdul = pyfits.open(output_dir+file[:-5]+suffix+'.fits')
                else:
                    hdul = pyfits.open(output_dir+file[temp+1:-5]+suffix+'.fits')
        try:
            data = hdul['SCI-MOD'].data
            erro = hdul['ERR-MOD'].data
        except:
            data = hdul['SCI'].data
            erro = hdul['ERR'].data
        if (data.ndim != 2):
            raise UserWarning('Only implemented for 2D image')
        sy, sx = data.shape
        suffix_out = '_windowed'
        
        data_windowed = data.copy()
        erro_windowed = erro.copy()
        sgmask = core.super_gauss(sy,
                                  sx,
                                  self.wrad,
                                  between_pix=False)
        data_windowed *= sgmask
        erro_windowed *= sgmask
        
        if (output_dir is None):
            path = file[:-5]
        else:
            temp = file.rfind('/')
            if (temp == -1):
                path = output_dir+file[:-5]
            else:
                path = output_dir+file[temp+1:-5]
        
        if (self.plot == True):
            f, ax = plt.subplots(1, 2, figsize=(1.5*6.4, 0.75*4.8))
            vmin = np.min(np.log10(np.abs(data)))
            vmax = np.max(np.log10(np.abs(data)))
            p0 = ax[0].imshow(np.log10(np.abs(data)), origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(p0, ax=ax[0])
            ax[0].set_title('Full frame (log-scale)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            p1 = ax[1].imshow(np.log10(np.abs(data_windowed)), origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(p1, ax=ax[1])
            t1 = ax[1].text(0.01, 0.01, 'wrad = %.0f pix' % self.wrad, color='white', ha='left', va='bottom', transform=ax[1].transAxes, size=12)
            t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[1].set_title('Full frame (log-scale, windowed)', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Window frames step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if (show_plots == True):
                plt.show()
            plt.close()
        
        try:
            hdul['SCI-MOD'].data = data_windowed
            hdul['SCI-MOD'].header['WRAD'] = self.wrad
            hdul['ERR-MOD'].data = erro_windowed
            hdul['ERR-MOD'].header['WRAD'] = self.wrad
        except:
            hdu_sci_mod = pyfits.ImageHDU(data_windowed)
            hdu_sci_mod.header['EXTNAME'] = 'SCI-MOD'
            hdu_sci_mod.header['WRAD'] = self.wrad
            hdu_err_mod = pyfits.ImageHDU(erro_windowed)
            hdu_err_mod.header['EXTNAME'] = 'ERR-MOD'
            hdu_err_mod.header['WRAD'] = self.wrad
            hdul += [hdu_sci_mod, hdu_err_mod]
        hdu_win = pyfits.ImageHDU(sgmask)
        hdu_win.header['EXTNAME'] = 'WINMASK'
        hdu_win.header['WRAD'] = self.wrad
        hdul += [hdu_win]
        hdul.writeto(path+suffix_out+'.fits', output_verify='fix', overwrite=True)
        hdul.close()
        
        print('Done')
        
        return suffix_out

class extract_kerphase():
    """
    Extract the kernel-phase.
    
    The FITS file structure has been agreed upon by the participants of Steph
    Sallum's masking & kernel-phase hackathon in 2021 and is defined here:
    https://docs.google.com/document/d/1iBbcCYiq9J2PpLSr21-xB4AXP8X_6tSszxnHY1VDGXg/edit?usp=sharing.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        self.plot = True
        self.instrume_allowed = ['NIRCAM', 'NIRISS']
        self.bmax = None # m
        
        pass
    
    def step(self,
             file,
             suffix,
             output_dir):
        """
        Function to run the step.
        """
        
        print('--> Running extract kerphase step...')
        
        if (suffix == ''):
            hdul = pyfits.open(file[:-5]+suffix+'.fits')
        else:
            if (output_dir is None):
                hdul = pyfits.open(file[:-5]+suffix+'.fits')
            else:
                temp = file.rfind('/')
                if (temp == -1):
                    hdul = pyfits.open(output_dir+file[:-5]+suffix+'.fits')
                else:
                    hdul = pyfits.open(output_dir+file[temp+1:-5]+suffix+'.fits')
        try:
            data = hdul['SCI-MOD'].data
            erro = hdul['ERR-MOD'].data
        except:
            data = hdul['SCI'].data
            erro = hdul['ERR'].data
        if (data.ndim != 2):
            raise UserWarning('Only implemented for 2D image')
        sy, sx = data.shape
        INSTRUME = hdul[0].header['INSTRUME']
        FILTER = hdul[0].header['FILTER']
        PSCALE = np.sqrt(hdul['SCI'].header['PIXAR_A2'])*1000. # mas
        V3I_YANG = hdul['SCI'].header['V3I_YANG'] # deg, counter-clockwise
        suffix_out = '_kpfits'
        
        if (INSTRUME not in self.instrume_allowed):
            raise UserWarning('Unknown instrument')
        else:
            if (INSTRUME == 'NIRCAM'):
                filter_allowed = wave_nircam.keys()
            elif (INSTRUME == 'NIRISS'):
                filter_allowed = wave_niriss.keys()
        if (FILTER not in filter_allowed):
            raise UserWarning('Unknown filter')
        
        if (INSTRUME == 'NIRCAM'):
            path = os.path.realpath(__file__)
            temp = path.rfind('/')
            fname = path[:temp]+'/../jwst/nircam_clear_pupil.fits'
            wave = wave_nircam[FILTER]*1e-6 # m
        elif (INSTRUME == 'NIRISS'):
            path = os.path.realpath(__file__)
            temp = path.rfind('/')
            fname = path[:temp]+'/../jwst/niriss_clear_pupil.fits'
            wave = wave_niriss[FILTER]*1e-6 # m
        
        theta = np.deg2rad(V3I_YANG) # rad
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        hdul_pup = pyfits.open(fname)
        xxc = hdul_pup['APERTURE'].data['XXC']
        yyc = hdul_pup['APERTURE'].data['YYC']
        trm = hdul_pup['APERTURE'].data['TRM']
        hdul_pup.close()
        
        txt = ''
        for i in range(len(trm)):
            temp = rot.dot(np.array([xxc[i], yyc[i]]))
            txt += '%+.5f %+.5f %.5f\n' % (temp[0], temp[1], trm[i])
        txtfile = open('pupil_model.txt', 'w')
        txtfile.write(txt)
        txtfile.close()
        
        KPO = kpo.KPO(fname='pupil_model.txt',
                      array=None,
                      ndgt=5,
                      bmax=self.bmax,
                      ID='')
        KPO.extract_KPD_single_frame(data,
                                     PSCALE,
                                     wave,
                                     target=None,
                                     recenter=False,
                                     wrad=None,
                                     method='LDFT1')
        
        if (output_dir is None):
            path = file[:-5]
        else:
            temp = file.rfind('/')
            if (temp == -1):
                path = output_dir+file[:-5]
            else:
                path = output_dir+file[temp+1:-5]
        
        if (self.plot == True):
            f, ax = plt.subplots(2, 2, figsize=(1.5*6.4, 1.5*4.8))
            d00 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data)))
            p00 = ax[0, 0].imshow(np.angle(d00), origin='lower', vmin=-np.pi, vmax=np.pi)
            c00 = plt.colorbar(p00, ax=ax[0, 0])
            c00.set_label('Fourier phase [rad]', rotation=270, labelpad=20)
            m2pix = core.mas2rad(PSCALE)*sx/wave
            xx = KPO.kpi.UVC[:, 0]*m2pix+sx//2
            yy = KPO.kpi.UVC[:, 1]*m2pix+sy//2
            ax[0, 0].scatter(xx, yy, s=0.2, c='red')
            ax[0, 0].set_title('Fourier phase', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            ww = np.argsort(KPO.kpi.BLEN)
            ax[1, 0].plot(np.angle(KPO.CVIS[0][0, ww]))
            ax[1, 0].axhline(0., ls='--', color='black')
            ax[1, 0].set_ylim([-np.pi, np.pi])
            ax[1, 0].grid(axis='y')
            ax[1, 0].set_xlabel('Index sorted by baseline length')
            ax[1, 0].set_ylabel('Fourier phase [rad]')
            ax[1, 0].set_title('Fourier phase', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            ax[1, 1].plot(KPO.KPDT[0][0, :])
            ax[1, 1].axhline(0., ls='--', color='black')
            # ax[1, 1].set_ylim([-np.pi, np.pi])
            ax[1, 1].grid(axis='y')
            ax[1, 1].set_xlabel('Index')
            ax[1, 1].set_ylabel('Kernel-phase [rad]')
            ax[1, 1].set_title('Kernel-phase', y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Extract kerphase step', size=18)
            plt.tight_layout()
            plt.savefig(path+suffix_out+'.pdf')
            if (show_plots == True):
                plt.show()
            plt.close()
        
        print('Done')
        
        return suffix_out

class empirical_uncertainties():
    """
    Compute empirical uncertainties for the kernel-phase.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = True
        self.plot = True
        
        pass
    
    def step(self,
             file,
             suffix,
             output_dir):
        """
        """
        
        raise UserWarning('Not implemented yet')
        
        pass
