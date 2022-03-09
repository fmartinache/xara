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

import core
import kpi
import kpo


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
        
        pass
    
    def run(self,
            fitsfile):
        """
        Function to run the pipeline.
        """
        
        pass
    
    def _write_as_kpfits(self):
        """
        Function to write the extracted data into a kernel-phase FITS file.
        
        The FITS file structure has been agreed upon by the participants of
        Steph Sallum's masking & kernel-phase hackathon in 2021 and is defined
        here: https://docs.google.com/document/d/1iBbcCYiq9J2PpLSr21-xB4AXP8X_6tSszxnHY1VDGXg/edit?usp=sharing
        """
        
        pass

class fix_bad_pixels():
    """
    Fix bad pixels using the Kammerer & Ireland method.
    
    The corresponding references are https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
    and https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        
        pass

class recenter_frames():
    """
    Re-center the individual frames.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        
        pass

class window_frames():
    """
    Window the individual frames.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        
        pass

class extract_kerphase():
    """
    Extract the kernel-phase.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        
        pass

class empirical_uncertainties():
    """
    Compute empirical uncertainties for the kernel-phase.
    """
    
    def __init__(self):
        """
        """
        
        self.skip = False
        
        pass
