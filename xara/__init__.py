#!/usr/bin/env python

'''------------------------------------------------------------------
                XARA: Extreme Angular Resolution Astronomy
    ------------------------------------------------------------------
    ---
    XARA is a python module to create, and extract Kernel-phase data
    structures, using the theory of Martinache, 2010, ApJ, 724, 464.
    ----

    The module is constructed around two main classes:
    -------------------------------------------------

    - KPI: Kernel-Phase Information

      An object packing the data structures that guide the
      interpretation of images from an inteferometric point of view,
      leading to applications like kernel-phase and/or wavefront
      sensing

    - KPO: Kernel-Phase Observation

      An object that contains a KPI along with optional data extracted
      from the Fourier transform of images, using the KPI model and a
      handful additional pieces of information: wavelength, pixel scale,
      detector position angle and epoch to enable their interpretation
      ---------------------------------------------------------------- '''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import astropy.io.fits as pf

import copy
import pickle
import os
import sys


from scipy.optimize import leastsq
from scipy.interpolate import griddata

from . import core
from .core import *

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

dtor = np.pi/180.0

from . import kpi
from .kpi import *

from . import kpo
from .kpo import *

from .iwfs import *

version_info = (1, 3, 0)
__version__ = '.'.join(str(c) for c in version_info)

# -------------------------------------------------
# set some defaults to display images that will
# look more like the DS9 display....
# -------------------------------------------------
(plt.rcParams)['image.origin'] = 'lower'
(plt.rcParams)['image.interpolation'] = 'nearest'
# -------------------------------------------------

# =========================================================================
# =========================================================================
def field_map_cbar(gmap, gstep, cmap=cm.viridis, fsize=5.2,
                   vlabel="", vmin=None, vmax=None):
    ''' Produces a plot of a field map accompanied by a colorbar
        -------------------------------------------------------------
        Suited to the production of:
        - contrast detection limit map (kpd_binary_cdet_map)
        - binary signal colinearity map (kpd_binary_match_map)

        Parameters:
        ----------
        - gmap  : the 2D array to plot
        - gstep : grid step (in mas per pixel)
        - cmap  : a valid matplotlib colormap
        - fsize : the vertical size of the figure (in inches)
        - vlabel: a label to put by the colorbar
        - vmin  : lower cutoff for the values represented in the map
        - vmax  : upper cutoff for the values represented in the map
        ------------------------------------------------------------- '''
    f1, (ax1, cax) = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [1, 0.05]})
    f1.set_size_inches(1.25*fsize, fsize, forward=True)
    smax = gmap.shape[0]/2*gstep
    im1 = ax1.imshow(gmap,
                     extent=(-smax, smax, -smax, smax),
                     vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_xlabel("right ascension (mas)")
    ax1.set_ylabel("declination (mas)")
    cbar = f1.colorbar(im1, cax=cax, orientation="vertical")
    cbar.set_label(vlabel, fontsize=14)
    f1.set_tight_layout(True)
    return f1
