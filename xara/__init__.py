''' --------------------------------------------------------------------
                PYSCO: PYthon Self Calibrating Observables
    --------------------------------------------------------------------
    ---
    pysco is a python module to create, and extract Kernel-phase data 
    structures, using the theory of Martinache, 2010, ApJ, 724, 464.
    ----

    The module is constructed around two main classes:
    -------------------------------------------------

    - KPI: Kernel-Phase Information
      object that contains the linear model for the optical system 
      of interest. 

    - KPO: Kernel-Phase Observation
      object that contains Ker-phase data extracted from actual
      images, using the model of KerPhase_Relation + some other
      information: plate scale, wavelength, ...
      -------------------------------------------------------------------- '''

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import astropy.io.fits as pf
except:
    import pyfits as pf
    
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

version_info = (1,1,0)
__version__ = '.'.join(str(c) for c in version_info)

# -------------------------------------------------
# set some defaults to display images that will
# look more like the DS9 display....
# -------------------------------------------------
#plt.set_cmap(cm.gray)
(plt.rcParams)['image.origin']        = 'lower'
(plt.rcParams)['image.interpolation'] = 'nearest'
# -------------------------------------------------

plt.ion()
plt.show()

