from __future__ import division

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

sys.path.append('/Users/jkammerer/Documents/Code/xara/xara')
from core import create_discrete_model, symetrizes_model
import kpi


# =============================================================================
# NIRISS CLEAR PUPIL MODEL
# =============================================================================

# Parameters.
step = 0.2 # m, grid step size
tmin = 1e-2 # minimum transmission for a sub-aperture to be considered
binary = False # binary or grey mask
textpath = 'nircam_clear_pupil.txt'
fitspath = 'nircam_clear_pupil.fits'
bmax = None

# Load pupil.
hdul = pyfits.open('MASKCLEAR.fits')
aper = hdul[0].data
pxsc = hdul[0].header['REALSCAL'] # m, pupil pixel scale

# Create discrete pupil model using XARA.
model = create_discrete_model(aper, pxsc, step, binary=binary, tmin=tmin)
model = symetrizes_model(model)
np.savetxt(textpath, model, fmt='%+.6e %+.6e %.2f')
KPI = kpi.KPI(fname=textpath, bmax=bmax)
KPI.package_as_fits(fname=fitspath)

# Plot pupil model.
f = KPI.plot_pupil_and_uv(cmap='inferno')
plt.savefig(textpath[:-3]+'pdf')
plt.show(block=True)
plt.close()

import pdb; pdb.set_trace()
