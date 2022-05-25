from __future__ import division

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

from xara.core import create_discrete_model, symetrizes_model
from xara import kpi


# =============================================================================
# NIRISS CLEAR PUPIL MODEL
# =============================================================================

# Parameters.
step = 0.3 # m, grid step size
tmin = 1e-1 # minimum transmission for a sub-aperture to be considered
binary = False # binary or grey mask
# textpath = 'niriss_clear_pupil.txt'
# fitspath = 'niriss_clear_pupil.fits'
textpath = 'niriss_clear_pupil_085_norot.txt'
fitspath = 'niriss_clear_pupil_085_norot.fits'
bmax = None

# Load pupil.
hdul = pyfits.open('MASK_CLEARP.fits')
aper = hdul[0].data
pxsc = hdul[0].header['PUPLSCAL'] # m, pupil pixel scale
pxsc *= 0.85

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


# =============================================================================
# NIRISS NRM PUPIL MODEL
# =============================================================================

# Parameters.
step = 0.3 # m, grid step size
tmin = 1e-1 # minimum transmission for a sub-aperture to be considered
binary = False # binary or grey mask
textpath = 'niriss_nrm_pupil.txt'
fitspath = 'niriss_nrm_pupil.fits'
bmax = None

# Load pupil.
hdul = pyfits.open('MASK_NRM.fits')
aper = hdul[0].data
pxsc = hdul[0].header['PUPLSCAL'] # m, pupil pixel scale

# Create discrete pupil model using XARA.
model = create_discrete_model(aper, pxsc, step, binary=binary, tmin=tmin)
np.savetxt(textpath, model, fmt='%+.6e %+.6e %.2f')
KPI = kpi.KPI(fname=textpath, bmax=bmax)
KPI.package_as_fits(fname=fitspath)

# Plot pupil model.
f = KPI.plot_pupil_and_uv(cmap='inferno')
plt.savefig(textpath[:-3]+'pdf')
plt.show(block=True)
plt.close()

import pdb; pdb.set_trace()
