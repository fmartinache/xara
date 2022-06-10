from __future__ import division

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

from scipy.ndimage import rotate, shift

from opticstools import opticstools as ot
from xara.core import create_discrete_model, symetrizes_model
from xara import kpi


# =============================================================================
# NIRISS CLEAR PUPIL MODEL (REGULAR GRID)
# =============================================================================

# Parameters.
step = 0.3 # m, grid step size
tmin = 1e-1 # minimum transmission for a sub-aperture to be considered
binary = False # binary or grey mask
textpath = 'niriss_clear_pupil.txt'
fitspath = 'niriss_clear_pupil.fits'
bmax = None # m
hexa = True

# Load pupil.
hdul = pyfits.open('MASK_CLEARP.fits')
aper = hdul[0].data
# aper = rotate(aper, 0.56126717, order=1)
# plt.ioff()
# plt.imshow(aper, origin='lower')
# plt.colorbar()
# plt.show()
# import pdb; pdb.set_trace()
pxsc = hdul[0].header['PUPLSCAL'] # m, pupil pixel scale

# Create discrete pupil model using XARA.
model = create_discrete_model(aper, pxsc, step, binary=binary, tmin=tmin)
model = symetrizes_model(model)
np.savetxt(textpath, model, fmt='%+.10e %+.10e %.2f')
KPI = kpi.KPI(fname=textpath, bmax=bmax, hexa=hexa)
KPI.filter_baselines(KPI.RED > 10)
# KPI.filter_baselines(KPI.RED > 8)
KPI.package_as_fits(fname=fitspath)

# Plot pupil model.
f = KPI.plot_pupil_and_uv(cmap='inferno')
plt.savefig(textpath[:-3]+'pdf')
plt.show(block=True)
plt.close()

import pdb; pdb.set_trace()


# =============================================================================
# NIRISS CLEAR PUPIL MODEL (HEXAGONAL GRID)
# =============================================================================

# Parameters.
step = 0.3 # m, grid step size
tmin = 1e-1 # minimum transmission for a sub-aperture to be considered
binary = False # binary or grey mask
textpath = 'niriss_clear_pupil_hex.txt'
fitspath = 'niriss_clear_pupil_hex.fits'
bmax = None # m
hexa = True

# Load pupil.
hdul = pyfits.open('MASK_CLEARP.fits')
aper = hdul[0].data
# aper = rotate(aper, 0.56126717, order=1)
# plt.ioff()
# plt.imshow(aper, origin='lower')
# plt.colorbar()
# plt.show()
# import pdb; pdb.set_trace()
pxsc = hdul[0].header['PUPLSCAL'] # m, pupil pixel scale

# 
# xy = mirror segment centers
# xy_all = subaperture centers
d_hex = 1.32 # m; short diagonal of an individual mirror segment
D_hex = d_hex*2./np.sqrt(3.) # m; long diagonal of an individual mirror segment
xy = np.array([[0., 0.],
               [0., -d_hex],
               [0., d_hex],
               [0., -2.*d_hex],
               [0., 2.*d_hex],
               [0., -3.*d_hex],
               [0., 3.*d_hex],
               [-0.75*D_hex, -0.5*d_hex],
               [-0.75*D_hex, 0.5*d_hex],
               [-0.75*D_hex, -1.5*d_hex],
               [-0.75*D_hex, 1.5*d_hex],
               [-0.75*D_hex, -2.5*d_hex],
               [-0.75*D_hex, 2.5*d_hex],
               [0.75*D_hex, -0.5*d_hex],
               [0.75*D_hex, 0.5*d_hex],
               [0.75*D_hex, -1.5*d_hex],
               [0.75*D_hex, 1.5*d_hex],
               [0.75*D_hex, -2.5*d_hex],
               [0.75*D_hex, 2.5*d_hex],
               [-1.5*D_hex, 0.],
               [-1.5*D_hex, -d_hex],
               [-1.5*D_hex, d_hex],
               [-1.5*D_hex, -2.*d_hex],
               [-1.5*D_hex, 2.*d_hex],
               [1.5*D_hex, 0.],
               [1.5*D_hex, -d_hex],
               [1.5*D_hex, d_hex],
               [1.5*D_hex, -2.*d_hex],
               [1.5*D_hex, 2.*d_hex],
               [-2.25*D_hex, -0.5*d_hex],
               [-2.25*D_hex, 0.5*d_hex],
               [-2.25*D_hex, -1.5*d_hex],
               [-2.25*D_hex, 1.5*d_hex],
               [2.25*D_hex, -0.5*d_hex],
               [2.25*D_hex, 0.5*d_hex],
               [2.25*D_hex, -1.5*d_hex],
               [2.25*D_hex, 1.5*d_hex]])
xy_all = []
for i in range(xy.shape[0]):
    for j in range(3):
        xx = d_hex/3.*np.sin(j/3.*2.*np.pi)
        yy = d_hex/3.*np.cos(j/3.*2.*np.pi)
        xy_all += [[xy[i, 0]+xx, xy[i, 1]+yy]]
xy_all = np.array(xy_all)

mask = np.zeros_like(aper)
for i in range(xy_all.shape[0]):
    sapt = ot.hexagon(aper.shape[0], 0.45*D_hex/pxsc)
    sapt = shift(sapt, (xy_all[i, 1]/pxsc, xy_all[i, 0]/pxsc), order=0)
    mask = (mask > 0.5) | (sapt > 0.5)

ext = aper.shape[0]*pxsc/2.
plt.ioff()
plt.imshow(aper, origin='lower', extent=(-ext, ext, -ext, ext))
plt.imshow(mask, origin='lower', extent=(-ext, ext, -ext, ext), alpha=0.5)
plt.scatter(xy_all[:, 0], xy_all[:, 1])
plt.xlim([-ext, ext])
plt.ylim([-ext, ext])
plt.title('Undersized subapertures for illustrative purposes')
plt.show()

model = []
for i in range(xy_all.shape[0]):
    sapt = ot.hexagon(aper.shape[0], 0.50*D_hex/pxsc)
    sapt = shift(sapt, (xy_all[i, 1]/pxsc, xy_all[i, 0]/pxsc), order=0)
    ww = sapt > 0.5
    if (np.sum(ww) < 0.5):
        tt = 0.
    else:
        tt = np.mean(aper[ww])
    if (tt == 0.):
        continue
    else:
        model += [[xy_all[i, 0], xy_all[i, 1], tt]]
model = np.array(model)
model = symetrizes_model(model)
np.savetxt(textpath, model, fmt='%+.10e %+.10e %.2f')
KPI = kpi.KPI(fname=textpath, bmax=bmax, hexa=hexa)
KPI.filter_baselines(KPI.RED > 2)
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
