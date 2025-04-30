# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import xaosim as xs
import xara
from xaosim.pupil import PHARO
from scipy.ndimage import rotate

# %%
TEST_GEN = False
OLD_MODEL = False

binary_model = False  # use a finer transmission model for the aperture
PSZ = 4978 * 2  # size of the array for the model
pdiam = 4.978  # telescope diameter in meters
mstep = 0.160  # step size in meters

data_dir = Path("./tests/data/PHARO")
assert data_dir.exists(), f"Data directory {data_dir} not found"
out_dir = Path("./tests/PHARO")
out_dir.mkdir(exist_ok=True)

mtype = "bina" if binary_model else "grey"
if OLD_MODEL and not TEST_GEN:
    fname = f"p3k_med_{mtype}_model_old.fits"
else:
    fname = f"p3k_med_{mtype}_model.fits"

if TEST_GEN:
    pmask = PHARO(PSZ, PSZ / 2, mask="med")
    pmask2 = PHARO(PSZ, PSZ / 2, mask="med", ang=-2)  # rotated!
    ppscale = pdiam / PSZ

    if binary_model:
        p3k_model = xara.core.create_discrete_model(
            pmask, ppscale, mstep, binary=True, tmin=0.4
        )
    else:
        p3k_model = xara.core.create_discrete_model(
            pmask, ppscale, mstep, binary=False, tmin=0.05
        )
        p3k_model[:, 2] = np.round(p3k_model[:, 2], 2)

    # rotate the model by two degrees
    # --------------------------------

    th0 = -2.0 * np.pi / 180.0  # rotation angle
    rmat = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])

    p3k_model[:, :2] = p3k_model[:, :2].dot(rmat)

    # -------------------------
    #      simple plot
    # -------------------------
    f0 = plt.figure(0)
    ax = f0.add_subplot(111)
    ax.imshow(pmask2)
    ax.plot(PSZ / 2 + p3k_model[:, 0] / ppscale, PSZ / 2 + p3k_model[:, 1] / ppscale, "b.")
    f0.set_size_inches(5, 5, forward=True)
    f0.savefig(out_dir / "rotated_pupil.png")
    plt.show()

    # compute the kernel-phase data structure
    # TODO: Should KPO be added back to xara directly or accessed via xara.kpo.KPO now?
    kpo_0 = xara.kpo.KPO(array=p3k_model, bmax=4.646)

    # show side by side, the pupil model and its associated uv-coverage
    kpo_0.kpi.plot_pupil_and_uv(
        xymax=2.5, cmap=cm.plasma_r, ssize=9, figsize=(10, 5), marker="o"
    )
    plt.show()

    # and save to a multi-extension kernel-phase fits file for later use
    print(f"saving {fname}")
    model_path = out_dir / fname
    kpo_0.save_as_fits(model_path)
else:
    # TODO:: Test backward compat with older model from online
    model_path = data_dir / fname


# %%
tgt_cube = fits.getdata(data_dir / "tgt_cube.fits")  # alpha Ophiuchi
ca2_cube = fits.getdata(data_dir / "ca2_cube.fits")  # epsilon Herculis

pscale = 25.0  # plate scale of the image in mas/pixels
wl = 2.145e-6  # central wavelength in meters (Hayward paper)
ISZ = tgt_cube.shape[1]  # image size
kpo1 = xara.kpo.KPO(fname=str(model_path))
kpo2 = kpo1.copy()

kpo1.extract_KPD_single_cube(
    tgt_cube, pscale, wl, target="alpha Ophiuchi", recenter=True
)
kpo2.extract_KPD_single_cube(
    ca2_cube, pscale, wl, target="epsilon Herculis", recenter=True
)

# %%
data1 = np.array(kpo1.KPDT)[0]
data2 = np.array(kpo2.KPDT)[0]

mydata = np.median(data1, axis=0) - np.median(data2, axis=0)
myerr  = np.sqrt(np.var(data1, axis=0) / (kpo1.KPDT[0].shape[0] - 1) + np.var(data2, axis=0) / (kpo2.KPDT[0].shape[0] - 1))
if OLD_MODEL:
    myerr = np.sqrt(myerr**2 + 1.2**2)  # was 1.2 in tutorial, could convert with kpm_norm?
else:
    myerr = np.sqrt(myerr**2 + 0.012**2)  # was 1.2 in tutorial, could convert with kpm_norm?

# %%
print("\ncomputing colinearity map...")
gsize = 100 # gsize x gsize grid
gstep = 10 # grid step in mas
xx, yy = np.meshgrid(
        np.arange(gsize) - gsize/2, np.arange(gsize) - gsize/2)
azim = -np.arctan2(xx, yy) * 180.0 / np.pi
dist = np.hypot(xx, yy) * gstep

#mmap = kpo1.kpd_binary_match_map(100, 10, mydata/myerr, kpo1.CWAVEL[1], norm=True)
mmap = kpo1.kpd_binary_match_map(100, 10, mydata, kpo1.CWAVEL[0], norm=True)
x0, y0 = np.argmax(mmap) % gsize, np.argmax(mmap) // gsize
print("max colinearity found for sep = %.2f mas and ang = %.2f deg" % (
        dist[y0, x0], azim[y0, x0]))

f1 = plt.figure(figsize=(5,5))
ax1 = f1.add_subplot(111)
ax1.imshow(mmap, extent=(
        gsize/2*gstep, -gsize/2*gstep, -gsize/2*gstep, gsize/2*gstep))
ax1.set_xlabel("right ascension (mas)")
ax1.set_ylabel("declination (mas)")
ax1.plot([0,0], [0,0], "w*", ms=16)
ax1.set_title("Calibrated signal colinearity map")
ax1.grid()
f1.set_tight_layout(True)
plt.show()

# %%
print("\nbinary model fitting...")
p0 = [dist[y0, x0], azim[y0, x0], mmap.max()] # good starting point

mfit = kpo1.binary_model_fit(p0, calib=kpo2)
p1 = mfit[0] # the best fit parameter vector (sep, P.A., contrast)

cvis_b = xara.core.cvis_binary(
        kpo1.kpi.UVC[:,0], kpo1.kpi.UVC[:,1], wl, p1) # binary
ker_theo = kpo1.kpi.KPM.dot(np.angle(cvis_b))

# %%
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

ax.errorbar(ker_theo, mydata, yerr=myerr, fmt="none", ecolor='c')
ax.plot(ker_theo, mydata, 'b.')
mmax = np.abs(mydata).max()
ax.plot([-mmax,mmax],[-mmax,mmax], 'r')
ax.set_ylabel("data kernel-phase")
ax.set_xlabel("model kernel-phase")
ax.set_title('kernel-phase correlation diagram')
ax.axis("equal")
fig.set_tight_layout(True)
plt.show()

# %%
if myerr is not None:
        chi2 = np.sum(((mydata - ker_theo)/myerr)**2) / kpo1.kpi.nbkp
else:
        chi2 = np.sum(((mydata - ker_theo))**2) / kpo1.kpi.nbkp

print("sep = %3f, ang=%3f, con=%3f => chi2 = %.3f" % (p1[0], p1[1], p1[2], chi2))
print("correlation matrix of parameters")
print(np.round(mfit[1], 2))
