''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data 
    structures, using the theory described in the two following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.

    This file contains several tools used by the KPI and KPO classes to
    create and manipulate kernel- and eigen-phase data structures.
    -------------------------------------------------------------------- '''

import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
from scipy.signal import medfilt2d as medfilt
from scipy.special import j1
from cameras import *

''' ================================================================
    small tools and functions useful for the manipulation of
    Ker-phase data.
    ================================================================ '''

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

dtor = np.pi/180.0

# =========================================================================
# =========================================================================

def mas2rad(x):
    ''' Convenient little function to convert milliarcsec to radians '''
    return x*np.pi/648000000.0 # = x*np.pi/(180*3600*1000)

# =========================================================================
# =========================================================================
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# =========================================================================
# =========================================================================
def negentropy(a):
    ''' Evaluate the negentropy associated to an strict positive array "a".
    The result is a positive number.
    '''
    a1 = a.copy() / a.sum()
    entm = np.log10(a1) * a1
    return -1.0 * entm.sum()

# =========================================================================
# =========================================================================
def cvis_binary(u, v, wavel, p, norm=None):
    ''' Calc. complex vis measured by an array for a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)
    
    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)

    - norm=None (required for vis2)
    ---------------------------------------------------------------- '''

    p = np.array(p)
    # relative locations
    th = (p[1] + 90.0) * np.pi / 180.0
    ddec =  mas2rad(p[0] * np.sin(th))
    dra  = -mas2rad(p[0] * np.cos(th))

    # baselines into number of wavelength
    x = np.sqrt(u*u+v*v)/wavel

    # decompose into two "luminosity"
    l2 = 1. / (p[2] + 1)
    l1 = 1 - l2
    
    # phase-factor
    phi = np.zeros(u.size, dtype=complex)
    phi.real = np.cos(-2*np.pi*(u*dra + v*ddec)/wavel)
    phi.imag = np.sin(-2*np.pi*(u*dra + v*ddec)/wavel)

    # optional effect of resolved individual sources
    if p.size == 5:
        th1, th2 = mas2rad(p[3]), mas2rad(p[4])
        v1 = 2*j1(np.pi*th1*x)/(np.pi*th1*x)
        v2 = 2*j1(np.pi*th2*x)/(np.pi*th2*x)
    else:
        v1 = np.ones(u.size)
        v2 = np.ones(u.size)

    cvis = l1 * v1 + l2 * v2 * phi

    return cvis

# =========================================================================
# Experiment: original version does not reflect the reality of a
# 1:1 binary, for which things should be computed from the barycenter
# note that this doesn't matter when dealing with kernel-phase
# =========================================================================
def cvis_binary2(u, v, wavel, p, norm=None):
    ''' Calc. complex vis measured by an array for a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)
    
    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)

    - norm=None (required for vis2)
    ---------------------------------------------------------------- '''

    p = np.array(p)
    # relative locations
    th = (p[1] + 90.0) * np.pi / 180.0
    ddec =  mas2rad(p[0] * np.sin(th))
    dra  = -mas2rad(p[0] * np.cos(th))

    # baselines into number of wavelength
    x = np.sqrt(u*u+v*v)/wavel

    # decompose into two "luminosity"
    l2 = 1. / (p[2] + 1)
    l1 = 1 - l2
    
    # phase-factor
    phi1 = np.zeros(u.size, dtype=complex)
    phi1.real = np.cos(-2*np.pi*l1*(u*dra + v*ddec)/wavel)
    phi1.imag = np.sin(-2*np.pi*l1*(u*dra + v*ddec)/wavel)

    phi2 = np.zeros(u.size, dtype=complex)
    phi2.real = np.cos(2*np.pi*l2*(u*dra + v*ddec)/wavel)
    phi2.imag = np.sin(2*np.pi*l2*(u*dra + v*ddec)/wavel)

    # optional effect of resolved individual sources
    if p.size == 5:
        th1, th2 = mas2rad(p[3]), mas2rad(p[4])
        v1 = 2*j1(np.pi*th1*x)/(np.pi*th1*x)
        v2 = 2*j1(np.pi*th2*x)/(np.pi*th2*x)
    else:
        v1 = np.ones(u.size)
        v2 = np.ones(u.size)

    cvis = l1 * v1 * phi1 + l2 * v2 * phi2

    return cvis

# =========================================================================
# =========================================================================
def grid_precalc_aux_cvis(u, v, wavel, mgrid, gscale):
    ''' Pre-calculates an auxilliary array necessary for the complex 
    visibility modeling of a "complex" (i.e. grid-type) source
    ----------------------------------------------------------------
    - u,v:    baseline coordinates (meters)
    - wavel:  wavelength (meters)
    - mgrid:  square grid array describing the object
    - gscale: scalar defining the "mgrid" pitch size (in mas)
    ---------------------------------------------------------------- '''
    # relative locations
    sz, dz = mgrid.shape[0], mgrid.shape[0] / 2
    xx,yy = np.meshgrid(np.arange(sz)-dz, np.arange(sz)-dz)

    # flatten all 2D arrays 
    dra  = mas2rad(gscale * np.ravel(yy))
    ddec = mas2rad(gscale * np.ravel(xx))

    c0 = -2j * np.pi / wavel # pre-compute coeff for speed?
    phi = np.exp(c0 * (np.outer(u, dra) + np.outer(v, ddec)))
    return phi

# =========================================================================
# =========================================================================
def grid_src_cvis(u, v, wavel, mgrid, gscale, phi=None):
    ''' Calc. complex vis measured by an array for a complex object
    ----------------------------------------------------------------
    - u,v:    baseline coordinates (meters)
    - wavel:  wavelength (meters)
    - mgrid:  square grid array describing the object
    - gscale: scalar defining the "mgrid" pitch size (in mas)
    - phi:    pre-computed auxilliary array to speed up calculation
    ---------------------------------------------------------------- '''

    if phi == None:
        phi = cvis_precalc_aux(u, v, wavel, mgrid, gscale)
    cvis = np.dot(phi, np.ravel(mgrid))
    
    return cvis

# =========================================================================
# =========================================================================
def phase_binary(u, v, wavel, p, deg=True):
    ''' Calculate the phases observed by an array on a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)
    
    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)
    ---------------------------------------------------------------- '''
    cvis = cvis_binary(u,v,wavel, p)
    phase = np.angle(cvis, deg=deg)
    if deg:
        return np.mod(phase + 10980., 360.) - 180.0
    else:
        return phase

# =========================================================================
# =========================================================================
def vis2_binary(u, v, wavel, p):
    ''' Calc. squared vis. observed by an array on a binary star
    --------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
      p[0] = sep (mas)
      p[1] = PA (deg) E of N.
      p[2] = contrast ratio (primary/secondary)

    optional:
      p[3] = angular size of primary (mas)
      p[4] = angular size of secondary (mas)

    u,v: baseline coordinates (meters)
    wavel: wavelength (meters)
    ---------------------------------------------------------------- '''

    cvis = cvis_binary(u, v, wavel, p, norm=True)
    return np.abs(cvis)**2

# =========================================================================
# =========================================================================

def super_gauss(xs, ys, x0, y0, w):
    ''' Returns an 2D super-Gaussian function
    ------------------------------------------
    Parameters:
    - (xs, ys) : array size
    - (x0, y0) : center of the Super-Gaussian
    - w        : width of the Super-Gaussian 
    ------------------------------------------ '''

    x = np.outer(np.arange(xs), np.ones(ys))-x0
    y = np.outer(np.ones(xs), np.arange(ys))-y0
    dist = np.sqrt(x**2 + y**2)

    gg = np.exp(-(dist/w)**4)
    return gg

# =========================================================================
# =========================================================================

def centroid(image, threshold=0, binarize=0):                        
    ''' ------------------------------------------------------
        simple determination of the centroid of a 2D array
    ------------------------------------------------------ '''

    signal = np.where(image > threshold)
    sy, sx = image.shape[0], image.shape[1] # size of "image"
    bkg_cnt = np.median(image)                                       

    temp = np.zeros((sy, sx))
    if (binarize == 1): temp[signal] = 1.0
    else:               temp[signal] = image[signal]

    profx = 1.0 * temp.sum(axis=0)
    profy = 1.0 * temp.sum(axis=1)
    profx -= np.min(profx)                                           
    profy -= np.min(profy)

    x0 = (profx*np.arange(sx)).sum() / profx.sum()
    y0 = (profy*np.arange(sy)).sum() / profy.sum()

    return (x0, y0)

# =========================================================================
# =========================================================================

def find_psf_center(img, verbose=True, nbit=10):                     
    ''' Name of function self explanatory: locate the center of a PSF.

    ------------------------------------------------------------------
    Uses an iterative method with a window of shrinking size to 
    minimize possible biases (non-uniform background, hot pixels, etc)

    Options:
    - nbit: number of iterations (default 10 is good for 512x512 imgs)
    - verbose: in case you are interested in the convergence
    ------------------------------------------------------------------ '''
    temp = img.copy()
    bckg = np.median(temp)   # background level
    temp -= bckg
    mfilt = medfilt(temp, 3) # median filtered, kernel size = 3
    (sy, sx) = mfilt.shape   # size of "image"
    xc, yc = sx/2, sy/2      # first estimate for psf center

    signal = np.zeros_like(img)
    signal[mfilt > 0.1*mfilt.max()] = 1.0

    for it in xrange(nbit):
        sz = sx/2/(1.0+(0.1*sx/2*it/(4*nbit)))
        x0 = np.max([int(0.5 + xc - sz), 0])
        y0 = np.max([int(0.5 + yc - sz), 0])
        x1 = np.min([int(0.5 + xc + sz), sx])
        y1 = np.min([int(0.5 + yc + sz), sy])
                                                                     
        mask = np.zeros_like(img)
        mask[y0:y1, x0:x1] = 1.0
        
        #plt.clf()
        #plt.imshow((mfilt**0.2) * mask)
        #plt.draw()

        profx = (mfilt*mask*signal).sum(axis=0)
        profy = (mfilt*mask*signal).sum(axis=1)
        
        xc = (profx*np.arange(sx)).sum() / profx.sum()
        yc = (profy*np.arange(sy)).sum() / profy.sum()
                  
        #pdb.set_trace()
                                                   
        if verbose:
            print("it #%2d center = (%.2f, %.2f)" % (it+1, xc, yc))
            
    return (xc, yc)                                                  

# =========================================================================
# =========================================================================

def recenter(im0, sg_rad=25.0, verbose=True, nbit=10):
    ''' ------------------------------------------------------------
         The ultimate image centering algorithm... eventually...

        im0:    of course, the array to be analyzed
        sg_rad: super-Gaussian mask radius
        bflag:  if passed as an argument, a "bad" boolean is returned
        ------------------------------------------------------------ '''

    szh = im0.shape[1] # horiz
    szv = im0.shape[0] # vertic

    temp = np.max(im0.shape) # max dimension of image

    for sz in [64, 128, 256, 512, 1024, 2048]:
        if sz >= temp: break

    dz = sz/2.           # image half-size

    sgmask = super_gauss(sz, sz, dz, dz, sg_rad)
    x,y = np.meshgrid(np.arange(sz)-dz, np.arange(sz)-dz)
    wedge_x, wedge_y = x*np.pi/dz, y*np.pi/dz
    offset = np.zeros((sz, sz), dtype=complex) # to Fourier-center array

    # insert image in zero-padded array (dim. power of two)
    im = np.zeros((sz, sz))
    orih, oriv = (sz-szh)/2, (sz-szv)/2
    im[oriv:oriv+szv,orih:orih+szh] = im0

    (x0, y0) = find_psf_center(im, verbose, nbit)
    
    im -= np.median(im)

    dx, dy = (x0-dz), (y0-dz)
    im = np.roll(np.roll(im, -int(dx), axis=1), -int(dy), axis=0)

    sys.stdout.write("\rrecenter: dx=%.2f, dy=%.2f" % (dx, dy))
    sys.stdout.flush()
    dx -= np.int(dx)
    dy -= np.int(dy)

    temp = im * sgmask
    mynorm = temp.sum()

    # array for Fourier-translation
    dummy = shift(dx * wedge_x + dy * wedge_y)
    offset.real, offset.imag = np.cos(dummy), np.sin(dummy)
    dummy = np.abs(shift(ifft(offset * fft(shift(im*sgmask)))))

    #dummy = im
    # image masking, and set integral to right value
    dummy *= sgmask

    return (dummy * mynorm / dummy.sum())


# =========================================================================
# =========================================================================
def extract_from_array(array, hdr, kpi, wrad, save_im=True, re_center=True,
                       plotim=False, plotuv=False):
    ''' Extract the Kernel-phase signal from a ndarray + header info.
    
    ----------------------------------------------------------------
    Assumes that the array has been cleaned.
    In order to be able to extract information at the right place,
    a header must be provided as additional argument.
    
    In addition to the actual Kernel-phase signal, some information
    is extracted from the fits header to help with the interpretation
    of the data to follow.
    
    Parameters are:
    - array: the frame to be examined
    - kpi:   the k-phase info structure to decode the data
    - wrad:  window radius in pixels (e.g. = 25)
    - save_im: optional flag to set to False to forget the images
    and save some RAM space

    Options:
    -re_center: re-centers the frame before extraction
    - plotim:   plots image
    - plotuv:   plots uv phase map

    The function returns a tuple:
    - information, kernal-phase, uv-phase,  square visibiities
    - (kpd_info,   kpd_signal,   kpd_phase, vis2)
    - (kpd_info,   kpd_signal,   kpd_phase, vis2, im, ac)
    ---------------------------------------------------------------- '''

    if 'Keck II' in hdr['TELESCOP']: kpd_info = get_keck_keywords(hdr)
    if 'HST'     in hdr['TELESCOP']: kpd_info = get_nic1_keywords(hdr)
    if 'simu'    in hdr['TELESCOP']: kpd_info = get_simu_keywords(hdr)
    if 'Hale'    in hdr['TELESCOP']: kpd_info = get_pharo_keywords(hdr)
    
    # read and fine-center the frame
    if re_center: im = recenter(array, sg_rad=wrad, verbose=False, nbit=20)
    else:         im = array.copy()

    sz, dz = im.shape[0], im.shape[0]/2  # image is now square

    # meter to pixel conversion factor
    m2pix = mas2rad(kpd_info['pscale']) * sz / kpd_info['filter']
    uv_samp = kpi.uv * m2pix + dz # uv sample coordinates in pixels
    
    # calculate and normalize Fourier Transform
    ac = shift(fft(shift(im)))
    ac /= (np.abs(ac)).max() / kpi.nbh

    xx = np.cast['int'](np.round(uv_samp[:,1]))
    yy = np.cast['int'](np.round(uv_samp[:,0]))
    data_cplx = ac[xx, yy]

    vis2   = np.abs(data_cplx)**2 # square visibilities
    vis2  /= np.abs(ac[dz,dz])**2 # normalized (just to be sure)

    # ---------------------------
    # calculate the Kernel-phases
    # ---------------------------
    #uvph = kpi.RED * np.angle(data_cplx) # in radians for WFS
    uvph = np.angle(data_cplx) # uv-phase (in radians for WFS)
    #kpd_signal = np.dot(kpi.KerPhi, kpi.RED*np.angle(data_cplx)) / dtor
    kpd_signal = np.dot(kpi.KerPhi, np.angle(data_cplx)) / dtor

    if (save_im): res = (kpd_info, kpd_signal, uvph, vis2, im, ac)
    else:         res = (kpd_info, kpd_signal, uvph, vis2)

    uvw = np.max(uv_samp)/2

    if plotim or plotuv:
        plt.clf()
        f0 = plt.subplot(121)
        f0.imshow(im[dz-wrad:dz+wrad,dz-wrad:dz+wrad]**0.5)
        f1 = plt.subplot(122)
        f1.imshow(np.angle(ac))
        f1.plot(uv_samp[:,0], uv_samp[:,1], 'b.')
        f1.axis((dz-uvw, dz+uvw, dz-uvw, dz+uvw))
        plt.draw()
    return res

