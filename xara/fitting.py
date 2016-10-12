''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data 
    structures, using the theory described in the two following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.

    This file contains several tools used for parametric model fitting
    of data extracted by the xara package from astronomical images.
    -------------------------------------------------------------------- '''

import numpy as np
from scipy.optimize import leastsq
from core import *
import sys
from numpy.random import rand, randn

# =========================================================================
# =========================================================================
def grid_src_KPD(mgrid, gscale, kpi, hdr, phi=None, deg=False):
    ''' Creates a kernel-phase model for a "complex" 
    (i.e. grid-type) source.

    ------------------------------------------------------------------
    - mgrid  : a square 2D flux model
    - gscale : scalar defining the "mgrid" pitch size (in mas)
    - kpi    : kernel phase info structure
    - hdr    : header data information
    - phi    : pre-computed auxilliary array to speed up calculation
    ------------------------------------------------------------------ '''

    filter = hdr['filter']
    cvis = grid_src_cvis(kpi.uv[:,0], kpi.uv[:,1], 
                         filter, mgrid, gscale, phi)
    kphase = np.dot(kpi.KerPhi, np.angle(cvis, deg=deg))
    return kphase

# =========================================================================
# =========================================================================
def grid_src_KPD_fit_residuals(mgrid, gscale, kpo, phi=None, deg=False, reg=0.0):
    ''' Function to evaluate fit residuals, to be used in a leastsq
    fitting procedure, for grid-type sources. 

    Options include:
    - phi: auxilliary table of precalculated cvis for fast calculation
    - deg: bool, do manipulate phases in degrees
    - reg: regularization parameter, for entropy
    '''

    nel = mgrid.size # total number of elements in square grid
    if np.size(mgrid.shape) == 1:
        g2 = mgrid.reshape((np.sqrt(nel), np.sqrt(nel)))
    else:
        g2 = mgrid.copy()
    test = grid_src_KPD(g2, gscale, kpo.kpi, kpo.hdr, phi, deg)
    err = kpo.kpd - test
    if kpo.kpe != None:
        err /= kpo.kpe
    if reg != 0.0:
        err += reg * negentropy(mgrid+1e-9)
    return err

# =========================================================================
# =========================================================================
def grid_src_KPD_fit(mgrid, gscale, kpo, phi=None, deg=False, reg=0.0):
    '''Performs a best binary fit search for the datasets.
    
    -------------------------------------------------------------
    p0 is the initial guess for the parameters 3 parameter vector
    typical example would be : [100.0, 0.0, 5.0].
    
    returns the full solution of the least square fit:
    - soluce[0] : best-fit parameters
    - soluce[1] : covariance matrix
    ------------------------------------------------------------- '''
    
    soluce = leastsq(grid_src_KPD_fit_residuals,
                     mgrid, args=((gscale, kpo, phi, deg, reg)),
                     full_output=1, ftol = 1e-10, factor=10.)
    
    covar = soluce[1]
    return soluce

# =========================================================================
# =========================================================================
def binary_model(params, kpi, hdr, vis2=False, deg=True):
    ''' Creates a binary Kernel-phase model.
    
    ------------------------------------------------------------------ 
    uses a simple 5 parameter binary star model for the uv phases that
    should be observed with the provided geometry.
    
    Additional parameters are:
    - kpi, a kernel phase info structure
    - hdr, a header data information
    ------------------------------------------------------------------ '''
    
    params2 = np.copy(params)
    if 'Hale' in hdr['tel']: params2[1] += 220.0 + hdr['orient']
    if 'HST'  in hdr['tel']: params2[1] -= hdr['orient']
    else:                    params2[1] += 0.0

    filter = hdr['filter']
    
    testPhi = phase_binary(kpi.uv[:,0], kpi.uv[:,1], filter, params2, deg)
    res = np.dot(kpi.KerPhi, testPhi)

    if vis2:
        res = vis2_binary(kpi.uv[:,0], kpi.uv[:,1], filter, params2, deg)
    return res

# =========================================================================
# =========================================================================
def binary_KPD_model(kpo, params):
    ''' Returns a 1D or 2D array of binary models.
    
    parameters are:
    - the kp structure (kpo) to be used as a template
    - the list of parameters of the binary model. '''

    models = np.zeros_like(kpo.kpd)

    nm = np.size(kpo.hdr) # number of kp realizations in kpo

    if nm == 1:
        models = binary_model(params, kpo.kpi, kpo.hdr)
    else:
        for i in xrange(nm):
            models[i] = binary_model(params, kpo.kpi, kpo.hdr[i])

    return models

# =========================================================================
# =========================================================================
def binary_KPD_fit_residuals(params, kpo):
    ''' Function to evaluate fit residuals, to be used in a leastsq
    fitting procedure. '''

    #test = binary_model(params, kpo.kpi, kpo.hdr)
    #test = binary_model(params, kpo.kpi, kpo.hdr[0])
    test = binary_KPD_model(kpo, params)
    err = kpo.kpd - test
    if kpo.kpe != None:
        err /= kpo.kpe
    return err

# =========================================================================
# =========================================================================
def binary_KPD_fit(kpo, p0):
    '''Performs a best binary fit search for the datasets.
    
    -------------------------------------------------------------
    p0 is the initial guess for the parameters 3 parameter vector
    typical example would be : [100.0, 0.0, 5.0].
    
    returns the full solution of the least square fit:
    - soluce[0] : best-fit parameters
    - soluce[1] : covariance matrix
    ------------------------------------------------------------- '''
    
    soluce = leastsq(binary_KPD_fit_residuals, 
                     p0, args=((kpo,)),
                     full_output=1)
    
    covar = soluce[1]
    return soluce

# =========================================================================
# =========================================================================
def chi2map_sep_con(kpo, pa=0, reduced=False, cmap=None,
                    srng=[10.0, 200.0, 100], crng=[10.0, 50.0, 100]):

    ''' Draws a 2D chi2 map of the binary parameter space for a given PA
    ---------------------------------------
    PA: position angle in degrees.
    other options:
    - reduced: (boolean) reduced chi2 or not
    - cmap: (matplotlib color map) "jet" or "prism" are good here
    - srng: range of separations [min, max, # steps]
    - crng: range of contrasts   [min, max, # steps]
    --------------------------------------- '''
    nullvar = np.var(kpo.kpd)
    seps = srng[0] + (srng[1]-srng[0]) * np.arange(srng[2])/float(srng[2])
    cons = crng[0] + (crng[1]-crng[0]) * np.arange(crng[2])/float(crng[2])

    chi2map = np.zeros((srng[2], crng[2]))

    for i,sep in enumerate(seps):
        for j,con in enumerate(cons):
            chi2map[i,j] = np.sum(
                binary_KPD_fit_residuals([sep, pa, con], kpo)**2)
        sys.stdout.write("\r(%3d/%d): sep = %.2f mas" % (i+1, srng[2], sep))
        sys.stdout.flush()
    if reduced:
        chi2map /= kpo.kpd.size
        nullvar /= kpo.kpd.size
    plt.clf()

    asp = 1./np.abs((srng[1]-srng[0])/(crng[1]-crng[0]))
    plt.imshow(chi2map, aspect=asp, cmap=cmap,
               extent=[crng[0], crng[1], srng[0], srng[1]])
    plt.ylabel("ang. sep. (mas)")
    plt.xlabel("contrast prim/sec")

    print("\n----------------------------------------")
    print("Best "+reduced*"reduced "+"chi2 = %.3f" % (chi2map.min(),))
    print(reduced*"reduced "+"chi2 for null hypothesis: %.3f" % (nullvar,))
    arg = chi2map.argmin()
    print("Obtained for sep = %.1f mas" % (seps[arg // chi2map.shape[1]]))
    print("Obtained for con = %.1f"     % (cons[arg  % chi2map.shape[1]]))
    print("----------------------------------------")
    return chi2map
    
# =========================================================================
# =========================================================================
def chi2map_sep_pa(kpo, con=10., reduced=False, cmap=None,
                    srng=[10.0, 200.0, 100], arng=[0.0, 360.0, 60]):

    ''' Draws a 2D chi2 map of the binary parameter space for a given contrast
    ---------------------------------------
    con: contrast (primary/secondary)
    other options:
    - reduced: (boolean) reduced chi2 or not
    - cmap: (matplotlib color map) "jet" or "prism" are good here
    - srng: range of separations     [min, max, # steps]
    - arng: range of position angles [min, max, # steps]
    --------------------------------------- '''

    nullvar = np.var(kpo.kpd)
    seps = srng[0] + (srng[1]-srng[0]) * np.arange(srng[2])/float(srng[2])
    angs = arng[0] + (arng[1]-arng[0]) * np.arange(arng[2])/float(arng[2])
    nkphi = kpo.kpi.nkphi

    chi2map = np.zeros((srng[2], arng[2]))

    for i,sep in enumerate(seps):
        for j,th in enumerate(angs):
            chi2map[i,j] = np.sum(
                binary_KPD_fit_residuals([sep, th, con], kpo)**2)
        sys.stdout.write("\r(%3d/%d): sep = %.2f mas" % (i+1, srng[2], sep))
        sys.stdout.flush()

    if reduced:
        chi2map /= kpo.kpd.size
        nullvar /= kpo.kpd.size
    plt.clf()

    asp = 1./np.abs((srng[1]-srng[0])/(arng[1]-arng[0]))

    plt.imshow((chi2map), aspect=asp, cmap=cmap,
               extent=[arng[0], arng[1], srng[0], srng[1]])
    plt.xlabel("Position Angle (deg)")
    plt.ylabel("Separation (mas)")

    print("\n----------------------------------------")
    print("Best "+reduced*"reduced "+"chi2 = %.3f" % (chi2map.min()))
    print(reduced*"reduced "+"chi2 for null hypothesis: %.3f" % (nullvar,))
    arg = chi2map.argmin()
    print("Obtained for sep = %.1f mas" % (seps[arg // chi2map.shape[1]]))
    print("Obtained for ang = %.1f deg" % (angs[arg  % chi2map.shape[1]]))
    print("----------------------------------------")
    return chi2map
    
# =========================================================================
# =========================================================================
def super_cal_coeffs(src, cal, model=None, regul="None"):
    ''' Determine the best combination of vectors in the "cal" array of
    calibrators for the source "src". 

    Regularisation is an option:
    - "None"         ->  no regularisation
    - anything else  ->  Tikhonov regularisation  '''

    A      = np.matrix(cal.kpd).T # vector base matrix
    ns     = A.shape[1]           # size of the vector base
    b      = src.kpd              # column vector
    if model != None: b -= model  # optional model subtraction

    if regul == "None":
        coeffs = np.dot(np.linalg.pinv(np.dot(A.T,A)),
                        np.dot(A.T,b).T)
    else:
        coeffs = np.dot(np.linalg.pinv(np.dot(A.T,A)+np.identity(ns)),
                        np.dot(A.T,b).T)
    return coeffs.T

# =========================================================================
# =========================================================================
def cal_search(src, cal, regul="None"):
    ''' Proceed to an exhaustive search of the parameter space.

    In each point of the space, the best combination of calibrator frames
    is found, and saved. Not entirely sure what this is going to be useful
    for...
    '''
    
    ns, s0, s1 = 50, 20.0, 200.0
    nc, c0, c1 = 50,  2.0, 100.0
    na, a0, a1 = 60,  0.0, 360.0
    
    seps = s0 + np.arange(ns) * (s1-s0) / ns
    angs = a0 + np.arange(na) * (a1-a0) / na
    cons = c0 + np.arange(nc) * (c1-c0) / nc

    coeffs = super_cal_coeffs(src, cal, None, "tik")
    cals = np.zeros((ns, na, nc, coeffs.size))

    for i,sep in enumerate(seps):
        for j, ang in enumerate(angs):
            for k, con in enumerate(cons):
                model  = binary_model([sep, ang, con], src.kpi, src.hdr)
                coeffs = super_cal_coeffs(src, cal, model, "tik")
                cals[i,j,k] = coeffs

    return cals

# =========================================================================
# =========================================================================
def chi2_volume(src, cal=None, regul="None"):
    ''' Proceed to an exhaustive search of the parameter space
    '''

    ns, s0, s1 = 50, 20.0, 200.0
    nc, c0, c1 = 50,  2.0, 100.0
    na, a0, a1 = 60,  0.0, 360.0
    
    seps = s0 + np.arange(ns) * (s1-s0) / ns
    angs = a0 + np.arange(na) * (a1-a0) / na
    cons = c0 + np.arange(nc) * (c1-c0) / nc

    chi2vol = np.zeros((ns,na,nc))

    sig = src.kpd
    if cal != None: sig -= cal.kpd

    for i,sep in enumerate(seps):
        for j, ang in enumerate(angs):
            for k, con in enumerate(cons):
                res = binary_multi_KPD_fit_residuals([sep, ang, con], src)
                chi2vol[i,j,k] = ((res)**2).sum()

    return chi2vol

# =========================================================================
# =========================================================================
def correlation_plot(kpo, params=[250., 0., 5.], plot_error=True, fig=None):
    '''Correlation plot between KP object and a KP binary model
    
    Parameters are:
    --------------
    - kpo: one instance of kernel-phase object
    - params: a 3-component array describing the binary (sep, PA and contrast)

    Option:
    - plot_error: boolean, errorbar or regular plot
    - fig:        id of the display for the plot
    --------------------------------------------------------------------------
    '''
    mm = np.round(np.max(np.abs(kpo.kpd)), -1)
    
    f1 = plt.figure(fig)
    sp0 = f1.add_subplot(111)
    if plot_error:
        sp0.errorbar(binary_KPD_model(kpo, params), kpo.kpd, 
                     yerr=kpo.kpe, linestyle='None')
    else:
        sp0.plot(binary_KPD_model(kpo, params), kpo.kpd, 'bo')
    sp0.plot([-mm,mm],[-mm,mm], 'g')
    sp0.axis([-mm,mm,-mm,mm])

    rms = np.std(binary_KPD_fit_residuals(params, kpo))
    msg  = "Model:\n sep = %6.2f mas" % (params[0],)
    msg += "\n   PA = %6.2f deg" % (params[1],)
    msg += "\n  con = %6.2f" % (params[2],)        
    msg += "\n(rms = %.2f deg)" % (rms,)
            
    plt.text(0.0*mm, -0.75*mm, msg, 
             bbox=dict(facecolor='white'), fontsize=14)
            
    msg = "Target: %s\nTelescope: %s\nWavelength = %.2f um" % (
        kpo.kpi.name, kpo.hdr['tel'], kpo.hdr['filter']*1e6)
            
    plt.text(-0.75*mm, 0.5*mm, msg,
              bbox=dict(facecolor='white'), fontsize=14)
    
    plt.xlabel('Data kernel-phase signal (deg)')
    plt.ylabel('Kernel-phase binary model (deg)')
    plt.draw()
    return None

# =========================================================================
# =========================================================================

def nested(kpo, params=[30.0, 250.0, 0.0, 360.0, 1.5, 5.0], 
           npts=100, nstp = 3000, njmp=20, mode='kp'):
    ''' Nested sampling routine.
    -------------------------------------------------------------
    - params [sepmin, sepmax, anglemin, anglemax, cmin, cmax]
    - npts: number of points
    - nstp: number of steps
    - njmp: number of jumps

    Call with argument mode = 'vis' to use vis fitting etc.

    This is sufficiently modular that with a different loglikelihood function
    you can use it straight out of the box.

    Note: this function requires properly calibrated observables.
    ------------------------------------------------------------- '''
    plt.figure(0, figsize=(10,5))

    [smin, smax, amin, amax, cmin, cmax] = params

    # Generate npts active points using different priors
    angs = amin + (amax-amin) * rand(npts)               # Uniform
    seps = np.exp(rand(npts) * np.log(smax/smin)) * smin # Jeffreys
    cons = np.exp(rand(npts) * np.log(cmax/cmin)) * cmin # Jeffreys

    L = np.zeros(npts)

    rejected = np.zeros((nstp, 3))
    rejL     = np.zeros(nstp)

    refrate = 200
    progress = refrate # counter for displays

    for k in range(npts): # calculate likelihood
        L[k] = loglikelihood(kpo, seps[k], angs[k], cons[k], mode=mode)

    for j in range(nstp): # loop for iterations
        # store rejected binary parameters and re-sort L

        jumps = (smax-smin) / 200. # step rescaling parameters
        jumpa = (amax-amin) / 200.
        jumpc = (cmax-cmin) / 200.

        throw = (L==L.min())
            
        rejected[j,0] = seps[throw][0]
        rejected[j,1] = angs[throw][0]
        rejected[j,2] = cons[throw][0]

        rejL[j] = L[throw][0]
        
        args = np.argsort(L)
        seps, angs, cons, L = seps[args], angs[args], cons[args], L[args]

        naccept = 0.0 # initialise for coming loop
        nreject = 0.0
        l = 0
        trysd, tryad, trycd = 0,0,0

        choose = np.random.randint(0, npts) # pick a random index

        starts, starta, startc = seps[choose],angs[choose],cons[choose]
        news, newa, newc = starts, starta, startc
        newl = loglikelihood(kpo, starts, starta, startc, mode=mode)
        
        while l < njmp: # random walk to generate a new active point

            trya = np.mod(newa + jumpa*randn(), amax)
            trys = news + jumps * randn()
            tryc = newc + jumpc * randn()

            like = loglikelihood(kpo, trys, trya, tryc, mode=mode)
            
            # accept and move on
            if (like > L[0]) and (smin < trys < smax) and (cmin < tryc < cmax): 
                naccept += 1.0
                news, newa, newc, newl = trys, trya, tryc, like
            else: # dwell on it
                nreject += 1.0

            #adjust jumps - factors from John Skilling's example

            if naccept > nreject: # jump further if you can afford to
                jumps *= np.exp(1/naccept)
                jumpa *= np.exp(1/naccept)
                jumpc *= np.exp(1/naccept)
                
            else: # don't jump as far if you're making it worse too often
                jumps /= np.exp(1/nreject)
                jumpa /= np.exp(1/nreject)
                jumpc /= np.exp(1/nreject)
                
            l += 1

            if (l >= njmp and naccept == 0): # start somewhere else
                l = 0
                choose = np.random.randint(0, npts) # pick a random index
                news,newa,newc = seps[choose],angs[choose],cons[choose]

        msg = '\rIt %4d, accept. %.2f (Sep,PA,con) = (%.2f, %.2f, %.2f)' 
        sys.stdout.write(msg % (j, naccept, news, newa, newc))
        sys.stdout.flush()

        seps[0], angs[0], cons[0], L[0] = news, newa, newc, newl

        if progress == refrate: # make a plot every refrate iterations
            plt.clf()
            plt.subplot(121)
            plt.scatter(seps,angs)
            plt.axis([smin,smax,amin,amax])
            plt.xlabel('Separation (mas)')
            plt.ylabel('Position Angle (degrees)')
            plt.title('Active Points')

            plt.subplot(122)
            plt.scatter(seps,cons)
            plt.axis([smin,smax,cmin,cmax])
            plt.xlabel('Separation (mas)')
            plt.ylabel('Contrast Ratio')
            plt.title('Active Points')
            plt.draw()
            
            progress = 0
        else:
            progress += 1

    '''-----------------------------------------------
    We can now integrate these results - we choose the
    rectangle rule wi = xi-x(i-1)
    -----------------------------------------------'''

    x =  np.exp(-np.arange(nstp)/float(npts))

    seps = rejected[:,0]
    angs = rejected[:,1]
    cons = rejected[:,2]
    logpre = 0 #workaround for the moment

    xx = np.zeros(len(x)+2) #dummy padded array for reflecting bcs
    xx[0]    = 2-x[0]
    xx[-1]   = x[-1]
    xx[1:-1] = x

    w = 0.5*np.abs(np.array([xx[j-1]-xx[j+1] for j in range(1,len(xx)-1)]))
    logw = np.log(w)

    E = np.sum(w*np.exp(rejL-rejL.max()))
    logE = np.log(E) + rejL.max()

    logpdf = rejL+logw-logE

    logE += logpre # get the prefactor back in

    pdf = np.exp(logpdf)
            
    H = np.sum(pdf*(rejL-logE-logpre))

    dlogE = np.sqrt(H/nstp)

    sepmean = np.sum(pdf*seps)
    dsep = np.sqrt(np.sum(pdf * (seps-sepmean)**2))

    anglemean = np.sum(pdf*angs)
    dangle = np.sqrt(np.sum(pdf*(angs-anglemean)**2))

    cmean = np.sum(pdf*cons)
    dc = np.sqrt(np.sum(pdf*(cons-cmean)**2))

    output = [sepmean,dsep,anglemean,dangle,cmean,dc,logE,dlogE]

    return output

# =========================================================================
# =========================================================================

def loglikelihood(kpo, sep, angle, contrast, mode='kp'):
    '''Define the chi-squared function - omits logpre!
    Takes a structure a and parameters sep, angle and contrast.
    Pass the mode 'vis' to use visibility fitting'''

    if mode == 'kp':
        modl_ker = binary_model([sep,angle,contrast],kpo.kpi, kpo.hdr)

        chisquared = np.sum(((modl_ker - kpo.kpd)/(kpo.kpe))**2)
    elif mode == 'vis':
        test = vis2_binary(kpo.kpi.uv[:,0],kpo.kpi.uv[:,1], kpo.hdr['filter'],
                           [sep, angle, contrast])
        chisquared = np.sum(((test - kpo.v2)/(kpo.vis_error))**2)
    else:
        print 'Invalid mode'

    #logpre = a.nkphi*np.sum(-1/2 * np.log(2*np.pi) - np.log(a.kpe))
    #logpre = 0 
    #like = logpre-chisquared/2
    like = -chisquared/2
    
    return like
