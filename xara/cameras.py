''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data 
    structures, using the theory described in the two following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.

    This file contains the tools that can extract the relevant information
    out of FITS header files.
    -------------------------------------------------------------------- '''

import numpy as np

# =========================================================================
# =========================================================================

def get_keck_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This version is adapted to handle NIRC2 data. '''
    data = {
        'tel'    : hdr['TELESCOP'],        # telescope
        'pscale' : 10.0,                   # NIRC2 narrow plate scale (mas)
        'fname'  : hdr['FILENAME'],        # original file name
        'odate'  : hdr['DATE-OBS'],        # UTC date of observation
        'otime'  : hdr['UTC'     ],        # UTC time of observation
        'tint'   : hdr['ITIME'   ],        # integration time (sec)
        'coadds' : hdr['COADDS'  ],        # number of coadds
        'RA'     : hdr['RA'      ],        # right ascension (deg)
        'DEC'    : hdr['DEC'     ],        # declination (deg)
        'filter' : hdr['CENWAVE' ] * 1e-6, # central wavelength (meters)
        # P.A. of the frame (deg) (formula from M. Ireland)
        'orient' : 360+hdr['PARANG']+hdr['ROTPPOSN']-hdr['EL']-hdr['INSTANGL']
        }
    print("parang = %.2f, rotpposn = %.2f, el=%.2f, instangl=%.2f" % \
        (hdr['PARANG'],hdr['ROTPPOSN'],hdr['EL'],hdr['INSTANGL']))
    return data

# =========================================================================
# =========================================================================
def get_nic1_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This version is adapted to handle NICMOS1 data. '''
    data = {
        'tel'    : hdr['TELESCOP'],         # telescope
        'pscale' : 43.1,                    # HST NIC1 plate scale (mas)
        'fname'  : hdr['FILENAME'],         # original file name
        'odate'  : hdr['DATE-OBS'],         # UTC date of observation
        'otime'  : hdr['TIME-OBS'],         # UTC time of observation
        'tint'   : hdr['EXPTIME' ],         # integration time (sec)
        'coadds' : 1,                       # as far as I can tell...
        'RA'     : hdr['RA_TARG' ],         # right ascension (deg)
        'DEC'    : hdr['DEC_TARG'],         # declination (deg)
        'filter' : hdr['PHOTPLAM'] * 1e-10, # central wavelength (meters)
        'orient' : hdr['ORIENTAT'] # P.A. of image y axis (deg e. of n.)
        }
    return data

# =========================================================================
# =========================================================================
def get_pharo_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This version is adapted to handle PHARO data. '''

    data = {
        'tel'      : hdr['TELESCOP'],         # telescope
        'pscale'   : 25.2,                    # HST NIC1 plate scale (mas)
        'odate'    : hdr['DATE-OBS'],         # UTC date of observation
        'otime'    : hdr['TIME-OBS'],         # UTC time of observation
        'tint'     : hdr['T_INT' ],           # integration time (sec)
        'coadds'   : 1,                       # as far as I can tell...
        'RA'       : hdr['CRVAL1'],           # right ascension (deg)
        'DEC'      : hdr['CRVAL2'],           # declination (deg)
        'filter'   : np.nan, # place-holder   # central wavelength (meters)
        'filtname' : hdr['FILTER'],           # Filter name
        'grism'    : hdr['GRISM'],            # additional filter/nd
        'pupil'    : hdr['LYOT'],             # Lyot-pupil wheel position
        'orient'   : hdr['CR_ANGLE']          # Cassegrain ring angle
        }

    if 'H'       in data['filtname'] : data['filter'] = 1.635e-6
    if 'K'       in data['filtname'] : data['filter'] = 2.196e-6
    if 'CH4_S'   in data['filtname'] : data['filter'] = 1.570e-6
    if 'K_short' in data['filtname'] : data['filter'] = 2.145e-6
    if 'BrG'     in data['filtname'] : data['filter'] = 2.180e-6
    if 'FeII'     in data['grism']   : data['filter'] = 1.648e-6
    
    if np.isnan(data['filter']):
        print("Filter configuration un-recognized. Analysis will fail.")
    return data

# =========================================================================
# =========================================================================
def get_simu_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This is a special version for simulated data. '''
    data = {
        'tel'    : hdr['TELESCOP'],        # telescope
        'pscale' : 2.0,#43.1,                   # simulation plate scale (mas)
        'fname'  : "simulation",           # original file name
        'odate'  : "Jan 1, 2000",          # UTC date of observation
        'otime'  : "0:00:00.00",           # UTC time of observation
        'tint'   : 1.0,                    # integration time (sec)
        'coadds' : 1,                      # number of coadds
        'RA'     : 0.000,                  # right ascension (deg)
        'DEC'    : 0.000,                  # declination (deg)
        'filter' : 1.6* 1e-6,              # central wavelength (meters)
        'orient' : 0.0                     # P.A. of the frame (deg)
        }
    return data
