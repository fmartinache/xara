''' --------------------------------------------------------------------
XARA: a package for eXtreme Angular Resolution Astronomy
--------------------------------------------------------------------
This file is used to store older pieces of code that are no longer
part of the really useful part of the code. These are kept here for
reference and to provide a buffer before eventual final suppression
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

    # =========================================================================
    # =========================================================================
    def __extract_KPD_NIRISS(self, fnames, target=None,
                             recenter=False, wrad=None, method="LDFT1"):

        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))

        cvis = []    # complex visibility
        kpdata = []  # Kernel-phase data
        detpa = []   # detector position angle
        mjdate = []  # modified Julian date

        hdul = fits.open(fnames[0])
        xsz = hdul[0].header['NAXIS1']              # image x-size
        ysz = hdul[0].header['NAXIS2']              # image y-size
        pscale = hdul[0].header['PIXELSCL'] * 1e3   # plate scale (mas)
        cwavel = hdul[0].header['WAVELEN']          # central wavelength
        imsize = xsz                                # chop image
        m2pix = core.mas2rad(pscale)*imsize/cwavel  # Fourier scaling
        tdiam = 6.5                                 # telescope "diameter" (m)
        spix = core.rad2mas(cwavel/tdiam)/pscale    # image sampling (pixels)

        if target is None:
            try:
                target = hdul[0].header['TARGNAME']    # Target name
            except KeyError:
                target = "NONAME_TARGET"

        hdul.close()

        for ii in range(nf):
            hdul = fits.open(fnames[ii])

        nslice = 1
        if hdul[0].header['NAXIS'] == 3:
            nslice = hdul[0].header['NAXIS3']
        data = hdul[0].data.reshape((nslice, ysz, xsz))

        # ---- prepare the super-Gaussian apodization mask ----
        self.sgmask = None
        if wrad is not None:
            self.sgmask = core.super_gauss(ysz, xsz, ysz/2, xsz/2, wrad)

        # ---- extract the Fourier data ----
        for jj in range(nslice):
            if recenter is True:
                (x0, y0) = core.determine_origin(data[jj], mask=self.sgmask,
                                                 algo="BCEN", verbose=False,
                                                 wmin=2.0*spix)
                dy, dx = (y0-ysz/2), (x0-xsz/2)

            img = data[jj]

            if self.sgmask is not None:  # apodization mask before extraction
                if recenter is True:
                    img *= np.roll(
                        np.roll(self.sgmask, int(round(dx)), axis=1),
                        int(round(dy)), axis=0)
                else:
                    img *= self.sgmask

            temp = self.extract_cvis_from_img(img, m2pix, method)

            if recenter is True:  # centering error compensation on cvis
                uvc = self.kpi.UVC * self.M2PIX
                corr = np.exp(i2pi * uvc.dot(np.array([dx, dy])/float(ysz)))
                temp *= corr

            cvis.append(temp)
            kpdata.append(self.kpi.KPM.dot(np.angle(temp)))

            print("\rFile %s, slice %3d" % (fnames[ii], jj+1),
                  end="", flush=True)

            mjdate.append(0.0)  # currently not available in sim data
            detpa.append(0.0)   # currently not available in sim data
        hdul.close()

        self.CWAVEL = cwavel
        self.PSCALE = pscale
        self.WRAD = wrad

        return target, cvis, kpdata, detpa, mjdate

    # =========================================================================
    # =========================================================================
    def __extract_KPD_HST(self, fnames, target=None,
                          recenter=True, wrad=None, method="LDFT1"):

        # @FIXME!!
        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))

        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date

        hdul   = fits.open(fnames[0])
        xsz    = hdul['SCI'].header['NAXIS1']      # image x-size
        ysz    = hdul['SCI'].header['NAXIS2']      # image y-size
        pscale = 43.1                              # plate scale (mas)
        cwavel = hdul[0].header['PHOTPLAM']*1e-10  # central wavelength
        isz    = 128                               # chop image
        m2pix  = core.mas2rad(pscale)*isz/cwavel   # Fourier scaling
        tdiam  = 2.4                               # telescope diameter (m)
        spix   = core.rad2mas(cwavel/tdiam)/pscale # image sampling (pixels)
            
        if target is None:
            target = hdul[0].header['TARGNAME']    # Target name
        hdul.close()

        # prepare the super-Gaussian apodization mask
        self.sgmask = None
        if wrad is not None:
            self.sgmask  = core.super_gauss(ysz, xsz, ysz/2, xsz/2, wrad)

        for ii in range(nf):
            hdul = fits.open(fnames[ii])

        nslice = 1
        if hdul['SCI'].header['NAXIS'] == 3:
            nslice = hdul['SCI'].header['NAXIS3']

        data = hdul['SCI'].data.reshape((nslice, ysz, xsz))
                
        # ---- extract the Fourier data ----
        for jj in range(nslice):
            # images must first be chopped down to reasonable size
            (x0, y0) = core.determine_origin(data[jj], mask=self.sgmask,
                                             algo="BCEN", verbose=False,
                                             wmin=2.0*spix)

            x1, y1 = x0-isz/2, y0-isz/2
            img = data[jj,y1:y1+isz, x1:x1+isz] # image is now (isz x isz)
            dy, dx   = (y0-ysz/2), (x0-xsz/2)

            self.sgmask = core.super_gauss(isz, isz, isz/2, isz/2, wrad)
            (x0, y0) = core.determine_origin(img, mask=self.sgmask,
                                             algo="BCEN", verbose=False,
                                             wmin=2.0*spix)
            
            #img = core.recenter(data[jj], sg_rad=50, verbose=False)
            img = img[192:320,192:320] # from 512x512 -> 128x128
            temp = self.extract_cvis_from_img(img, m2pix, method)
            cvis.append(temp)
            kpdata.append(self.kpi.KPM.dot(np.angle(temp)))
            print("File %s, slice %2d" % (fnames[ii], jj+1))
            
            mjdate.append(hdul['SCI'].header['ROUTTIME'])
            detpa.append(hdul[0].header['ORIENTAT'])
        hdul.close()

        self.CWAVEL = cwavel
        self.PSCALE = pscale
        self.WRAD   = wrad

        return target, cvis, kpdata, detpa, mjdate

    # =========================================================================
    # =========================================================================
    def __extract_KPD_VLT_SPHERE(self, fnames, target=None,
                                 recenter=False, wrad=None, method="LDFT1"):
        
        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))
        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date
        
        hdul   = fits.open(fnames[0])
        xsz    = hdul[0].header['NAXIS1']
        ysz    = hdul[0].header['NAXIS2']
        cwavel = 0.8168 * 1e-6 # FIXME
        pscale = 7.0 # FIXME (ZIMPOL)
        m2pix  = core.mas2rad(pscale)*xsz/cwavel

        if target is None:
            target = hdul[0].header['HIERARCH ESO OBS TARG NAME']
        hdul.close()

        index = 0
        for ii in range(nf):
            hdul = fits.open(fnames[ii])

            nslice = 1
            if hdul[0].header['NAXIS'] == 3:
                nslice = hdul[0].header['NAXIS3']
            data = hdul[0].data.reshape((nslice, ysz, xsz))

            # ---- extract the Fourier data ----
            for jj in range(nslice):
                if recenter:
                    img = core.recenter(data[jj], sg_rad=150, verbose=False)
                else:
                    img = data[jj]

                index += 1
                temp = self.extract_cvis_from_img(img, m2pix, method)
                cvis.append(temp)
                kpdata.append(self.kpi.KPM.dot(np.angle(temp)))
                print("File %s, slice %2d" % (fnames[ii], jj+1))
                
                mjdate.append(hdul[0].header['MJD-OBS'])
            # --- detector position angle read globally ---
            detpa.append(0.0)#hdul[0].data['pa'])

            hdul.close()        

        self.CWAVEL = cwavel
        self.PSCALE = pscale
        self.WRAD   = wrad

        return target, cvis, kpdata, detpa, mjdate

    # =========================================================================
    # =========================================================================
    def __extract_KPD_NACO(self, fnames, target=None,
                           recenter=True, wrad=None, method="LDFT1"):
        
        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))
        
        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date

        hdul   = fits.open(fnames[0])
        xsz    = hdul[0].header['NAXIS1']
        ysz    = hdul[0].header['NAXIS2']
        pscale = hdul[0].header['HIERARCH ESO INS PIXSCALE']*1000.
        cwavel = hdul[0].header['HIERARCH ESO INS CWLEN'] * 1e-6
        imsize = hdul[0].header['NAXIS1']
        m2pix  = core.mas2rad(pscale)*imsize/cwavel
        
        if target is None:
            target = hdul[0].header['HIERARCH ESO OBS TARG NAME']  # Target name
        hdul.close()

        index = 0
        for ii in range(nf):
            hdul = fits.open(fnames[ii])
            sys.stdout.write("File %s" % (fnames[ii],))
            sys.stdout.flush()

            nslice = 1
            if hdul[0].header['NAXIS'] == 3:
                nslice = hdul[0].header['NAXIS3']
            data = hdul[0].data.reshape((nslice, ysz, xsz))

            # ---- extract the Fourier data ----
            for jj in range(nslice):
                if recenter:
                    img = core.recenter(data[jj], sg_rad=50, verbose=False)
                else:
                    img = data[jj]

                index += 1
                temp = self.extract_cvis_from_img(img, m2pix, method)
                cvis.append(temp)
                kpdata.append(self.kpi.KPM.dot(np.angle(temp)))
                sys.stdout.write("slice %2d" % (jj+1,))
                sys.stdout.flush()
                
                mjdate.append(hdul[0].header['MJD-OBS'])
            # --- detector position angle read globally ---
            detpa.append(hdul[1].data['pa'])

            hdul.close()
                
        self.CWAVEL = cwavel
        self.PSCALE = pscale
        self.WRAD   = wrad

        return target, cvis, kpdata, detpa, mjdate
    
    # =========================================================================
    # =========================================================================
    def __extract_KPD_Keck(self, fnames, target=None,
                           recenter=False, wrad=None, method="LDFT1"):

        nf = fnames.__len__()
        print("%d data fits files will be opened" % (nf,))
        
        cvis   = [] # complex visibility
        kpdata = [] # Kernel-phase data
        detpa  = [] # detector position angle
        mjdate = [] # modified Julian date

        hdul   = fits.open(fnames[0])
        xsz    = hdul[0].header['NAXIS1']           # image x-size
        ysz    = hdul[0].header['NAXIS2']           # image y-size
        pscale = 10.0                               # plate scale (mas)
        cwavel = hdul[0].header['CENWAVE'] * 1e-6   # central wavelength
        imsize = hdul[0].header['NAXIS1']           # image size (pixls)
        m2pix  = core.mas2rad(pscale)*imsize/cwavel # Fourier scaling
        tdiam  = 10.9                               # telescope diameter (m)
        spix   = core.rad2mas(cwavel/tdiam)/pscale  # image sampling (pixels)
        
        if target is None:
            target = hdul[0].header['OBJECT']      # Target name
        hdul.close()

        # prepare the super-Gaussian apodization mask
        self.sgmask = None
        if wrad is not None:
            self.sgmask  = core.super_gauss(ysz, xsz, ysz/2, xsz/2, wrad)

        index = 0

        for ii in range(nf):
            hdul = fits.open(fnames[ii])

            nslice = 1
            if hdul[0].header['NAXIS'] == 3:
                nslice = hdul[0].header['NAXIS3']
            data = hdul[0].data.reshape((nslice, ysz, xsz))

            # ---- extract the Fourier data ----
            for jj in range(nslice):
                if recenter is True:
                    (x0, y0) = core.determine_origin(data[jj], mask=self.sgmask,
                                                     algo="BCEN", verbose=False,
                                                     wmin=2.0*spix)
                    dy, dx   = (y0-ysz/2), (x0-xsz/2)

                index += 1
                img = data[jj]
                
                if self.sgmask is not None: # use apodization mask before extraction
                    if recenter is True:
                        img *= np.roll(np.roll(self.sgmask, int(round(dx)), axis=1),
                                       int(round(dy)), axis=0)
                    else:
                        img *= self.sgmask
                        
                temp = self.extract_cvis_from_img(img, m2pix, method)
                
                if recenter is True: # centering error compensation on cvis
                    uvc  = self.kpi.UVC * self.M2PIX
                    corr = np.exp(i2pi * uvc.dot(np.array([dx, dy])/float(ysz)))
                    temp *= corr

                cvis.append(temp)
                kpdata.append(self.kpi.KPM.dot(np.angle(temp)))
                sys.stdout.write("\rFile %s, slice %2d" % (fnames[ii], jj+1))
                sys.stdout.flush()
                
                mjdate.append(hdul[0].header['MJD-OBS'])

            # --- detector position angle read globally ---
            detpa.append(hdul[1].data['pa'])

            hdul.close()

        self.CWAVEL = cwavel
        self.PSCALE = pscale
        self.WRAD   = wrad
                
        return target, cvis, kpdata, detpa, mjdate

    # =========================================================================
    # =========================================================================
    def extract_KPD(self, path, target=None,
                    recenter=True, wrad=None, method="LDFT1"):
        ''' extract kernel-phase data from one or more files (use regexp).

        ---------------------------------------------------------------------
        If the path leads to a fits data cube, or to multiple single frame
        files, the extracted kernel-phases are consolidated into a unique
        KPD array.

        The details of the extraction procedures will depend on the origin
        of the file and the way the header keywords are organized.

        Parameters:
        ----------
        - path     : path to one or more data fits files
        - target   : a 8 character string ID (default: get from fits file)

        - recenter : fine-centers the frame (default = True)
        - wrad     : Super-Gaussian window radius in pixels (default=None)
        - method   : string describing the extraction method. Default="LDFT1"
          + "LDFT1" : one sided DFT (recommended: most flexible)
          + "LDFT2" : two-sided DFT (FASTER, but for cartesian grids only)
          + "FFT"   : the old fashioned way - not as accurate !!
        ---------------------------------------------------------------------
        '''
        fnames   = sorted(glob.glob(path))
        hdul     = fits.open(fnames[0])
        try:
            tel_name = hdul[0].header['TELESCOP']
        except KeyError:
            tel_name = "JWST" # seems to be the exception?
            
        # ------------------------------------------------------------
        if 'Keck II' in tel_name:
            print("The data comes from Keck")
            
            tgt, cvis, kpd, detpa, mjdate = self.__extract_KPD_Keck(
                fnames, target=target, recenter=recenter, wrad=wrad,
                method=method)
        # ------------------------------------------------------------
        elif 'HST' in tel_name:
            print("The data comes from HST")

            tgt, cvis, kpd, detpa, mjdate = self.__extract_KPD_HST(
                fnames, target=target, recenter=recenter, wrad=wrad,
                method=method)
        # ------------------------------------------------------------
        elif 'VLT' in tel_name:
            if 'SPHERE' in hdul[0].header['INSTRUME']:
                print("The data comes from VLT/SPHERE")
                tgt, cvis, kpd, detpa, mjdate = self.__extract_KPD_VLT_SPHERE(
                    fnames, target=target, recenter=recenter, wrad=wrad,
                    method=method)
                
            if 'NAOS+CONICA' in hdul[0].header['INSTRUME']:
                print("The data comes from VLT/NACO")
                tgt, cvis, kpd, detpa, mjdate = self.__extract_KPD_NACO(
                    fnames, target=target, recenter=recenter, wrad=wrad,
                    method=method)
        # ------------------------------------------------------------
        elif 'JWST' in tel_name:
            # eventually, NIRCAM should be one option here
            #if 'NIRISS' in hdul[0].header['INSTRUME']:
            print("The data comes from JWST/NIRISS")
            tgt, cvis, kpd, detpa, mjdate = self.__extract_KPD_NIRISS(
                fnames, target=target, recenter=recenter, wrad=wrad,
                method=method)

        # ------------------------------------------------------------
        else:
            print("Extraction for %s not implemented." % (tel_name,))
            return
        # ------------------------------------------------------------
        hdul.close()

        self.TARGET.append(tgt)
        self.CVIS.append(np.array(cvis))
        self.KPDT.append(np.array(kpd))
        self.DETPA.append(np.array(detpa).flatten())
        self.MJDATE.append(np.array(mjdate))
        return
