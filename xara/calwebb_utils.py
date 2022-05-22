from typing import Optional, Union
from pathlib import Path
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList


def open_fits(
    original_fpath: Union[Path, str],
    suffix: Optional[str] = None,
    dirpath: Optional[Union[str, Path]] = None,
):
    fpath = Path(original_fpath)

    suffix = suffix or ""

    if suffix == "":
        open_path = fpath
    else:
        # Handle potential compressed files (e.g. fits.gz)
        suffixes = fpath.suffixes[-2:]
        n_suffixes = len(suffixes)
        fext = "".join(suffixes)
        pdir = fpath.parent
        fstem = fpath
        for _ in range(n_suffixes):
            fstem = Path(fstem.stem)
        fstem = str(fstem)
        basename = fstem + suffix + fext

        if dirpath is None:
            open_path = pdir / basename
        else:
            open_path = Path(dirpath) / basename

    return fits.open(open_path)

def get_data(hdul: HDUList):
    try:
        data = hdul['SCI-MOD'].data
        erro = hdul['ERR-MOD'].data
    except KeyError:
        data = hdul['SCI'].data
        erro = hdul['ERR'].data

    if data.ndim not in [2, 3]:
        raise ValueError('Only implemented for 2D image/3D data cube')
    sy, sx = data.shape[-2:]
    return data, erro, sy, sx
