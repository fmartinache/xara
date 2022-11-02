from typing import Optional, Union
from pathlib import Path
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList


def split_file_path(fpath: Union[Path, str]):
    fpath = Path(fpath)
    suffixes = fpath.suffixes[-2:]
    n_suffixes = len(suffixes)
    fext = "".join(suffixes)
    pdir = str(fpath.parent)
    fstem = fpath
    for _ in range(n_suffixes):
        fstem = Path(fstem.stem)
    fstem = str(fstem)

    return pdir, fstem, fext


def get_output_base(input_path: Union[Path, str], output_dir: Optional[Union[Path, str]] = None):
    input_path = Path(input_path)
    parent_dir, fstem, _ = split_file_path(input_path)
    if output_dir is None:
        output_base = Path(parent_dir) / fstem
    else:
        output_base = Path(output_dir) / fstem

    return str(output_base)


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
        pdir, fstem, fext = split_file_path(fpath)
        # Handle potential compressed files (e.g. fits.gz)
        basename = fstem + suffix + fext

        if dirpath is None:
            open_path = Path(pdir) / basename
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
    return data, erro, sx, sy
