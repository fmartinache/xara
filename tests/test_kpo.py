from pathlib import Path
import numpy as np
from astropy.io import fits
import pytest
from xara.kpo import KPO
from xara.kpi import KPI


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def tmp_path(data_dir: Path):
    return data_dir / "tmp_test_kpfits.fits"


@pytest.fixture
def pharo_kpo(data_dir: Path) -> KPO:
    model_path = data_dir / "PHARO/p3k_med_grey_model.fits"
    return KPO(fname=str(model_path))


def test_kpo_from_pupil_fits(pharo_kpo):
    assert isinstance(pharo_kpo.CWAVEL, list)
    assert isinstance(pharo_kpo.DETPA, list)
    assert isinstance(pharo_kpo.CVIS, list)
    assert isinstance(pharo_kpo.KPDT, list)

    assert len(pharo_kpo.CWAVEL) == 0


def test_kpo_from_kpfits():
    kpfits_path = (
        Path(__file__).parent / "./data/jw01093011001_03103_00001_nis_emp_kpfits.fits"
    )

    with pytest.raises(ValueError, match="Extracting data from file"):
        KPO(fname=kpfits_path)

    kpo = KPO(fname=kpfits_path, input_format="KPFITS")
    hdul = fits.open(kpfits_path)

    # TODO:: Test KPI bits in a kpi test and compare here with KPI?
    # TODO: CWAVEL, DETPA, etc.
    kpdt_arr = np.concatenate(kpo.KPDT)  # Collapse dim 0 of "multiple cubes"
    cvis_arr = np.concatenate(kpo.CVIS)
    kpsig_arr = np.concatenate(kpo.KPSIG)
    kpcov_arr = np.concatenate(kpo.KPCOV)
    cvis_from_hdul = hdul["CVIS-DATA"].data[0] + 1j * hdul["CVIS-DATA"].data[1]
    assert isinstance(kpo.KPDT, list)
    # Flag shape if unequal -> easier to debug
    assert hdul["KP-DATA"].data[:, 0].shape == kpdt_arr.shape
    assert cvis_from_hdul[:, 0].shape == cvis_arr.shape
    # Then check all is equal-ish
    np.testing.assert_allclose(hdul["KP-DATA"].data[:, 0], kpdt_arr)
    np.testing.assert_allclose(cvis_from_hdul[:, 0], cvis_arr)
    np.testing.assert_allclose(hdul["KP-SIGM"].data[:, 0], kpsig_arr)
    np.testing.assert_allclose(hdul["KP-COV"].data[:, 0], kpcov_arr)
    # TODO: MJDATE?


def test_kpo_to_kpfits(data_dir, tmp_path):
    kpfits_path = data_dir / "jw01093011001_03103_00001_nis_emp_kpfits.fits"
    if tmp_path.exists():
        raise FileExistsError(f"Wanted to create {tmp_path} for a test but it exists")
    kpo = KPO(fname=kpfits_path, input_format="KPFITS")
    real_hdul = fits.open(kpfits_path)
    kpo_hdul = kpo.save_as_kpfits(tmp_path)
    tmp_path.unlink()

    # TODO: Just iterate?
    # TODO: CWAVEL, DETPA, etc.
    np.testing.assert_allclose(real_hdul["KP-DATA"].data, kpo_hdul["KP-DATA"].data)
    np.testing.assert_allclose(real_hdul["CVIS-DATA"].data, kpo_hdul["CVIS-DATA"].data)
    np.testing.assert_allclose(real_hdul["KP-SIGM"].data, kpo_hdul["KP-SIGM"].data)
    np.testing.assert_allclose(real_hdul["KP-COV"].data, kpo_hdul["KP-COV"].data)


def test_kpo_extract_cube(pharo_kpo: KPO, data_dir: Path, tmp_path: Path):
    with fits.open(data_dir / "PHARO/tgt_cube.fits") as hdul:
        tgt_cube = hdul[0].data
    pscale = 25.0  # plate scale of the image in mas/pixels
    wl = 2.145e-6  # central wavelength in meters (Hayward paper)
    assert len(pharo_kpo.KPDT) == 0
    pharo_kpo.extract_KPD_single_cube(
        tgt_cube, pscale, wl, target="alpha Ophiuchi", recenter=True
    )
    assert len(pharo_kpo.KPDT) == 1
    assert pharo_kpo.KPDT[0].ndim == 2
    assert pharo_kpo.KPDT[0].shape == (tgt_cube.shape[0], pharo_kpo.kpi.KPM.shape[0])

    hdul = pharo_kpo.save_as_kpfits(tmp_path)
    tmp_path.unlink()
    saved_kp = hdul["KP-DATA"].data
    cvis_from_hdul = hdul["CVIS-DATA"].data[0] + 1j * hdul["CVIS-DATA"].data[1]
    # TODO: Sig and cov? They are not generated within xara for now so should be OK? Not affected by extraction, only "round-tripped"
    np.testing.assert_allclose(hdul["KP-DATA"].data, np.expand_dims(pharo_kpo.KPDT[0], axis=1))
    np.testing.assert_allclose(cvis_from_hdul, np.expand_dims(pharo_kpo.CVIS[0], axis=1))


def test_kpo_extract_frame(pharo_kpo: KPO, data_dir: Path, tmp_path: Path):
    with fits.open(data_dir / "PHARO/tgt_cube.fits") as hdul:
        tgt_cube = hdul[0].data
    pscale = 25.0  # plate scale of the image in mas/pixels
    wl = 2.145e-6  # central wavelength in meters (Hayward paper)
    assert len(pharo_kpo.KPDT) == 0
    for tgt_frame in tgt_cube:
        pharo_kpo.extract_KPD_single_frame(
            tgt_frame, pscale, wl, target="alpha Ophiuchi", recenter=True
        )
    assert len(pharo_kpo.KPDT) == len(tgt_cube)
    assert pharo_kpo.KPDT[0].ndim == 1
    assert pharo_kpo.KPDT[0].shape == (pharo_kpo.kpi.KPM.shape[0],)

    hdul = pharo_kpo.save_as_kpfits(tmp_path)
    tmp_path.unlink()
    cvis_from_hdul = hdul["CVIS-DATA"].data[0] + 1j * hdul["CVIS-DATA"].data[1]
    # TODO: Sig and cov?
    np.testing.assert_allclose(hdul["KP-DATA"].data, np.expand_dims(pharo_kpo.KPDT, axis=1))
    np.testing.assert_allclose(cvis_from_hdul, np.expand_dims(pharo_kpo.CVIS, axis=1))


# TODO: Test saving to kpfits with both extractions
