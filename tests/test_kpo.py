from pathlib import Path
from xara.kpo import KPO

def test_kpo_from_fname():
    model_path = Path(__file__) / "data/PHARO/p3k_med_grey_model.fits"
    kpo = KPO(fname=str(model_path))

    assert isinstance(kpo.CWAVEL, list)
    assert isinstance(kpo.DETPA, list)
    assert isinstance(kpo.CVIS, list)
    assert isinstance(kpo.KPDT, list)

    assert len(kpo.CWAVEL) == 0
