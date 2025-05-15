import numpy as np

from xara import core


def test_xyic():
    ny, nx = 10, 15
    yx = core._xyic(ny, nx)

    ## Check shapes are what we expect
    assert yx[0].shape == yx[1].shape
    assert yx[0].shape == (ny, nx)

    # Check that between does what we expect
    yx_between = core._xyic(ny, nx, between_pix=True)
    np.testing.assert_array_equal(yx_between[0], yx[0] + 0.5)
    np.testing.assert_array_equal(yx_between[1], yx[1] + 0.5)

    # Cross-check with meshgrid
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    xy = np.meshgrid(x, y)
    np.testing.assert_array_equal(xy[1], yx[0])
    np.testing.assert_array_equal(xy[0], yx[1])


def test_dist():
    ny, nx = 10, 20
    d = core._dist(ny, nx)
    # We shift by half so will have absolute max of n//2
    max_expect = np.sqrt((ny // 2) ** 2 + (nx // 2) ** 2)

    assert d.max() == max_expect
    assert d.min() == 0

    # We shift by half and add 0.5 so (n-1)/2 on each side.
    # And smallest offset is 0.5 each side
    d_between = core._dist(ny, nx, between_pix=True)
    max_expect_between = np.sqrt(((ny - 1) / 2) ** 2 + ((nx - 1) / 2) ** 2)
    min_expect_between = np.sqrt(0.5**2 + 0.5**2)

    assert d_between.max() == max_expect_between
    assert d_between.min() == min_expect_between


def test_rad2mas_mas2rad():
    rad = np.linspace(0, 2 * np.pi)
    mas = core.rad2mas(rad)
    mas_check = rad * 180 / np.pi * 3600 * 1000
    np.testing.assert_allclose(mas, mas_check)

    rad_back = core.mas2rad(mas)
    np.testing.assert_allclose(rad, rad_back)

    rad_check = core.mas2rad(mas_check)
    np.testing.assert_allclose(rad, rad_check)
