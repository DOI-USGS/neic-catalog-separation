import os

import pytest


def test_get_slab_moho_depth():
    """Test getting slab2 properties and moho depth"""
    from classify_catalog_Funcs import get_slab_moho_info

    # set parameters to tests
    working_dir = os.getcwd()
    smod = "sam"
    lat = -33.683
    lon = -72.161

    # set desired value
    desired_sdep = 20.79
    desired_sstr = 13.38
    desired_sdip = 15.32
    desired_sunc = 12.13
    desired_mohoDepth = 10.57

    sdep, sstr, sdip, sunc, mohoDepth = get_slab_moho_info(smod, working_dir, lat, lon)

    # within a 1% tolerance is ok
    assert desired_sdep == pytest.approx(sdep, rel=0.01)
    assert desired_sstr == pytest.approx(sstr, rel=0.01)
    assert desired_sdip == pytest.approx(sdip, rel=0.01)
    assert desired_sunc == pytest.approx(sunc, rel=0.01)
    assert desired_mohoDepth == pytest.approx(mohoDepth, rel=0.01)
