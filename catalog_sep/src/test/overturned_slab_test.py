import os

import numpy as np


def test_overturned_slab_sol():
    "Test find closest slab location for tilted/supplemental Solomon slab"
    from classify_catalog_Funcs import overturned_slab

    working_dir = os.getcwd()
    # set desired values
    desired_sdep = 552.73
    desired_sstr = 272.21
    desired_sdip = 89.91

    # run test
    sdep, sstr, sdip = overturned_slab("sol", working_dir, 149.123, -4.25)

    assert desired_sdep == sdep
    assert desired_sstr == sstr
    assert desired_sdip == sdip


def test_overturned_slab_sol_nan():
    "Test earthquake that is too far from tilted slab for accurate classification"
    from classify_catalog_Funcs import overturned_slab

    working_dir = os.getcwd()

    # run test
    sdep, sstr, sdip = overturned_slab("sol", working_dir, 149.123, -7.25)

    # returned values should be np.nan
    assert np.isnan(sdep)
    assert np.isnan(sstr)
    assert np.isnan(sdip)
