import os


def test_closest_slab_mue():
    "Test closest slab function"
    from classify_catalog_Funcs import determine_closest_slab

    working_dir = os.getcwd()
    # set desired value
    desired_smod = "mue"
    # run test
    smod = determine_closest_slab("mue", "car", working_dir, 19.815, -71.175, 10)

    assert desired_smod == smod


def test_closest_slab_car():
    "Test closest slab function"
    from classify_catalog_Funcs import determine_closest_slab

    working_dir = os.getcwd()
    # set desired value
    desired_smod = "car"
    # run test
    smod = determine_closest_slab("mue", "car", working_dir, 14.901, -60.077, 23)

    assert desired_smod == smod
