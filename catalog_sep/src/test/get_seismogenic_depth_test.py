import os


def test_get_seismogenic_depth_sam():
    """Test getting seismogenic zone depth for South American (sam)"""
    from classify_catalog_Funcs import get_seismogenic_depth

    # set parameters to tests
    working_dir = os.getcwd()
    slab = "sam"
    nshm = False

    # set desired value
    desired_sz_deep = 45
    desired_rake = 85

    # run test
    sz_deep, srake = get_seismogenic_depth(working_dir, slab, nshm)

    assert desired_sz_deep == sz_deep
    assert desired_rake == srake


def test_get_seismogenic_depth_default():
    """Test getting default seismogenic zone depth"""
    from classify_catalog_Funcs import get_seismogenic_depth

    # set parameters to tests
    working_dir = os.getcwd()
    slab = "cas"
    nshm = False

    # set desired value
    desired_sz_deep = 40
    desired_rake = 90

    # run test
    sz_deep, srake = get_seismogenic_depth(working_dir, slab, nshm)

    assert desired_sz_deep == sz_deep
    assert desired_rake == srake


def test_get_seismogenic_depth_nshm():
    """Test getting seismogenic zone depth fro Caribbean (car) NSHM value"""
    from classify_catalog_Funcs import get_seismogenic_depth

    # set parameters to tests
    working_dir = os.getcwd()
    slab = "car"
    nshm = True

    # set desired value
    desired_sz_deep = 50
    desired_rake = 90

    # run test
    sz_deep, srake = get_seismogenic_depth(working_dir, slab, nshm)

    assert desired_sz_deep == sz_deep
    assert desired_rake == srake
