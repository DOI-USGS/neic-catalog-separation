import pytest

from neic_catalog_separation import data_sources


@pytest.fixture
def seismogenic_depth():
    """Create seismogenic depth test subject."""
    import pathlib

    data_dir = pathlib.Path(data_sources.__file__).parent.resolve() / "data" / "Slab2"
    return data_sources.seismogenic_depth(data_dir)


def test_sam(seismogenic_depth):
    """Test getting seismogenic zone depth for South American (sam)"""
    slab_region = "sam"
    nshm = False

    expected_sz_deep = 45
    expected_rake = 85

    sz_deep, sz_rake = seismogenic_depth.get_values(slab_region=slab_region, nshm=nshm)
    assert expected_sz_deep == sz_deep
    assert expected_rake == sz_rake


def test_default(seismogenic_depth):
    """Test getting default seismogenic zone depth"""
    slab_region = "cas"
    nshm = False

    expected_sz_deep = 40
    expected_rake = 90

    sz_deep, sz_rake = seismogenic_depth.get_values(slab_region=slab_region, nshm=nshm)
    assert expected_sz_deep == sz_deep
    assert expected_rake == sz_rake


def test_nshm(seismogenic_depth):
    """Test getting seismogenic zone depth for Caribbean (car) NSHM value"""
    slab_region = "car"
    nshm = True

    expected_sz_deep = 50
    expected_rake = 90

    sz_deep, sz_rake = seismogenic_depth.get_values(slab_region=slab_region, nshm=nshm)
    assert expected_sz_deep == sz_deep
    assert expected_rake == sz_rake
