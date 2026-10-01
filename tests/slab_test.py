import pathlib

import numpy as np
import pytest

from neic_catalog_separation import data_sources


@pytest.fixture
def data_dir():
    return pathlib.Path(data_sources.__file__).parent.resolve() / "data" / "Slab2"


@pytest.fixture
def slab_mue(data_dir):
    """"""
    return data_sources.slab(region="mue", data_dir=data_dir)


@pytest.fixture
def slab_car(data_dir):
    """"""
    return data_sources.slab(region="car", data_dir=data_dir)


@pytest.fixture
def slab_sam(data_dir):
    """"""
    return data_sources.slab(region="sam", data_dir=data_dir)


@pytest.fixture
def slab_sol(data_dir):
    """"""
    return data_sources.slab(region="sol", data_dir=data_dir)


class TestSlab:
    def test_get_values(self, slab_sam):
        eq = {"lat": -33.683, "lon": -72.161}

        expected_depth = 20.79
        expected_strike = 13.38
        expected_dip = 15.32
        expected_uncertainty = 12.13
        expected_thickness = 71.44

        depth, strike, dip, thickness, uncertainty = slab_sam.get_values(**eq)

        # within a 1% tolerance is ok
        tolerance = 0.01
        assert expected_depth == pytest.approx(depth, rel=tolerance)
        assert expected_strike == pytest.approx(strike, rel=tolerance)
        assert expected_dip == pytest.approx(dip, rel=tolerance)
        assert expected_thickness == pytest.approx(thickness, rel=tolerance)
        assert expected_uncertainty == pytest.approx(uncertainty, rel=tolerance)

    def test_overturned_sol(self, slab_sol):
        "Test find closest slab location for tilted/supplemental Solomon slab"
        expected_depth = 552.73
        expected_strike = 272.21
        expected_dip = 89.91

        depth, strike, dip, thickness = slab_sol.get_overturned(lon=149.123, lat=-4.25)

        assert expected_depth == depth
        assert expected_strike == strike
        assert expected_dip == dip

    def test_overturned_sol_nan(self, slab_sol):
        "Test earthquake that is too far from tilted slab for accurate classification"
        depth, strike, dip, thickness = slab_sol.get_overturned(lon=149.123, lat=-7.25)

        # returned values should be np.nan
        assert np.isnan(depth)
        assert np.isnan(strike)
        assert np.isnan(dip)
        assert np.isnan(thickness)

    def test_find_closest_overlap(self, slab_mue, slab_car):
        "Test closest slab function where two slabs overlap"
        eq_mue = {"lon": -66.745, "lat": 18.261, "depth": 80.0}  # Depth of MUE slab
        eq_car = {"lon": -66.745, "lat": 18.261, "depth": 100.0}  # depth of CAR slab

        slab_closest = data_sources.Slab.find_closest(
            slab1=slab_mue, slab2=slab_car, **eq_mue
        )
        assert "mue" == slab_closest.region

        slab_closest = data_sources.Slab.find_closest(
            slab2=slab_mue, slab1=slab_car, **eq_mue
        )
        assert "mue" == slab_closest.region

        slab_closest = data_sources.Slab.find_closest(
            slab1=slab_mue, slab2=slab_car, **eq_car
        )
        assert "car" == slab_closest.region

        slab_closest = data_sources.Slab.find_closest(
            slab2=slab_mue, slab1=slab_car, **eq_car
        )
        assert "car" == slab_closest.region

    def test_find_closest_mue(self, slab_mue, slab_car):
        "Test closest slab function where only MUE slab exists"
        eq = {"lon": -68.256, "lat": 17.750, "depth": 20.0}
        expected_region = "mue"

        slab_closest = data_sources.Slab.find_closest(
            slab1=slab_mue, slab2=slab_car, **eq
        )
        assert expected_region == slab_closest.region

        slab_closest = data_sources.Slab.find_closest(
            slab2=slab_mue, slab1=slab_car, **eq
        )
        assert expected_region == slab_closest.region

    def test_find_closest_car(self, slab_mue, slab_car):
        "Test closest slab function where only CAR slab exists"
        eq = {"lon": -66.745, "lat": 19.100, "depth": 20.0}
        expected_region = "car"

        slab_closest = data_sources.Slab.find_closest(
            slab1=slab_mue, slab2=slab_car, **eq
        )
        assert expected_region == slab_closest.region

        slab_closest = data_sources.Slab.find_closest(
            slab2=slab_mue, slab1=slab_car, **eq
        )
        assert expected_region == slab_closest.region
