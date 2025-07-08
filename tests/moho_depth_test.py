import pathlib

import pytest

from neic_catalog_separation import data_sources


@pytest.fixture
def moho_depth():
    """"""
    data_dir = pathlib.Path(data_sources.__file__).parent.resolve() / "data" / "crust1.0"
    return data_sources.moho_depth(data_dir=data_dir)


class TestMohoDepth:

    def test_get_value(self, moho_depth):
        eq = {"lat": -33.683, "lon": -72.161}

        expected_depth = 10.57

        depth = moho_depth.get_value(**eq)

        # within a 1% tolerance is ok
        tolerance = 0.01
        assert expected_depth == pytest.approx(depth, rel=tolerance)
