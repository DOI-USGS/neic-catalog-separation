import pathlib

import numpy as np
import pytest

from neic_catalog_separation import (funcs, io)
from neic_catalog_separation.eqtype import EqType


@pytest.fixture
def config():
    config_dir = pathlib.Path(funcs.__file__).parent.resolve() / "config"
    return io.read_config(config_dir / "subduction.yml")


class TestDetermineQuality:

    def test_quality_a(self):
        "Test quality determination for A quality (qual)"
        expected_quality = "A"
        quality = funcs.determine_quality(
            EqType.CRUSTAL, 100, 0, 0, 113.342, 80)

        assert expected_quality == quality

    def test_quality_b(self):
        "Test quality determination for B quality (qual)"
        expected_quality = "B"
        quality = funcs.determine_quality(
            EqType.SUBDUCTION_INTRASLAB, 0, 100, 0, np.nan, 80)

        assert expected_quality == quality

    def test_quality_c(self):
        "Test quality determination for C quality (qual)"
        expected_quality = "C"
        quality = funcs.determine_quality(EqType.CRUSTAL, 75, 5, 20, 235, -80)

        assert expected_quality == quality

    def test_quality_d(self):
        "Test quality determination for D quality (qual)"
        expected_quality = "D"
        quality = funcs.determine_quality(
            EqType.SUBDUCTION_INTERFACE, 0, 46.667, 53.333, np.nan, 0)

        assert expected_quality == quality

    def test_quality_eq(self):
        "Test quality determination for equal (eq) probabilty"
        expected_quality = "D"
        quality = funcs.determine_quality(
            EqType.SUBDUCTION_INTERFACE, 0, 50, 50, np.nan, 0)

        assert expected_quality == quality


class TestGetKaganAngle:

    def test_angle(self):
        """Test Kagan angle functionality"""
        desired_kagan = 56.64
        kagan = funcs.get_kagan_angle(
            221.26015,
            21.01511,
            90,
            161.819,
            65.326,
            34.721,
        )

        # within a 1% tolerance is ok
        assert desired_kagan == pytest.approx(kagan, rel=0.01)


class TestGetProbability:

    def test_p1(self, config):
        """Test ramp function calculation that returns probability of p1"""

        x = 15.0
        x1 = 25.0
        x2 = 40.0
        p1 = config["p_interface_sz"]["p1"]
        p2 = config["p_interface_sz"]["p2"]
        prob = funcs.get_probability(x, x1, p1, x2, p2)

        assert p1 == prob

    def test_p2(self, config):
        """Test ramp function calculation that returns probability of p2"""
        x = 50.0
        x1 = 25.0
        x2 = 40.0
        p1 = config["p_interface_sz"]["p1"]
        p2 = config["p_interface_sz"]["p2"]
        prob = funcs.get_probability(x, x1, p1, x2, p2)

        assert p2 == prob

    def test_ramp(self, config):
        """Test ramp function calculation that returns probability defined by ramp"""
        x = 30.0
        x1 = 25.0
        x2 = 40.0
        p1 = config["p_interface_sz"]["p1"]
        p2 = config["p_interface_sz"]["p2"]

        expected_prob = 0.66
        prob = funcs.get_probability(x, x1, p1, x2, p2)

        # within a 3% tolerance is ok
        assert expected_prob == pytest.approx(prob, rel=0.03)
