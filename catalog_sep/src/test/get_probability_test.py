import configparser
import os

import pytest


def test_get_kagan_angle_p1():
    """Test ramp function calculation that returns probability of p1"""
    from classify_catalog_Funcs import get_probability

    # get info from config file
    working_dir = os.getcwd()
    config = configparser.ConfigParser()
    config.sections()
    config.read(f"{working_dir}/catalog_sep/Input/config/subduction.conf")

    # define values for test
    x = 15
    x1 = 25
    x2 = 40
    p1 = float(config["p_int_sz"]["p1"])
    p2 = float(config["p_int_sz"]["p2"])

    # run test
    prob = get_probability(x, x1, p1, x2, p2)

    assert p1 == prob


def test_get_kagan_angle_p2():
    """Test ramp function calculation that returns probability of p2"""
    from classify_catalog_Funcs import get_probability

    # get info from config file
    working_dir = os.getcwd()
    config = configparser.ConfigParser()
    config.sections()
    config.read(f"{working_dir}/catalog_sep/Input/config/subduction.conf")

    # define values for test
    x = 50
    x1 = 25
    x2 = 40
    p1 = float(config["p_int_sz"]["p1"])
    p2 = float(config["p_int_sz"]["p2"])

    # run test
    prob = get_probability(x, x1, p1, x2, p2)

    assert p2 == prob


def test_get_kagan_angle_ramp():
    """Test ramp function calculation that returns probability defined by ramp"""
    from classify_catalog_Funcs import get_probability

    # get info from config file
    working_dir = os.getcwd()
    config = configparser.ConfigParser()
    config.sections()
    config.read(f"{working_dir}/catalog_sep/Input/config/subduction.conf")

    # define values for test
    x = 30
    x1 = 25
    x2 = 40
    p1 = float(config["p_int_sz"]["p1"])
    p2 = float(config["p_int_sz"]["p2"])

    # set desired value
    desired_prob = 0.66
    # run test
    prob = get_probability(x, x1, p1, x2, p2)

    # within a 3% tolerance is ok
    assert desired_prob == pytest.approx(prob, rel=0.03)
