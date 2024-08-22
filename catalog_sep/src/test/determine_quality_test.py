import numpy as np


def test_determine_quality_a():
    "Test quality determination for A quality (qual)"
    from classify_catalog_Funcs import determine_quality

    # set desired value
    desired_qual = "A"
    # run test
    qual = determine_quality("u", 100, 0, 0, 113.342, 80)

    assert desired_qual == qual


def test_determine_quality_b():
    "Test quality determination for B quality (qual)"
    from classify_catalog_Funcs import determine_quality

    # set desired value
    desired_qual = "B"
    # run test
    qual = determine_quality("l", 0, 100, 0, np.nan, 80)

    assert desired_qual == qual


def test_determine_quality_c():
    "Test quality determination for C quality (qual)"
    from classify_catalog_Funcs import determine_quality

    # set desired value
    desired_qual = "C"
    # run test
    qual = determine_quality("u", 75, 5, 20, 235, -80)

    assert desired_qual == qual


def test_determine_quality_d():
    "Test quality determination for D quality (qual)"
    from classify_catalog_Funcs import determine_quality

    # set desired value
    desired_qual = "D"
    # run test
    qual = determine_quality("i", 0, 46.667, 53.333, np.nan, 0)

    assert desired_qual == qual


def test_determine_quality_eq():
    "Test quality determination for equal (eq) probabilty"
    from classify_catalog_Funcs import determine_quality

    # set desired value
    desired_qual = "D"
    # run test
    qual = determine_quality("ieq", 0, 50, 50, np.nan, 0)

    assert desired_qual == qual
