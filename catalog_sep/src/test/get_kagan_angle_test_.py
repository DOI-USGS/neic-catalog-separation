import pytest


def test_get_kagan_angle():
    """Test Kagan angle functionality"""
    from classify_catalog_Funcs import get_kagan_angle

    # set desired value
    desired_kagan = 56.64
    # run test
    kagan = get_kagan_angle(
        221.26015,
        21.01511,
        90,
        161.819,
        65.326,
        34.721,
    )

    # within a 1% tolerance is ok
    assert desired_kagan == pytest.approx(kagan, rel=0.01)
