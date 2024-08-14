import os

import pandas as pd

HEADER = [
    "lat",
    "lon",
    "depth",
    "unc",
    "ID",
    "etype",
    "mag",
    "time",
    "Paz",
    "Ppl",
    "Taz",
    "Tpl",
    "S1",
    "D1",
    "R1",
    "S2",
    "D2",
    "R2",
    "lon1",
    "lat1",
    "msc",
    "event ID",
    "src",
    "p_crust",
    "p_int",
    "p_slab",
    "loc",
    "quality",
    "slab region",
]


def test_end_to_end():
    """Run an end-to-end test of the code and compare to output file"""
    from classify_catalog import classify_catalog

    # run test
    classify_catalog(
        "cas", "catalog_sep/src/test/test_data/cas_input_test.csv", False, False, True
    )

    # compare output file to test file:
    try:
        test_output = pd.read_csv(
            "catalog_sep/src/test/test_data/cas_separated_test_output.csv", names=HEADER
        )
        output = pd.read_csv("cas_separated.csv", names=HEADER)

        # verify that results for several event IDs are the same
        test_iscgem900805 = test_output[test_output["event ID"] == "iscgem900805"]
        iscgem900805 = output[output["event ID"] == "iscgem900805"]
        diff1 = test_iscgem900805.compare(iscgem900805)

        test_usp000d0fx = test_output[test_output["event ID"] == "usp000d0fx"]
        usp000d0fx = output[output["event ID"] == "usp000d0fx"]
        diff2 = test_usp000d0fx.compare(usp000d0fx)

        test_uw61251926 = test_output[test_output["event ID"] == "uw61251926"]
        uw61251926 = output[output["event ID"] == "uw61251926"]
        diff3 = test_uw61251926.compare(uw61251926)

        test_nc228027 = test_output[test_output["event ID"] == "nc228027"]
        nc228027 = output[output["event ID"] == "nc228027"]
        diff4 = test_nc228027.compare(nc228027)

        # assert that there are no differences in the comparisons
        assert len(diff1) == 0
        assert len(diff2) == 0
        assert len(diff3) == 0
        assert len(diff4) == 0

    # always remove the test run output
    finally:
        os.remove("cas_separated.csv")
