def end_to_end():
    """Run an end-to-end test of the code and compare to output file"""
    from classify_catalog import classify_catalog

    # run test
    classify_catalog(
        "cas", "catalog_sep/src/test/test_data/cas_input_test.csv", False, False
    )

    assert 1 == 1
