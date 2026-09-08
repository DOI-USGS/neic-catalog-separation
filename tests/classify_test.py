import pathlib

import polars

from neic_catalog_separation import classify_catalog


class TestClassifyCatalog:
    def test_classify_catalog(self):
        FILENAME_CATALOG = "cas_subset.csv"
        FILENAME_OUTPUT = "cas_output.csv"

        output_filepath = pathlib.Path(__file__).parent.resolve() / FILENAME_OUTPUT
        if output_filepath.exists():
            output_filepath.unlink()

        catalog_filepath = pathlib.Path(__file__).parent.resolve() / FILENAME_CATALOG
        classify_catalog.classify_catalog(
            output_filename=output_filepath,
            slab_region1="cas",
            slab_region2=None,
            catalog_filename=catalog_filepath,
            config_filename=None,
        )
        assert output_filepath.exists() and output_filepath.is_file()
        output = polars.read_csv(output_filepath)
        assert len(output) == 19
