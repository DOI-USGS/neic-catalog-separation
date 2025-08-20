# stdlib imports
import pathlib

# third-party imports
import polars


def separate_input(catalog_filename: pathlib.Path, moment_tensors: bool):
    """Separate the output from running classify_catalog into sub-files based on classification

    Args:
        catalog_filename (str): file path to input file to separate into sub-files
        moment_tensors (bool): If True, also create files for events that have moment tensor data (default is False)
    """
    CLASSIFICATION = {
        "crustal": "crustal",
        "subduction_intraslab": "intraslab",
        "subduction_interface": "interface",
        "subduction_outerrise": "outerrise",
    }

    data = polars.read_csv(catalog_filename)
    for key, group_name in CLASSIFICATION.items():
        data_group = data.filter(polars.col("eq_category") == key)
        filename = f"{group_name}.csv"
        data_group.write_csv(filename)
        if moment_tensors:
            data_mt = data_group.filter(polars.col("S1").is_not_nan())
            filename = f"{group_name}_mt.csv"
            data_mt.write_csv(filename)


def cli():
    """Command line interface."""
    import argparse

    desc = """
    Separate the output from running classify_catalog into files based on classification.
    """
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "--catalog",
        action="store",
        dest="catalog",
        required=True,
        help="Name of catalog file to separate into classified sub-files (crustal, interface, slab, mantle)",
    )
    argparser.add_argument(
        "--moment-tensors",
        action="store_true",
        dest="moment_tensors",
        default=False,
        required=False,
        help="Separate files for data that moment tensor information",
    )

    args = argparser.parse_args()
    separate_input(
        catalog_filename=pathlib.Path(args.catalog), moment_tensors=args.moment_tensors
    )
