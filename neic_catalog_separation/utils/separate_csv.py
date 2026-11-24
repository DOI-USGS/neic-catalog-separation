#!/usr/bin/env python3
# stdlib imports
import pathlib

# third-party imports
import polars


def category_filename_label(category: str) -> str:
    """Short label used in output file names for an earthquake category.

    The 'subduction_' prefix is dropped and the parts of a combined category are
    joined with an underscore, e.g., 'subduction_intraslab' -> 'intraslab' and
    'subduction_interface-mantle' -> 'interface_mantle'.
    """
    return "_".join(part.removeprefix("subduction_") for part in category.split("-"))


def separate_input(
    catalog_filename: pathlib.Path, output_file: str, moment_tensors: bool
) -> list:
    """Separate the output from running classify_catalog into sub-files based on classification

    One file, <output_file>_<category>.csv, is written for each category (eq_category)
    in the catalog, including combined categories (e.g., crustal-mantle ->
    <output_file>_crustal_mantle.csv). Every event is written to exactly one file.

    Args:
        catalog_filename (str): file path to input file to separate into sub-files
        output_file (str): prefix for the output file names
        moment_tensors (bool): If True, also create files for events that have moment tensor data (default is False)

    Returns:
        List of the files written.
    """
    data = polars.read_csv(catalog_filename, infer_schema_length=None)
    filenames = []
    for (category,), data_group in data.group_by("eq_category", maintain_order=True):
        label = category_filename_label(str(category))
        filename = f"{output_file}_{label}.csv"
        data_group.write_csv(filename)
        filenames.append(filename)
        if moment_tensors:
            data_mt = data_group.filter(
                polars.col("S1").is_not_null() & polars.col("S1").is_not_nan()
            )
            filename = f"{output_file}_{label}_mt.csv"
            data_mt.write_csv(filename)
            filenames.append(filename)
    return filenames


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
        help="Name of catalog file to separate into classified sub-files (crustal, interface, intraslab, mantle, outerrise, and combined categories)",
    )
    argparser.add_argument(
        "--output-file",
        action="store",
        dest="output_file",
        required=True,
        help="Name to prepend to output files",
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
        catalog_filename=pathlib.Path(args.catalog),
        output_file=args.output_file,
        moment_tensors=args.moment_tensors,
    )
