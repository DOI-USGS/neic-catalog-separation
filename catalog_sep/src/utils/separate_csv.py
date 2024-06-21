import argparse
import pandas as pd

# Header below refers to column names and may need to be updated
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
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "magnitude uncertainty",
    "adjusted moment mag",
    "counting factor",
    "event ID",
    "src",
    "p_crust",
    "p_int",
    "p_slab",
    "loc",
    "quality",
    "slab region",
]


def separate_input(input: str, mt: bool):
    """Separate the output from running classify_catalog into sub-files based on classification

    Args:
        input (str): file path to input file to separate into sub-files
        mt (bool): If True, also create files for events that have moment tensor data (default is False)
    """
    # open file
    dataFrame = pd.read_csv(
        input,
        names=HEADER,
    )
    # filter based on location
    # u = crustal/upper plate, l = intraslab/lower plate, i = interface, or = outer-rise
    crustalDF = dataFrame[dataFrame["loc"] == "u"]
    interfaceDF = dataFrame[dataFrame["loc"] == "i"]
    intraslabDF = dataFrame[dataFrame["loc"] == "l"]
    outerriseDF = dataFrame[dataFrame["loc"] == "or"]

    # save as CSV files with header
    crustalDF.to_csv(
        "crustal.csv",
        encoding="utf-8",
        index=False,
        header=HEADER,
        mode="a",
    )
    interfaceDF.to_csv(
        "interface.csv",
        encoding="utf-8",
        index=False,
        header=HEADER,
        mode="a",
    )
    intraslabDF.to_csv(
        "intraslab.csv",
        encoding="utf-8",
        index=False,
        header=HEADER,
        mode="a",
    )
    outerriseDF.to_csv(
        "outer-rise.csv",
        encoding="utf-8",
        index=False,
        header=HEADER,
        mode="a",
    )

    # if user also wants separated files consisting of events that have moment tensor data:
    if mt:
        crustalDF_mt = crustalDF[~crustalDF["S1"].isnull()]
        interfaceDF_mt = interfaceDF[~interfaceDF["S1"].isnull()]
        intraslabDF_mt = intraslabDF[~intraslabDF["S1"].isnull()]
        outerriseDF_mt = outerriseDF[~outerriseDF["S1"].isnull()]

        # save as CSV files with header
        crustalDF_mt.to_csv(
            "crustal_mt.csv",
            encoding="utf-8",
            index=False,
            header=HEADER,
            mode="a",
        )
        interfaceDF_mt.to_csv(
            "interface_mt.csv",
            encoding="utf-8",
            index=False,
            header=HEADER,
            mode="a",
        )
        intraslabDF_mt.to_csv(
            "intraslab_mt.csv",
            encoding="utf-8",
            index=False,
            header=HEADER,
            mode="a",
        )
        outerriseDF_mt.to_csv(
            "outer-rise_mt.csv",
            encoding="utf-8",
            index=False,
            header=HEADER,
            mode="a",
        )


if __name__ == "__main__":
    desc = """
    Separate the output from running classify_catalog into sub-files based on classification.
    """
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "input_file",
        help="Name of input file to separate into classified sub-files (crustal, interface, slab, mantle)",
    )
    argparser.add_argument(
        "-moment_tensors",
        "--mt",
        default=False,
        help="Specify True if you also want separate files for data that moment tensor information",
    )

    pargs, unknown = argparser.parse_known_args()
    input = pargs.input_file
    mt = pargs.mt
    separate_input(input, mt)
