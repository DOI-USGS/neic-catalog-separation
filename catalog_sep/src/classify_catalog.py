#!/usr/bin/env python3
# stdlib imports
import argparse
import multiprocessing as mp
import os
from functools import partial

# local imports
import classify_catalog_ParallelLoop as classify
import pandas as pd

# constants
FLEX = 15  # buffer


def classify_catalog(slab1: str, input_file: str, slab2, nshm: bool) -> None:
    """
    Reads in files and sets up arguments to the function classify_catalog_ParallelLoop, which runs in parallel and determines probability that an event occurred in the upper plate (crustal), lower plate (intraslab), or along the subduction interface.
    """
    # get Slab2 catalog for user requested slab model
    infile = input_file
    working_dir = os.getcwd()
    # specify output file name
    outfile = f"{working_dir}/{slab1}_separated"

    dataframe = pd.read_csv(infile, low_memory=False)
    # create list to get length of dataframe (needed for parallel loop)
    id_no = dataframe["id_no"].tolist()

    # run classify_catalog_ParallelLoops in parallel using max processors, unless this is a test run then use 1 processor:
    if test:
        noprocs = 1
    else:
        noprocs = mp.cpu_count()

    pool = mp.Pool(processes=noprocs)
    loop = partial(
        classify.classify_eqs,
        slab1,
        slab2,
        nshm,
        working_dir,
        dataframe,
        outfile,
        FLEX,
    )
    # iterate function through lengh of ID/catalog
    indices = [i for i in range(len(id_no))]
    pool.map_async(
        loop,
        indices,
        chunksize=1,
        error_callback=custom_error_callback,
    )
    pool.close()
    pool.join()


def custom_error_callback(error):
    print(f"ERROR: {error}")


if __name__ == "__main__":
    desc = """
    Classify a Slab2 Region Earthquake Catalog. Use one of the 3 letter abbreviations for the Slab2 region catalog to classify

    Slab2 regions include:

        Aleutians               alu
        Calabria                cal
        Central America         cam
        Caribbean               car
        Cascadia                cas
        Cotabato                cot
        Halmahera               hal
        Hellenic                hel
        Himalaya                him
        Hindu Kush              hin
        Izu-Bonin               izu
        Kermadec                ker
        Kuril                   kur
        Makran                  mak
        Manila                  man
        Muertos                 mue
        Pamir                   pam
        New Guinea              png
        Philippines             phi
        Puysegur                puy
        Ryukyu                  ryu
        South America           sam
        Scotia                  sco
        Solomon Islands         sol
        Sulawesi                sul
        Sumatra/Java            sum
        Vanuatu                 van
    """
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "slab2_region",
        help="Name of the Slab2 region pertaining to the earthquake catalog is required",
    )
    argparser.add_argument(
        "input_file",
        help="Path and file name for the input catalog to separate (must be CSV file format)",
    )
    argparser.add_argument(
        "--second_slab",
        default=False,
        help="Name of the second Slab2 region pertaining to the earthquake catalog. This is optional and only required for catalogs that span overlapping slabs.",
    )
    argparser.add_argument(
        "--nshm",
        default=False,
        choices=["True", "False"],
        help="Specify True if running for a NSHM catalog. This will classify earthquakes outside of the specified Slab2 region. Default is False.",
    )

    pargs, unknown = argparser.parse_known_args()
    slab1 = pargs.slab2_region
    input_file = pargs.input_file
    slab2 = pargs.second_slab
    nshm = pargs.nshm

    classify_catalog(slab1, input_file, slab2, nshm)
