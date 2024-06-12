#!/usr/bin/env python3
# stdlib imports
import argparse
import multiprocessing as mp
import os
from functools import partial

import pandas as pd

# local imports
import classify_catalog_ParallelLoop as classify

# constants
FLEX = 15  # buffer


def classify_catalog(slab1, slab2, nshm):
    """
    Reads in files and sets up arguments to the function classify_catalog_ParallelLoop, which runs in parallel and determines probability that an event occurred in the upper plate (crustal), lower plate (intraslab), or along the subduction interface.
    """
    # get Slab2 catalog for user requested slab model
    infile = f"{slab1}_04-18_input.csv"
    working_dir = os.getcwd()
    fpath = f"{working_dir}/catalog_sep/Input/Slab2Catalogs/{infile}"
    # determine output file name
    outfile = f"{working_dir}/{slab1}_eqtype"

    dataframe = pd.read_csv(fpath, low_memory=False)
    # create list to get length of dataframe (needed for parallel loop)
    id_no = dataframe["id_no"].tolist()

    # Defaults for slabs without SZT constraint
    srake = 90

    # run classify_catalog_ParallelLoops in parallel using max processors:
    noprocs = mp.cpu_count()
    # noprocs = 1

    pool = mp.Pool(processes=noprocs)
    loop = partial(
        classify.classify_eqs,
        slab1,
        slab2,
        nshm,
        working_dir,
        dataframe,
        srake,
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
        "--second_slab",
        default=False,
        help="Name of the second Slab2 region pertaining to the earthquake catalog. This is optional and only required for catalogs that span overlapping slabs.",
    )
    argparser.add_argument(
        "--nshm",
        default="False",
        choices=["True", "False"],
        help="Specify True if running for a NSHM catalog. This will classify earthquakes outside of the specified Slab2 region. Default is False.",
    )

    pargs, unknown = argparser.parse_known_args()
    slab1 = pargs.slab2_region
    slab2 = pargs.second_slab
    nshm = pargs.nshm
    classify_catalog(slab1, slab2, nshm)
