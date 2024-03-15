#!/usr/bin/env python3
# stdlib imports
import argparse
import csv
import multiprocessing as mp
import os
from functools import partial

import pandas as pd

# local imports
import classify_catalog_ParallelLoop as classify

# constants
FLEX = 15  # buffer


def classify_catalog(smod):
    """
    Reads in files and sets up arguments to the function classify_catalog_ParallelLoop, which runs in parallel and determines probability that an event occured in the upper plate (crustal), lower plate (intraslab), or along the subduction interface.
    """
    # get Slab2 catalog for user requested slab model
    infile = f"{smod}_04-18_input.csv"
    working_dir = os.getcwd()
    fpath = f"{working_dir}/catalog_sep/Input/Slab2Catalogs/{infile}"
    # get published Slab2 seismogenic thickness file
    sztfile = f"{working_dir}/catalog_sep/Input/szt.txt"
    # determine output file name
    outfile = f"{working_dir}/{smod}_eqtype"

    dataframe = pd.read_csv(fpath, low_memory=False)
    # create list to get length of dataframe (needed for parallel loop)
    id_no = dataframe["id_no"].tolist()

    # Initialize empty variables to obtain from seismogenic zone thickness file
    scode = []
    sd = []
    arak = []

    # read the seismogenic zone thickness file
    count = 0
    with open(sztfile) as file:
        reader = csv.reader(file, delimiter=",", skipinitialspace=True)
        next(reader)
        for one, two, three, four, five, six, seven, eight, nine, ten, eleven in reader:
            scode.append(str(three))
            sd.append(float(six))
            arak.append(float(ten))
            if scode[count] == smod:
                sz_deep = sd[count]
                srake = arak[count]
                break
            count = count + 1

    # Defaults for slabs without SZT constraint
    sz_deep = 40  # deep seismogenic limit
    srake = 90
    # determine dlim (deep seismogenic limit + flex (buffer/wiggle room))
    dlim = sz_deep + FLEX

    # run classify_catalog_ParallelLoops in parallel using max processors:
    noprocs = mp.cpu_count()

    pool = mp.Pool(processes=noprocs)
    loop = partial(
        classify.classify_eqs,
        smod,
        working_dir,
        dataframe,
        dlim,
        srake,
        outfile,
        FLEX,
    )
    # iterate function through lengh of ID/catalog
    indices = [i for i in range(len(id_no))]
    pool.map_async(loop, indices, 1)
    pool.close()
    pool.join()


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
        help="Name Slab2 region pertaining to the earthquake catalog is required",
        metavar="slab2_region",
    )

    pargs, unknown = argparser.parse_known_args()
    smod = pargs.slab2_region
    classify_catalog(smod)
