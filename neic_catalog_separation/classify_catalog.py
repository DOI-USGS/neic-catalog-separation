# stdlib imports
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# external imports
import polars
import tqdm

# local imports
from . import classify_earthquake, data_sources, io

# ---------------------------------------------------------------------------
# Worker-side global variables. Populated once per worker process by _init_worker(),
# so the (large) Slab2 and Crust1.0 models are pickled once per worker rather
# than once per earthquake.
# ---------------------------------------------------------------------------
_MODELS = {}


def _init_worker(slab1, slab2, moho_depth, seismogenic_depth, config, nshm):
    """Pool initializer: runs once per worker process."""
    _MODELS.update(
        slab1=slab1,
        slab2=slab2,
        moho_depth=moho_depth,
        seismogenic_depth=seismogenic_depth,
        config=config,
        nshm=nshm,
    )


def _classify_chunk(earthquakes: list) -> list:
    """Classify a batch of earthquakes inside a worker process.

    Only the earthquake dicts cross the process boundary; the models are
    already resident in the worker via _MODELS.
    """
    return [
        classify_earthquake.classify(earthquake=earthquake, **_MODELS)
        for earthquake in earthquakes
    ]


def _chunked(sequence: list, chunk_size: int):
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start : start + chunk_size]


def classify_catalog(
    output_filename: str,
    slab_region1: str,
    slab_region2: str,
    catalog_filename: str,
    config_filename: str,
    nshm: bool = False,
    num_processes: int = 1,
    chunk_size: int = 0,
) -> None:
    """Reads in files and sets up arguments to the function classify_earthquake,
    which runs in parallel and determines probability that an event occurred in
    the upper plate (crustal), lower plate (intraslab), or along the subduction
    interface.
    """
    top_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = top_dir / "data"
    slab1 = data_sources.slab(region=slab_region1, data_dir=data_dir / "Slab2")

    slab2 = None
    if slab_region2:
        slab2 = data_sources.slab(region=slab_region2, data_dir=data_dir / "Slab2")

    moho_depth = data_sources.moho_depth(data_dir=data_dir / "crust1.0")
    seismogenic_depth = data_sources.seismogenic_depth(data_dir=data_dir / "Slab2")

    conf_dir = top_dir / "config"
    config_filename = config_filename or conf_dir / "subduction.yml"
    config = io.read_config(config_filename)

    catalog = polars.read_csv(
        catalog_filename, null_values=["NaN", "nan"]
    ).with_columns(polars.all().fill_null(np.nan))

    earthquakes = list(catalog.iter_rows(named=True))

    if num_processes == 1:
        _init_worker(slab1, slab2, moho_depth, seismogenic_depth, config, nshm)
        results = []
        with tqdm.tqdm(total=len(earthquakes)) as progress_monitor:
            for earthquake in earthquakes:
                results.append(
                    classify_earthquake.classify(earthquake=earthquake, **_MODELS)
                )
                progress_monitor.update(1)
    else:
        # Aim for a few chunks per worker: enough to balance load, few enough
        # that per-task overhead stays negligible.
        if not chunk_size:
            chunk_size = max(1, len(earthquakes) // (num_processes * 8))
        chunks = list(_chunked(earthquakes, chunk_size))

        results = []
        with ProcessPoolExecutor(
            max_workers=num_processes,
            initializer=_init_worker,
            initargs=(slab1, slab2, moho_depth, seismogenic_depth, config, nshm),
        ) as executor:
            with tqdm.tqdm(total=len(earthquakes)) as progress_monitor:
                # map() preserves input order and streams results back as
                # workers finish, so memory stays bounded.
                for chunk_results in executor.map(_classify_chunk, chunks):
                    results.extend(chunk_results)
                    progress_monitor.update(len(chunk_results))

    classification = polars.from_dicts(results)
    classification.write_csv(output_filename)


def cli():
    """Command line interface."""
    import argparse

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
        "--slab2-regions",
        dest="slab2_regions",
        help="Comma separated list of names of Slab2 regions. Currently limited to a maximum of two regions.",
    )
    argparser.add_argument(
        "--catalog",
        dest="catalog_filename",
        help="Path and file name for the input catalog to separate (must be CSV file format).",
    )
    argparser.add_argument(
        "--output",
        dest="output",
        help="Path and file name for the output file.",
    )
    argparser.add_argument(
        "--config",
        dest="config_filename",
        help="Path and file name for the configuration file. If not provided, the default configuration is used.",
    )
    argparser.add_argument(
        "--nshm",
        action="store_true",
        help="Classify earthquakes outside of the specified Slab2 region.",
    )
    argparser.add_argument(
        "--num-processes",
        default=1,
        type=int,
        dest="num_processes",
        help="Number of processes to use in classification.",
    )

    args = argparser.parse_args()

    regions = args.slab2_regions.split(",")
    if len(regions) > 2:
        raise IOError(
            "Currently, the algorithm is limited to a maximum of two Slab2 regions."
        )
    slab_region1 = regions[0]
    slab_region2 = regions[1] if len(regions) > 1 else None

    classify_catalog(
        output_filename=args.output,
        slab_region1=slab_region1,
        slab_region2=slab_region2,
        catalog_filename=args.catalog_filename,
        config_filename=args.config_filename,
        num_processes=args.num_processes,
        nshm=args.nshm,
    )
