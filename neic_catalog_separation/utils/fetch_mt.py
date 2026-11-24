import json
import logging
import pathlib
import time
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import polars
import tqdm

COMCAT_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"
)

# Moment tensor components (columns in the output catalog)
MT_COMPONENTS = ("mrr", "mtt", "mpp", "mrt", "mrp", "mtp")

# Timeout (seconds) for a single ComCat request
REQUEST_TIMEOUT = 30


def parse_moment_tensor(event: dict) -> dict:
    """Extract the moment tensor components from a ComCat event (GeoJSON).

    Args:
        event: ComCat event as returned by the FDSN event web service
            (format=geojson).

    Returns:
        Dictionary with keys mrr, mtt, mpp, mrt, mrp, mtp. A component is NaN if the
        event has no moment tensor product or the component is missing/invalid.
    """
    nan_tensor = {name: np.nan for name in MT_COMPONENTS}
    try:
        properties = event["properties"]["products"]["moment-tensor"][0]["properties"]
    except (KeyError, IndexError, TypeError):
        return nan_tensor

    tensor = {}
    for name in MT_COMPONENTS:
        try:
            tensor[name] = float(properties.get(f"tensor-{name}"))
        except (TypeError, ValueError):
            tensor[name] = np.nan
    return tensor


def fetch_event(event_id) -> dict:
    """Download a single event from ComCat.

    Args:
        event_id: ComCat event id.

    Returns:
        ComCat event (GeoJSON) as a dictionary.
    """
    url = COMCAT_URL.format(event_id=quote(str(event_id), safe=""))
    with urlopen(url, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_mt_from_comcat(
    catalog_filename: pathlib.Path,
    output_filename: pathlib.Path,
    sleep_seconds: float = 1.0,
):
    """Query the ANSS earthquake catalog, ComCat, for moment tensors given event ID's in a catalog CSV file

    The moment tensor components (mrr, mtt, mpp, mrt, mrp, mtp) are written to the
    output catalog, replacing any existing columns with those names. Events without
    a moment tensor, or for which the request fails, have NaN components.

    Args:
        catalog_filename: Name of input earthquake catalog file (must have column 'id').
        output_filename: Name of output file.
        sleep_seconds: Delay between requests (avoids the ComCat rate limit of 500
            requests per 5 minutes).
    """
    logger = logging.getLogger(__name__)

    data = polars.read_csv(catalog_filename, infer_schema_length=None)
    event_ids = data["id"].to_list()
    components = {name: [] for name in MT_COMPONENTS}

    # loop through event IDs and get moment tensor products:
    for index, event_id in enumerate(tqdm.tqdm(event_ids)):
        if index > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        try:
            tensor = parse_moment_tensor(fetch_event(event_id))
        except Exception as err:  # network, HTTP, or JSON errors
            logger.warning("Cannot get event %s from ComCat: %s", event_id, err)
            tensor = parse_moment_tensor({})

        for name in MT_COMPONENTS:
            components[name].append(tensor[name])

    data = data.with_columns(
        [
            polars.Series(name, values, dtype=polars.Float64)
            for name, values in components.items()
        ]
    )
    data.write_csv(output_filename)


def cli():
    """Command line interface."""
    import argparse

    desc = """Query ComCat for moment tensors for event ID's provided in a catalog CSV file
    """
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "--catalog",
        action="store",
        required=True,
        help="Name of input earthquake catalog file containing earthquake ids.",
    )
    argparser.add_argument(
        "--output",
        action="store",
        required=True,
        help="Name of output file.",
    )

    args = argparser.parse_args()
    fetch_mt_from_comcat(catalog_filename=args.catalog, output_filename=args.output)
