import json
import logging
import pathlib
import time
from urllib.request import urlopen

import polars
import tqdm


def fetch_mt_from_comcat(catalog_filename: pathlib.Path, output_filename: pathlib.Path):
    """Query the ANSS earthquake catalog, ComCat, for moment tensors given event ID's in a catalog CSV file

    Args:
        catalog_filename: Name of input earthquake catalog file.
        output_fielname: Name of output file.
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"

    logger = logging.getLogger(__name__)

    data = polars.read_csv(catalog_filename)
    event_ids = data["id"].to_list()
    mt1 = []
    mt2 = []
    mt3 = []
    mt4 = []
    mt5 = []
    mt6 = []

    # loop through event IDs and get moment tensor products:
    for event_id in tqdm.tqdm(event_ids):
        # avoid rate limit of 500 requests per 5 minutes for FDSN requests
        time.sleep(1)

        id_url = url.format(event_id=event_id)
        try:
            url_lines = [x.decode("utf-8") for x in urlopen(id_url)]
            bigjson = json.loads("".join(url_lines))
        except Exception:
            logger.warning("Cannot connect to url %s for event %s.", id_url, event_id)
        try:
            smalljson = bigjson["properties"]["products"]
            fm_dict = smalljson["moment-tensor"][0]["properties"]
            mpp = fm_dict.get("tensor-mpp")
            mrp = fm_dict.get("tensor-mrp")
            mrr = fm_dict.get("tensor-mrr")
            mrt = fm_dict.get("tensor-mrt")
            mtp = fm_dict.get("tensor-mtp")
            mtt = fm_dict.get("tensor-mtt")
            print(mpp)
        except KeyError:
            mpp = float("nan")
            mrp = float("nan")
            mrr = float("nan")
            mrt = float("nan")
            mtp = float("nan")
            mtt = float("nan")
        mt1.append(mpp)
        mt2.append(mrp)
        mt3.append(mrr)
        mt4.append(mrt)
        mt5.append(mtp)
        mt6.append(mtt)

    data.with_columns(mpp=mt1, mrp=mt2, mrr=mt3, mrt=mt4, mg5=mtp, mtt=mt6)
    data.to_csv(output_filename)


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
