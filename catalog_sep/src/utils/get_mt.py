import json
from urllib.request import urlopen
import pandas as pd
import time


def get_mt_from_comcat(input):
    """Query the USGS earthquake catalog, ComCat, for moment tensors for event ID's provided in a catalog CSV file

    Args:
        input (str): name of input earthquake catalog file
    """


# read in catalog csv file and get event IDs:
data = pd.read_csv(input, low_memory=False)
event_id = data["eventid"].values
mt1 = []
mt2 = []
mt3 = []
mt4 = []
mt5 = []
mt6 = []

# loop through event IDs and get moment tensor products:
for i in event_id:
    # avoid rate limit of 500 requests per 5 minutes for FDSN requests
    time.sleep(1)
    print("ID:", i)
    id_url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={}" "&format=geojson"
    ).format(i)
    try:
        url_lines = [x.decode("utf-8") for x in urlopen(id_url)]
        bigjson = json.loads("".join(url_lines))
    except:
        print(f"Cant connect to url {id_url} for event {i}")
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
    except:
        mpp = float("nan")
        mrp = float("nan")
        mrr = float("nan")
        mrt = float("nan")
        mtp = float("nan")
        mtt = float("nan")
        print(mpp)
    mt1.append(mpp)
    mt2.append(mrp)
    mt3.append(mrr)
    mt4.append(mrt)
    mt5.append(mtp)
    mt6.append(mtt)

data["mpp"] = mt1
data["mrp"] = mt2
data["mrr"] = mt3
data["mrt"] = mt4
data["mtp"] = mt5
data["mtt"] = mt6

data.to_csv("MomentTensor.csv", mode="w", index=False, na_rep="nan")

if __name__ == "__main__":
    desc = """
    Query ComCat for moment tensors for event ID's provided in a catalog CSV file
    """
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "input_file",
        help="Name of input earthquake catalog file to get moment tensors for",
    )

    pargs, unknown = argparser.parse_known_args()
    input = pargs.input_file
    get_mt_from_comca(input)
