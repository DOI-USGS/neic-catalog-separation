# stdlib imports
import json
import pathlib
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

# third party imports
import polars


def calculate_bvalue(
    catalog_filename: pathlib.Path,
    mc_est: float,
    bin_size: float,
    show_plot: bool = True,
):
    """
    Calulate b-value and goodness of fit for a given earthquake catalog

    bValueInfo.xyz (output file): output data with b-value and related values
    bvalue.png (figure): Plot of b-value and goodness of fit


    Args:
        catalog_filename: Name of earthquake catalog file, must include column 'magnitude'.
        mc_est: Starting/estimate Mc (magnitude of completeness)
        bin_size: Earthquake bin size
    """
    eq_magnitude = polars.read_csv(catalog_filename)["magnitude"].to_numpy()

    if not eq_magnitude.any():
        # No magnitudes
        return

    stats, mag_bins = calculate_ww2000(
        mags=eq_magnitude, mc_est=mc_est, bin_size=bin_size
    )
    stats["gof"] = 100.0 - stats["residual"]

    # print b-value and related values to screen
    print(f"b-value: {stats['b_value']}")
    print(f"a-value: {stats['a_value']}")
    print(f"Computed Mc: {stats['mc']}")
    print(f"Residual: {stats['residual']}")
    print(f"Goodness-of-fit: {stats['gof']}")
    print(f"Standard Deviation: {stats['std_dev']}")
    print(f"Shi and Bolt b-value error: {stats['error_sb']}")

    # save results to JSON file
    filename = catalog_filename.replace(".csv", "_bvalue.json")
    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(stats, outfile)

    if show_plot:
        plot(
            mags=eq_magnitude,
            mag_bins=mag_bins,
            stats=stats,
            catalog_filename=catalog_filename,
        )


# modified from USGS code for earthquake catalog quality check (https://code.usgs.gov/ghsc/neic/utilities/catalog-qc)
def calculate_ww2000(mags: np.ndarray, mc_est: float, bin_size: float) -> tuple:
    """Wiemer and Wyss (2000) method for determining a and b values.
        This is a function is modified from neic-catalog-qc/QCutils.py
    Args:
        mc_est: An estimated value for the minimum magnitude of complete
            recording.
        mags: A list of magnitudes to calculate completeness for.
        bin_size: The bin size to use for calculating completeless.

    Returns:
        mc: The calculated minimum magnitude of complete recording.
        bvalue: The calculated b value associated with mc.
        avalue: The calculated a value associated with mc.
        lval: A synthetic Frequency Magnitude Distribution (FMD) determined using bvalue and avalue.
        mag_bins: A list of magnitudes used for binning.
        std_dev: The calculated standard deviation of bvalue.
    """
    mags = mags[~np.isnan(mags)]
    mags = np.around(mags, 1)

    # only look for possible Mc within this range
    mc_range = 1.0
    mag_bins = np.arange(
        mc_est - mc_range, mc_est + mc_range + bin_size / 2.0, bin_size
    )

    max_mag = np.max(mags)
    corr = bin_size / 2.0
    b_value = np.zeros_like(mag_bins)
    std_dev = np.zeros_like(mag_bins)
    a_value = np.zeros_like(mag_bins)
    residual = np.zeros_like(mag_bins)
    error_sb = np.zeros_like(mag_bins)

    for idx, mc in enumerate(mag_bins):
        # magnitudes above current magnitude threshold
        mval = mags[mags >= mc - 0.001]
        if len(mval) == 0:
            continue

        mag_bins_edges = np.arange(mc - bin_size / 2.0, max_mag + bin_size, bin_size)
        mag_bins_centers = np.arange(mc, max_mag + bin_size / 2.0, bin_size)

        # number of magnitudes above the current magnitude threshold/Mc
        cdf = np.zeros_like(mag_bins_centers)

        for jdx, _ in enumerate(cdf):
            cdf[jdx] = np.count_nonzero(
                ~np.isnan(mags[mags >= mag_bins_centers[jdx] - 0.001])
            )

        # Calculate a- and b-values and b-value error
        b_value[idx] = np.log10(np.e) / (
            np.average(mval) - (mc - corr)
        )  # Utsu, 1965 - binned/corrected version of Aki, 1965
        std_dev[idx] = b_value[idx] / sqrt(
            cdf[0]
        )  # b-value error from Aki, 1965 and Utsu, 1966, 1966
        error_sb[idx] = (
            2.3
            * (b_value[idx] ** 2)
            * sqrt((sum((mval - np.mean(mval)) ** 2)) / (cdf[0] * (cdf[0] - 1)))
        )  # b-value error from Shi and Bolt, 1982
        a_value[idx] = np.log10(len(mval)) + b_value[idx] * mc

        # calculate synthetic FMD using a- and b-values from observed data
        # bval is the observed cumulative no. of events in each bin and sval is the synthetic/predicted cumulative no. of events in each bin
        log_l = a_value[idx] - b_value[idx] * mag_bins_edges  # log_10 n = a - b*M
        # solving for N(M) = no. of EQs >= mag. M, which here is >= mag_bins_centers
        lval = 10.0**log_l
        # bval = number of observed events in each bin
        bval, _ = np.histogram(mval, mag_bins_edges)
        # print('np.histogram(mval, mag_bins_edges)',np.histogram(mval, mag_bins_edges))
        sval = abs(np.diff(lval))  # sval = no of synthetic events in each bin

        # determine goodness-of-fit by taking the absolute difference between number of synthetic and observed events in each magnitude bin
        # This R, or residual, percentage determines the fit to the observed FMD
        residual[idx] = (sum(abs(bval - sval)) / len(mval)) * 100

    # determine the index in which 90% or more of the observed data are modeled by a straight line
    # i.e., determine where the Residual is 10 or less (see Figure 3b in Woessner and Wiemer (2005))
    # 95% is ideal, but difficult to obtain, and so 90% is a compromise
    # however, that can still be hard to meet, so here if that percentage can NOT met, then take the lowest R
    ind = np.where(residual <= 10)[0]
    if len(ind) != 0:
        idx = ind[0]
    else:
        idx = np.argmin(residual)

    stats = {
        "mc": mag_bins[idx],
        "b_value": b_value[idx],
        "a_value": a_value[idx],
        "lval": lval,
        "std_dev": std_dev[idx],
        "residual": residual[idx],
        "error_sb": error_sb[idx],
    }
    lval = 10.0 ** (a_value - b_value * mag_bins)
    mag_bins = np.arange(0, max_mag + bin_size / 2.0, bin_size)

    return (stats, mag_bins)


def plot(
    mags: np.ndarray, mag_bins: np.ndarray, stats: dict, catalog_filename: pathlib.Path
):
    """
    Plot b-value and goodness-of-fit (gof)
    Args:
        mag (numpy array): Magnitudes
        mc (numpy array of floats): Predicted Mc
        bins (numpy array): bings
        bval (numpy array of floats): Calculated b-value
        gof (numpy array of floats): Calculated goodness-of-fit
        f_name (str): name of input magnitude file
    Returns:
        bvalue.png (figure): Plot of b-value and goodness of fit
    """
    # count the number of events in each bin & make some plots:
    rmag = mags[~np.isnan(mags)]
    rmag = np.around(rmag, 1)
    mc = stats["mc"]
    b_value = stats["b_value"]

    mcvalr = np.around(stats["mc"], 1)

    index = []
    count = np.zeros(len(mag_bins))

    for i in range(len(rmag)):
        for j in range(len(mag_bins)):
            if rmag[i] == mag_bins[j]:
                count[j] = count[j] + 1
            if np.around(mag_bins[j], 1) == mcvalr:
                index = (
                    j  # Index where in the mag_bins the Magnitude of completeness is
                )

    # Make the b-value line
    x = np.arange(mcvalr, 8.0, 0.1)
    if count[index] != 0:
        line = x * -1 * b_value + (np.log10(count[index]) + mcvalr * b_value)
    else:
        up = 0
        down = 0
        while count[index + up] == 0:
            up = up + 1
        while count[index - down] == 0:
            down = down + 1
        line = x * -1 * b_value + (
            np.log10((count[index + up] + count[index - down]) / 2) + mc * b_value
        )

    # Make b-value and goodness of fit plot
    figure = plt.figure(figsize=(10, 6), dpi=120, frameon=True)
    ax = figure.axes()
    ax.plot(x, line, "k")
    ax.avline(mc, "g:")  # Vertical cut off for magnitude of completeness
    ax.plot(mag_bins, np.log10(count), "k.")  # plot number of EQs in each bin
    ax.text(6, 0.75 * np.log10(max(count)), f"b-value = {b_value}", fontsize=12)
    ax.text(
        6, 1.00 * np.log10(max(count)), f"Goodness-of-Fit = {stats['gof']}", fontsize=12
    )
    ax.title(f"b-value for {catalog_filename}")
    ax.xlabel("Magnitude", fontsize=11)
    ax.ylabel("Number of Earthquakes", fontsize=11)
    ytic = [0, 1, 2, 3]
    logtic = [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"]
    ax.yticks(ytic, logtic)
    ax.grid(True)
    filename = catalog_filename.replace(".csv", "_bvalue.pdf")
    plt.savefig(filename)


def cli():
    """Command line interface."""
    import argparse

    desc = """Calculate b-value for a given catalog"""
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "--catalog",
        action="store",
        dest="catalog",
        required=True,
        help="Name of earthquake catalog file with a column 'magnitude'",
    )
    argparser.add_argument(
        "--completeness-magnitude",
        action="store",
        dest="completeness_magnitude",
        type=float,
        required=True,
        help="Starting/estimated Mc (magnitude of completion)",
    )
    argparser.add_argument(
        "--bin-size",
        action="store",
        dest="bin_size",
        type=float,
        default=0.1,
        help="Earthquake bin size given prevision of the magnitudes in the catalog, usually 0.1",
    )
    argparser.add_argument(
        "--plot",
        action="store_true",
        dest="show_plot",
        help="Generate plot showing fit of b_value and magnitude of completeness.",
    )

    args = argparser.parse_args()
    calculate_bvalue(
        catalog_filename=args.catalog,
        mc_est=args.completeness_magnitude,
        bin_size=args.bin_size,
        show_plot=args.show_plot,
    )
