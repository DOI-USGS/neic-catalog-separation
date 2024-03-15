#!/usr/bin/env python3
# stdlib imports
import argparse
import csv
from math import sqrt

import matplotlib.pyplot as plt

# third party imports
import numpy as np


def bvalue(f_name, Mc_est, binsize):
    """
    Calulate b-value and goodness of fit for a given earthquake catalog
    Args:
        f_name (str): input file name. This file should be 1 column of magnitude values pertaining to a particular earthquake catalog (no header)
        Mc_est (float): Starting/estimate Mc (magnitude of completeness)
        binsize (float): Earthquake bin size
    Returns:
        bValueInfo.xyz (output file): output data with b-value and related values
        bvalue.png (figure): Plot of b-value and goodness of fit
    """
    Mc_est = float(Mc_est)
    binsize = float(binsize)
    mag = []
    with open(f_name) as file:
        reader = csv.reader(file, delimiter=" ", skipinitialspace=True)
        for one in reader:
            one = one[0]
            mag.append(float(one))

    mag = np.array(mag)
    max_mag = max(mag)

    # if observed list of mags is empty, then dont run b-value code & set everything to 0
    # else, run through b-value calculation function
    if not mag.all():
        print("empty magnitude array, setting values to 0")
        mcval = 0
        bval = 0
        avalue = 0
        lval = 0
        bins = 0
        std_dev = 0
        error_sb = 0
        rvalue = 0
    else:
        [mcval, bval, avalue, lval, bins, std_dev, error_sb, rvalue] = calculate_ww2000(
            Mc_est, mag, binsize
        )
    # determine goodness-of-Fit
    if not mag.all():
        gof = 0
    else:
        gof = 100 - rvalue
    # print b-value and related values to screen
    print("b-value:", bval)
    print("a-value:", avalue)
    print("Correct Mc:", mcval)
    print("Residual:", rvalue)
    print("Goodness-of-fit:", gof)
    print("Standard Deviation:", std_dev)
    print("Shi and Bolt bvalue Error:", error_sb)

    # save results to ascii file
    outfile = open(f"bValueInfo.xyz", "w")
    outfile.write(
        "b-value, a-value, Mc, Std. Deviation, Shi-Bolt Error, Goodness-of-fit\n"
    )
    outfile.write(f"{bval},{avalue},{mcval},{std_dev},{error_sb},{gof}")
    outfile.close()

    # if the observed mag list is empty, then exit the program, else plot b-value and gof
    if not mag.all():
        exit()
    else:
        make_fig(mag, mcval, bins, bval, gof, f_name)


# modified from USGS code for earthquake catalog quality check (https://code.usgs.gov/ghsc/neic/utilities/catalog-qc)
def calculate_ww2000(est_mcval, mags, binsize):
    """Wiemer and Wyss (2000) method for determining a and b values.
        This is a function is modified from neic-catalog-qc/QCutils.py
    Args:
        est_mcval: An estimated value for the minimum magnitude of complete
            recording.
        mags: A list of magnitudes to calculate completeness for.
        binsize: The bin size to use for calculating completeless.

    Returns:
        mcval: The calculated minimum magnitude of complete recording.
        bvalue: The calculated b value associated with mcval.
        avalue: The calculated a value associated with mcval.
        lval: A synthetic Frequency Magnitude Distribution (FMD) determined using bvalue and avalue.
        mag_bins: A list of magnitudes used for binning.
        std_dev: The calculated standard deviation of bvalue.
    """
    mags = mags[~np.isnan(mags)]
    mags = np.around(mags, 1)

    # only look for possible Mc within this range
    mc_range = 1.0
    mc_vec = np.arange(
        est_mcval - mc_range, est_mcval + mc_range + binsize / 2.0, binsize
    )

    max_mag = max(mags)
    corr = binsize / 2.0
    bvalue = np.zeros(len(mc_vec))
    std_dev = np.zeros(len(mc_vec))
    avalue = np.zeros(len(mc_vec))
    rval = np.zeros(len(mc_vec))
    error_sb = np.zeros(len(mc_vec))

    for idx, _ in enumerate(mc_vec):
        # magnitudes above current magnitude threshold
        mval = mags[mags >= mc_vec[idx] - 0.001]
        if len(mval) == 0:
            continue

        mag_bins_edges = np.arange(
            mc_vec[idx] - binsize / 2.0, max_mag + binsize, binsize
        )
        mag_bins_centers = np.arange(mc_vec[idx], max_mag + binsize / 2.0, binsize)

        # number of magnitudes above the current magnitude threshold/Mc
        cdf = np.zeros(len(mag_bins_centers))

        for jdx, _ in enumerate(cdf):
            cdf[jdx] = np.count_nonzero(
                ~np.isnan(mags[mags >= mag_bins_centers[jdx] - 0.001])
            )

        # Calculate a- and b-values and b-value error
        bvalue[idx] = np.log10(np.e) / (
            np.average(mval) - (mc_vec[idx] - corr)
        )  # Utsu, 1965 - binned/corrected version of Aki, 1965
        std_dev[idx] = bvalue[idx] / sqrt(
            cdf[0]
        )  # b-value error from Aki, 1965 and Utsu, 1966, 1966
        error_sb[idx] = (
            2.3
            * (bvalue[idx] ** 2)
            * sqrt((sum((mval - np.mean(mval)) ** 2)) / (cdf[0] * (cdf[0] - 1)))
        )  # b-value error from Shi and Bolt, 1982
        avalue[idx] = np.log10(len(mval)) + bvalue[idx] * mc_vec[idx]

        # calculate synthetic FMD using a- and b-values from observed data
        # bval is the observed cumulative no. of events in each bin and sval is the synthetic/predicted cumulative no. of events in each bin
        log_l = avalue[idx] - bvalue[idx] * mag_bins_edges  # log_10 n = a - b*M
        lval = (
            10.0**log_l
        )  # solving for N(M) = no. of EQs >= mag. M, which here is >= mag_bins_centers
        bval, _ = np.histogram(
            mval, mag_bins_edges
        )  # bval = no of observed events in each bin
        # print('np.histogram(mval, mag_bins_edges)',np.histogram(mval, mag_bins_edges))
        sval = abs(np.diff(lval))  # sval = no of synthetic events in each bin

        # determine goodness-of-fit by taking the absolute difference between number of synthetic and observed events in each magnitude bin
        # This R, or residual, percentage determines the fit to the observed FMD
        rval[idx] = (sum(abs(bval - sval)) / len(mval)) * 100

    # determine the index in which 90% or more of the observed data are modeled by a straight line
    # i.e., determine where the Residual is 10 or less (see Figure 3b in Woessner and Wiemer (2005))
    # 95% is ideal, but difficult to obtain, and so 90% is a compromise
    # however, that can still be hard to meet, so here if that percentage can NOT met, then take the lowest R
    ind = np.where(rval <= 10)[0]
    if len(ind) != 0:
        idx = ind[0]
    else:
        idx = list(rval).index(min(rval))

    mcval = mc_vec[idx]
    bvalue = bvalue[idx]
    avalue = avalue[idx]
    std_dev = std_dev[idx]
    error_sb = error_sb[idx]
    rvalue = rval[idx]
    mag_bins = np.arange(0, max_mag + binsize / 2.0, binsize)
    lval = 10.0 ** (avalue - bvalue * mag_bins)

    return mcval, bvalue, avalue, lval, mag_bins, std_dev, error_sb, rvalue


def make_fig(mag, mcval, bins, bval, gof, f_name):
    """
    Plot b-value and goodness-of-fit (gof)
    Args:
        mag (numpy array): Magnitudes
        mcval (numpy array of floats): Predicted Mc
        bins (numpy array): bings
        bval (numpy array of floats): Calculated b-value
        gof (numpy array of floats): Calculated goodness-of-fit
        f_name (str): name of input magnitude file
    Returns:
        bvalue.png (figure): Plot of b-value and goodness of fit
    """
    # count the number of events in each bin & make some plots:
    rmag = mag[~np.isnan(mag)]
    rmag = np.around(rmag, 1)

    mcvalr = np.around(mcval, 1)

    index = []
    count = np.zeros(len(bins))

    for i in range(len(rmag)):
        for j in range(len(bins)):
            if rmag[i] == bins[j]:
                count[j] = count[j] + 1
            if np.around(bins[j], 1) == mcvalr:
                index = j  # Index where in the bins the Magnitude of completeness is

    # Create a vertical line for the Magnitude of Completeness
    y = np.arange(0, np.log10(max(count)) + 1, 0.5)
    mcval_ar = np.ones(len(y)) * mcval

    # Make the b-value line
    x = np.arange(mcvalr, 8.0, 0.1)
    if count[index] != 0:
        line = x * -1 * bval + (np.log10(count[index]) + mcvalr * bval)
    else:
        up = 0
        down = 0
        while count[index + up] == 0:
            up = up + 1
        while count[index - down] == 0:
            down = down + 1
        line = x * -1 * bval + (
            np.log10((count[index + up] + count[index - down]) / 2) + mcval * bval
        )

    # Make b-value and goodness of fit plot
    fig = plt.figure(figsize=(10, 6), dpi=120, frameon=True)
    plt.plot(x, line, "k")
    plt.plot(mcval_ar, y, "g:")  # Vertical cut off for magnitude of completeness
    plt.plot(bins, np.log10(count), "k.")  # plot number of EQs in each bin
    plt.text(6, 3 * np.log10(max(count)) / 4, "b-value = %f" % bval, fontsize=12)
    plt.text(6, 4 * np.log10(max(count)) / 4, "Goodness-of-Fit = %f" % gof, fontsize=12)
    plt.title("b-value for " + f_name)
    plt.xlabel("Magnitude", fontsize=11)
    plt.ylabel("Number of Earthquakes", fontsize=11)
    ytic = [0, 1, 2, 3]
    logtic = [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"]
    plt.yticks(ytic, logtic)
    plt.grid(True)
    plt.savefig(f"bvalue.png")


if __name__ == "__main__":
    desc = """Calculate b-value for a given catalog"""
    argparser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "file_name",
        help="Name of file is required for calculating b-value. This file should be 1 column of magnitude values pertaining to a particular earthquake catalog (no header).",
        metavar="file_name",
    )
    argparser.add_argument(
        "mc", help="Starting/estimated Mc (magnitude of completion)", metavar="mc"
    )
    argparser.add_argument(
        "binsize",
        help="Earthquake bin size. This is the ideal bin given the rounding; magnitudes for modern instrumental measurement usually binned/grouped by 0.1",
        metavar="binsize",
    )
    pargs, unknown = argparser.parse_known_args()
    f_name = pargs.file_name
    Mc_est = pargs.mc
    binsize = pargs.binsize
    bvalue(f_name, Mc_est, binsize)
