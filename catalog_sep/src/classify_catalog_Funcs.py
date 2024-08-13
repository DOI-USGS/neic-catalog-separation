# stdlib imports
import csv
import glob
from copy import deepcopy
from math import asin, cos, sqrt

import numpy as np

# third party imports
from mapio.reader import read


def find_closest_loc(lon, lat, supLon, supLat, supDepth, supStr, supDip):
    """
    Determine nearest lon,lat from titled/vertical slab data to earthquake location and take titled slab depth, strike, and dip at that point
    Args:
        lon (float): Earthquake (EQ) lon
        lat (float): EQ lat
        supLon (list): lon floats from titled data
        supLat (list): lat floats from titled data
        supDepth (list): depth floats from titled data
        supStr (list): strike floats from titled data
        supDip (list): dip floats from titled data
    Returns:
        nearLon (float): closest lon
        nearLat (float): closest lat
        nearDepth (float): closest depth
        nearStr (float): closest strike
        nearDip (float):closest dip
    """
    p = 0.017453292519943295  # Pi/180
    dist = []
    dist2 = np.empty((len(supLon), 6))
    # use the Haversine formula to determine the great circle distance between 2 points on a sphere given lon,lat of each point:
    for i in range(len(supLon)):
        dist = (
            0.5
            - cos((supLat[i] - lat) * p) / 2
            + cos(lat * p) * cos(supLat[i] * p) * (1 - cos((supLon[i] - lon) * p)) / 2
        )
        dist = 12742 * asin(sqrt(dist))
        dist2[i] = np.hstack(
            (supLon[i], supLat[i], dist, supDepth[i], supStr[i], supDip[i])
        )
    # find index corresponding to minimum distance & get data at that index
    minDist = np.where(dist2 == np.min(dist2, axis=0)[2])
    nearLon = dist2[minDist[0], 0]
    nearLat = dist2[minDist[0], 1]
    nearDepth = dist2[minDist[0], 3]
    nearStr = dist2[minDist[0], 4]
    nearDip = dist2[minDist[0], 5]

    # Convert from numpy array to python class float
    nearDepth = nearDepth.item()
    nearStr = nearStr.item()
    nearDip = nearDip.item()
    minDistv = np.min(dist2, axis=0)[2].item()

    return nearLon, nearLat, nearDepth, nearStr, nearDip, minDistv


# Calculate probability using ramp function
# Taken from ShakeMap (Worden et al., 2005) and STREC (https://code.usgs.gov/ghsc/esi/strec)
def get_probability(x, x1, p1, x2, p2):
    """Calculate probability using a ramped function.
    The subsections and parameters below reflect a series of ramp functions
    we use to calculate various probabilities.
        p1  |----+
            |     \
            |      \
            |       \
        p2  |        +-------
            |
            +-----------------
                x1  x2
    Args:
        x (float): Quantity for which we want corresponding probability.
        x1 (float): Minimum X value.
        p1 (float): Probability at or below minimum X value.
        x2 (float): Maximum X value.
        p2 (float): Probability at or below maximum X value.
    Returns:
        float: Probability at input x value.
    """
    print(x, x1, p1, x2, p2)
    if x <= x1:
        prob = p1
    elif x >= x2:
        prob = p2
    else:
        slope = (p1 - p2) / (x1 - x2)
        intercept = p1 - slope * x1
        prob = x * slope + intercept
    print("PROB NUMBER 1 IS", prob)
    return prob


# From Slab2 source code (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)
def get_kagan_angle(strike1, dip1, rake1, strike2, dip2, rake2):
    """Calculate the Kagan angle between two moment tensors defined by strike,dip and rake.

    Kagan, Y. "Simplified algorithms for calculating double-couple rotation",
    Geophysical Journal, Volume 171, Issue 1, pp. 411-418.

    Args:
        strike1 (float): strike of slab
        dip1 (float): dip of slab
        rake1 (float): rake of slab
        strike2 (float): strike of Eq moment tensor
        dip2 (float): dip of EQ moment tensor
        rake2 (float): rake of EQ moment tensor
    Returns:
        float: Kagan angle between two moment tensors
    """
    # convert from strike, dip , rake to moment tensor
    tensor1 = plane_to_tensor(strike1, dip1, rake1)
    tensor2 = plane_to_tensor(strike2, dip2, rake2)

    kagan = calc_theta(tensor1, tensor2)
    return kagan


# From Slab2 source code (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)
def plane_to_tensor(strike, dip, rake, mag=6.0):
    """Convert strike,dip,rake values to moment tensor parameters.

    Args:
        strike (float): Strike from (assumed) first nodal plane (degrees).
        dip (float): Dip from (assumed) first nodal plane (degrees).
        rake (float): Rake from (assumed) first nodal plane (degrees).
        magnitude (float): Magnitude for moment tensor
            (not required if using moment tensor for angular comparisons.)
    Returns:
        nparray: Tensor representation as 3x3 numpy matrix:
            [[mrr, mrt, mrp]
            [mrt, mtt, mtp]
            [mrp, mtp, mpp]]
    """
    # define degree-radian conversions
    d2r = np.pi / 180.0
    r2d = 180.0 / np.pi

    # get exponent and moment magnitude
    magpow = mag * 1.5 + 16.1
    mom = np.power(10, magpow)

    # get tensor components
    mrr = mom * np.sin(2 * dip * d2r) * np.sin(rake * d2r)
    mtt = -mom * (
        (np.sin(dip * d2r) * np.cos(rake * d2r) * np.sin(2 * strike * d2r))
        + (
            np.sin(2 * dip * d2r)
            * np.sin(rake * d2r)
            * (np.sin(strike * d2r) * np.sin(strike * d2r))
        )
    )
    mpp = mom * (
        (np.sin(dip * d2r) * np.cos(rake * d2r) * np.sin(2 * strike * d2r))
        - (
            np.sin(2 * dip * d2r)
            * np.sin(rake * d2r)
            * (np.cos(strike * d2r) * np.cos(strike * d2r))
        )
    )
    mrt = -mom * (
        (np.cos(dip * d2r) * np.cos(rake * d2r) * np.cos(strike * d2r))
        + (np.cos(2 * dip * d2r) * np.sin(rake * d2r) * np.sin(strike * d2r))
    )
    mrp = mom * (
        (np.cos(dip * d2r) * np.cos(rake * d2r) * np.sin(strike * d2r))
        - (np.cos(2 * dip * d2r) * np.sin(rake * d2r) * np.cos(strike * d2r))
    )
    mtp = -mom * (
        (np.sin(dip * d2r) * np.cos(rake * d2r) * np.cos(2 * strike * d2r))
        + (0.5 * np.sin(2 * dip * d2r) * np.sin(rake * d2r) * np.sin(2 * strike * d2r))
    )

    mt_matrix = np.array([[mrr, mrt, mrp], [mrt, mtt, mtp], [mrp, mtp, mpp]])
    mt_matrix = mt_matrix * 1e-7  # convert from dyne-cm to N-m
    return mt_matrix


# From Slab2 source code (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)
def calc_theta(vm1, vm2):
    """Calculate angle between two moment tensor matrices.
    Args:
        vm1 (ndarray): Moment Tensor matrix (see plane_to_tensor).
        vm2 (ndarray): Moment Tensor matrix (see plane_to_tensor).
    Returns:
        float: Kagan angle (degrees) between input moment tensors.
    """
    # calculate the eigenvectors of either moment tensor
    V1 = calc_eigenvec(vm1)
    V2 = calc_eigenvec(vm2)

    # find angle between rakes
    th = ang_from_R1R2(V1, V2)

    # calculate kagan angle and return
    for j in range(3):
        k = (j + 1) % 3
        V3 = deepcopy(V2)
        V3[:, j] = -V3[:, j]
        V3[:, k] = -V3[:, k]
        x = ang_from_R1R2(V1, V3)
        if x < th:
            th = x
    return th * 180.0 / np.pi


# From Slab2 source code (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)
def calc_eigenvec(TM):
    """Calculate eigenvector of moment tensor matrix.
    Args:
        ndarray: moment tensor matrix (see plane_to_tensor)
    Returns:
        ndarray: eigenvector representation of input moment tensor.
    """
    # calculate eigenvector
    V, S = np.linalg.eigh(TM)
    inds = np.argsort(V)
    S = S[:, inds]
    S[:, 2] = np.cross(S[:, 0], S[:, 1])
    return S


# From Slab2 source code (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)
def ang_from_R1R2(R1, R2):
    """Calculate angle between two eigenvectors.
    Args:
        R1 (ndarray): eigenvector of first moment tensor
        R2 (ndarray): eigenvector of second moment tensor
    Returns:
        float: angle between eigenvectors
    """
    return np.arccos((np.trace(np.dot(R1, R2.transpose())) - 1.0) / 2.0)


def overturned_slab(smod, working_dir, lon, lat):
    """
    Find nearest lon,lat to EQ lon,lat and extract the corresponding overturned/tilted slab depth, strike, and dip
    Args:
        smod (str): slab model
        working_dir (str): current working directory (used for looking for data/files)
        lon (float): lon of EQ
        lat (float): lat of EQ
    Returns:
        sdep (float): slab depth
        sstr (float): slab strike
        sdip (float): slab dip
    """
    print(
        "Slab depth in this area is NaN...If in an overturned region using that data..."
    )
    if smod == "man" or smod == "ker" or smod == "izu" or smod == "sol":
        slabdir = f"{working_dir}/catalog_sep/Input/Slab2Catalogs/"
        fpath = glob.glob(slabdir + smod + "_slab2_sup*.csv")
        fpath = fpath[0]
        supLon = []
        supLat = []
        supDepth = []
        supStr = []
        supDip = []
        with open(fpath) as file:
            reader = csv.reader(file, delimiter=",", skipinitialspace=True)
            next(reader)
            for one, two, three, four, five, six, seven, eight, nine in reader:
                supLon.append(float(one))
                supLat.append(float(two))
                supDepth.append(float(three))
                supStr.append(float(four))
                supDip.append(float(five))
        sdep, sstr, sdip, minDist = funcs.find_closest_loc(
            lon, lat, supLon, supLat, supDepth, supStr, supDip
        )
        # dont want to classify events that are far from the tilted slab (such as outer rise or events that may not have been filtered out), so correct for that here
        if minDist > 1:
            sdep = np.nan
    else:

        sdep = np.nan
        sstr = np.nan
        sdip = np.nan

    return sdep, sstr, sdip


def equal_prob(
    p_crustal, p_int, p_slab, mohoDepth, s1, sstr, sdep, kagan, ddiff, dlim, depth
):
    """
    If there is an equal probability, determine most likely classification.
    Args:
        p_crustal (float): probability the earthquake is crustal
        p_int (float): probability the earthquake occurred along the subduction interface
        p_slab (float): probability the earthquake is intraslab
        mohoDepth (float): Moho depth at location of earthquake
        s1 (float): earthquakes strike at nodal plane 1
        sstr (float): slabs strike at the earthquakes location
        sdep (float): slabs depth at the earthquakes location
        kagan (float): kagan angle between the fault plane and slab
        ddiff (float): depth difference between the earthquake and slab
        dlim (int): deep seismogenic limit + flex
        depth (float): depth of earthquake

    Returns:
        eqloc (str): most likely earthquake classification with 'eq' (meaning 'equal') appended to the end of the classification
    """
    if p_crustal == p_int:
        # If MT (moment tensor/nodal plane info) exists, use thresholds to determine if interface
        # If no MT exists, then use depth - if closer to moho than slab, then crustal, else interface
        diffMoho = np.abs(depth - mohoDepth)
        if s1 > 0 and sstr > 0:
            if kagan <= 30:
                eqloc = "ieq"
            else:
                eqloc = "ueq"
        elif diffMoho < ddiff:
            eqloc = "ueq"
        else:
            eqloc = "ieq"

    elif p_crustal == p_slab:
        # if closer to moho than slab = crustal, else slab
        diffMoho = np.abs(depth - mohoDepth)
        if diffMoho < ddiff:
            eqloc = "ueq"
        else:
            eqloc = "leq"

    elif p_slab == p_int:
        # If MT exists, use thresholds to determine if interface
        # If no MT exists, then if less than SZ depth & depth to slab surf is within 20 km = interface
        if s1 > 0 and sstr > 0:
            if kagan <= 30:
                eqloc = "ieq"
            else:
                eqloc = "leq"
        elif depth <= dlim and ddiff <= 20:
            eqloc = "ieq"
        else:
            eqloc = "leq"

    elif p_slab == p_int == p_crustal and p_slab > 0:
        # If MT exists, use thresholds to determine if interface
        # If no MT, then use depths
        diffMoho = np.abs(depth - mohoDepth)
        if s1 > 0 and sstr > 0:
            if kagan <= 30:
                eqloc = "ieq"
        else:
            if depth > sdep and depth > dlim:
                eqloc = "leq"
            elif depth <= dlim and ddiff <= 20:
                eqloc = "ieq"
            elif diffMoho < ddiff:
                eqloc = "ueq"
            else:
                eqloc = "unknown"
    return eqloc


def flag_mantle(s1, sstr, depth, dlim, sdep, sunc, Ppl, Tpl, eqloc):
    """
    Flag any possible mantle events.
    Args:
        s1 (float): earthquakes strike at nodal plane 1
        sstr (float): slabs strike at the earthquakes location
        depth (float): depth of earthquake
        dlim (int): deep seismogenic limit + flex
        sdep (float): slabs depth at the earthquakes location
        unc (float): slabs depth uncertainty at the earthquakes location
        Ppl (float): earthquakes P-axis plunge
        Tpl (float): earthquakes T-axis plunge
        eqloc (str): most likely earthquake classification
    Returns:
        eqloc (str): most likely earthquake classification with 'm' (meaning 'mantle') appended to the end of the classification
    """
    if s1 > 0 and sstr > 0:
        if depth > dlim and depth <= (sdep - sunc) and Ppl > 45:
            eqloc = "m" + eqloc
        elif depth > dlim and depth <= (sdep - sunc) and Tpl > 20:
            eqloc = "m" + eqloc
    elif depth > dlim and depth <= (sdep - sunc):
        eqloc = "m" + eqloc
    else:
        eqloc = eqloc
    return eqloc


def determine_quality(eqloc, p_crustal, p_slab, p_int, s1, sstr):
    """
    Determine quality (qual) of earthquake classification
    if kagan angle/MT is available: probability >=80% qual = A; 60-80% qual = B; 40-60% qual = C; <40% qual = D; if eq qual = D
    if kagan angle/MT not available: probability >=80% qual = B; 60-80% qual = C; <60% = D
    Args:
        eqloc (str): most likely earthquake classification
        p_crustal (float): probability the earthquake is crustal
        p_slab (float): probability the earthquake is intraslab
        p_int (float): probability the earthquake occurred along the subduction interface
        s1 (float): earthquakes strike at nodal plane 1
        sstr (float): slabs strike at the earthquakes location
    Returns:
        qual (str): quality/grade of earthquake classification (A, B, C, D)
    """
    print("DETERMINE QUALITY:", p_crustal, p_slab, p_int)
    if eqloc == "u":
        prob = p_crustal
    elif eqloc == "l":
        prob = p_slab
    elif eqloc == "i":
        prob = p_int
    if "eq" in eqloc:
        qual = "D"
        return qual
    if "m" in eqloc:
        qual = "D"
        return qual
    if "or" in eqloc:
        qual = "D"
        return qual
    if s1 > 0 and sstr > 0:
        print("PROB IS", prob)
        if prob >= 80:
            qual = "A"
        elif prob >= 60 and prob < 80:
            qual = "B"
        elif prob >= 40 and prob < 60:
            qual = "C"
        else:
            qual = "D"
    else:
        print("PROB IS", prob)
        if prob >= 80:
            qual = "B"
        elif prob >= 60 and prob < 80:
            qual = "C"
        else:
            qual = "D"
    return qual


# Function modified from STREC (https://code.usgs.gov/ghsc/esi/strec)
def get_slab_moho_info(smod, working_dir, lat, lon):
    """Return a dictionary with slab2 depth, dip, strike, and depth uncertainty at location of earthquake
    Args:
        smod (str): slab2 region
        working_dir (str): current working directory (used for looking for data/files)
        lat (float): earthquake lat
        lon (float): earthquake lon
    Returns:
        abs(sdep) (float): positive slab2 depth at earthquake location
        sstr (float): slab2 strike at earthquake location
        sdip (float): slab2 dip at earthquake location
        sunc (float): slab2 depth uncertainty at earthquake location
        abs(mohoDepth) (float): positive Moho depth from Crust1.0 at earthquake location
    """
    # get published Slab2 grid files
    slabdir = f"{working_dir}/catalog_sep/Slab2/"
    slabmod = glob.glob(slabdir + smod + "_slab2_dep*.grd")
    slabumod = glob.glob(slabdir + smod + "_slab2_unc*.grd")
    slabsmod = glob.glob(slabdir + smod + "_slab2_str*.grd")
    slabdmod = glob.glob(slabdir + smod + "_slab2_dip*.grd")
    slabmod = slabmod[0]
    slabsmod = slabsmod[0]
    slabdmod = slabdmod[0]
    slabumod = slabumod[0]

    # define path to crust1.0 moho data:
    moho = f"{working_dir}/catalog_sep/Input/crust1.0/depthtomoho.grd"

    # load the slab2 grid files
    depth_grid = read(slabmod)
    error_grid = read(slabumod)
    dip_grid = read(slabdmod)
    strike_grid = read(slabsmod)
    moho_grid = read(moho)

    # Try to get grid values at event lat, lon
    try:
        sdep = depth_grid.getValue(lat, lon)
        sunc = error_grid.getValue(lat, lon)
        sdip = dip_grid.getValue(lat, lon)
        sstr = strike_grid.getValue(lat, lon)
        sstr = sstr
        if sstr < 0:
            sstr += 360
        if lon > 180:
            lon = lon - 360
        mohoDepth = moho_grid.getValue(lat, lon)
    # except = outside of slab2 region
    except:
        sdep = np.nan
        sstr = np.nan
        sdip = np.nan
        sunc = np.nan
        try:
            mohoDepth = moho_grid.getValue(lat, lon)
        except:
            mohoDepth = np.nan

    return abs(sdep), sstr, sdip, sunc, abs(mohoDepth)


def write_file(df, outfile):
    """Write dataframe to csv file

    Args:
        df (pandas dataframe): dataframe containing information to output
        outfile (str): name of output file
    """
    # only keep rows where p_crustal is not NaN, else entire df will be output
    df = df[df["p_crustal"].notna()]
    # save to csv file
    df.to_csv(
        f"{outfile}.csv",
        encoding="utf-8",
        index=False,
        header=False,
        mode="a",
    )


def get_seismogenic_depth(working_dir, slab, nshm):
    """Get seismogenic zone thickness/depth

    Args:
        working_dir (str): current working directory (used for looking for data/files)
        slab (str): 3-letter code of slab2 region
        nshm (boolean): If running separation for USGS NSHM the this is True, else False (default)
    Return:
        sz_deep (int): depth to the deep limit of the seismogenic zone based on slab2
    """
    if nshm:
        if slab == "car":
            sz_deep = 50
        elif slab == "mue":
            sz_deep = 40
    else:
        # Initialize empty variables to obtain from seismogenic zone thickness file
        scode = []
        sd = []
        arak = []

        # get published Slab2 seismogenic thickness file
        sztfile = f"{working_dir}/catalog_sep/Input/szt.txt"
        count = 0
        with open(sztfile) as file:
            reader = csv.reader(file, delimiter=",", skipinitialspace=True)
            next(reader)
            for (
                one,
                two,
                three,
                four,
                five,
                six,
                seven,
                eight,
                nine,
                ten,
                eleven,
            ) in reader:
                scode.append(str(three))
                sd.append(float(six))
                arak.append(float(ten))
                if scode[count] == slab:
                    sz_deep = sd[count]
                    srake = arak[count]
                    break
                # else, use a default value
                else:
                    sz_deep = 40
                count = count + 1

    return sz_deep


def determine_closest_slab(slab1, slab2, working_dir, lat, lon, depth):
    """Determine which slab2 region to use

    Args:
        slab1 (str): 3-letter code of slab2 region
        slab2 (str): 3-letter code of second slab2 region
        working_dir (str): current working directory (used for looking for data/files)
        lat (float): latitude of earthquake hypocenter
        lon(float): longitude of earthquake hypocenter
        depth (float): depth (km) of earthquake hypocenter
    Return:
        smod (str): 3-letter code of the slab2 region to use
    """
    # get slab info for slab1:
    sdep1, sstr1, sdip1, sunc1, mohoDepth1 = get_slab_moho_info(
        slab1, working_dir, lat, lon
    )
    # get depth difference betwen slab1 depth and earthquake depth
    slab1_ddiff = np.abs(depth - sdep1)
    # get slab info for slab2:
    sdep2, sstr2, sdip2, sunc2, mohoDepth2 = get_slab_moho_info(
        slab2, working_dir, lat, lon
    )
    # get depth different between slab2 depth and earthquake depth
    slab2_ddiff = np.abs(depth - sdep2)

    # if both depths are nan, then the earthquake is outside of slab2 region, and is likely crustal or outer rise
    # this will be classified accordingly, so we can default to the first slab region supplied
    if np.isnan(sdep1) and np.isnan(sdep2):
        smod = slab1
    # if depth from first slab (slab1) is not nan and depth of earthquake is closest to depth of the slab1, then use slab1
    if sdep1 > 0:
        if np.isnan(slab2_ddiff):
            smod = slab1
        elif slab1_ddiff < slab2_ddiff:
            smod = slab1
    # if depth from second slab (slab2) is not nan and depth of earthquake is closest to depth of the slab2, then use slab2
    if sdep2 > 0:
        if np.isnan(slab1_ddiff):
            smod = slab2
        elif slab2_ddiff < slab1_ddiff:
            smod = slab2
    # if depth differences are the same, assume slab1 since this is most likely an intraslab event, it does not matter which slab to use
    if slab1_ddiff == slab2_ddiff:
        smod = slab1

    return smod
