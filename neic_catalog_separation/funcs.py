# stdlib imports
from copy import deepcopy
from math import asin, cos, sqrt

import numpy as np

from .eqtype import EqType


def find_closest_loc(lon, lat, sup_lon, sup_lat, sup_depth, sup_strike, sup_dip):
    """
    Determine nearest lon,lat from titled/vertical slab data to earthquake location and take titled slab depth, strike, and dip at that point
    Args:
        lon (float): Earthquake (EQ) lon
        lat (float): EQ lat
        sup_lon (list): lon floats from titled data
        sup_lat (list): lat floats from titled data
        sup_depth (list): depth floats from titled data
        sup_strike (list): strike floats from titled data
        sup_dip (list): dip floats from titled data
    Returns:
        near_lon (float): closest lon
        near_lat (float): closest lat
        near_depth (float): closest depth
        near_strike (float): closest strike
        near_dip (float):closest dip
    """
    d2r = np.pi / 180.0  # degrees to radians
    earth_diam = 12742  # km
    dist = []
    dist2 = np.empty((len(sup_lon), 6))
    # use the Haversine formula to determine the great circle distance between 2 points on a sphere given lon,lat of each point:
    for i in range(len(sup_lon)):
        dist = (
            0.5
            - cos((sup_lat[i] - lat) * d2r) / 2
            + cos(lat * d2r)
            * cos(sup_lat[i] * d2r)
            * (1 - cos((sup_lon[i] - lon) * d2r))
            / 2
        )
        dist = earth_diam * asin(sqrt(dist))
        dist2[i] = np.hstack(
            (sup_lon[i], sup_lat[i], dist, sup_depth[i], sup_strike[i], sup_dip[i])
        )
    # find index corresponding to minimum distance & get data at that index
    min_dist = np.where(dist2 == np.min(dist2, axis=0)[2])
    near_lon = dist2[min_dist[0], 0]
    near_lat = dist2[min_dist[0], 1]
    near_depth = dist2[min_dist[0], 3]
    near_strike = dist2[min_dist[0], 4]
    near_dip = dist2[min_dist[0], 5]

    # For slabs that are "curled", just assume the first entry
    near_lon = near_lon[0]
    near_lat = near_lat[0]
    near_depth = near_depth[0]
    near_strike = near_strike[0]
    near_dip = near_dip[0]
    # Get the nearest distance from dist2 at this near_lon/Lat location
    dist = (
        0.5
        - cos((near_lat - lat) * d2r) / 2
        + cos(lat * d2r) * cos(near_lat * d2r) * (1 - cos((near_lon - lon) * d2r)) / 2
    )
    dist = earth_diam * asin(sqrt(dist))
    min_dist_vert = np.min(dist2, axis=0)[2]

    return near_depth, near_strike, near_dip, min_dist_vert


def get_probability(x, x1, p1, x2, p2):
    """Calculate probability using a ramped function.

    Uses code from ShakeMap (Worden et al., 2005) and STREC (https://code.usgs.gov/ghsc/esi/strec)

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
    if x <= x1:
        prob = p1
    elif x >= x2:
        prob = p2
    else:
        slope = (p1 - p2) / (x1 - x2)
        intercept = p1 - slope * x1
        prob = x * slope + intercept
    return prob


def get_kagan_angle(strike1, dip1, rake1, strike2, dip2, rake2):
    """Calculate the Kagan angle between two moment tensors defined by strike,dip and rake.

    Uses code from Slab2 (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)

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


def plane_to_tensor(strike, dip, rake, mag=6.0):
    """Convert strike,dip,rake values to moment tensor parameters.

    Uses code from Slab2 (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)

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


def calc_theta(vm1, vm2):
    """Calculate angle between two moment tensor matrices.

    Uses code from Slab2 (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)

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


def calc_eigenvec(moment_tensor):
    """Calculate eigenvector of moment tensor matrix.

    Uses code from Slab2 (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)

    Args:
        ndarray: moment tensor matrix (see plane_to_tensor)
    Returns:
        ndarray: eigenvector representation of input moment tensor.
    """
    V, S = np.linalg.eigh(moment_tensor)
    inds = np.argsort(V)
    S = S[:, inds]
    S[:, 2] = np.cross(S[:, 0], S[:, 1])
    return S


def ang_from_R1R2(R1, R2):
    """Calculate angle between two eigenvectors.

    Uses code from Slab2 (Hayes et al., 2018; https://code.usgs.gov/ghsc/esi/slab2)

    Args:
        R1 (ndarray): eigenvector of first moment tensor
        R2 (ndarray): eigenvector of second moment tensor
    Returns:
        float: angle between eigenvectors
    """
    return np.arccos((np.trace(np.dot(R1, R2.transpose())) - 1.0) / 2.0)


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
        eq_category (str): most likely earthquake classification with 'eq' (meaning 'equal') appended to the end of the classification
    """
    if p_crustal == p_int:
        # If MT (moment tensor/nodal plane info) exists, use thresholds to determine if interface
        # If no MT exists, then use depth - if closer to moho than slab, then crustal, else interface
        diffMoho = np.abs(depth - mohoDepth)
        if s1 > 0 and sstr > 0:
            if kagan <= 30:
                eq_category = EqType.SUBDUCTION_INTERFACE
            else:
                eq_category = EqType.CRUSTAL
        elif diffMoho < ddiff:
            eq_category = EqType.CRUSTAL
        else:
            eq_category = EqType.SUBDUCTION_INTERFACE

    elif p_crustal == p_slab:
        # if closer to moho than slab = crustal, else slab
        diffMoho = np.abs(depth - mohoDepth)
        if diffMoho < ddiff:
            eq_category = EqType.CRUSTAL
        else:
            eq_category = EqType.SUBDUCTION_INTRASLAB

    elif p_slab == p_int:
        # If MT exists, use thresholds to determine if interface
        # If no MT exists, then if less than SZ depth & depth to slab surf is within 20 km = interface
        if s1 > 0 and sstr > 0:
            if kagan <= 30:
                eq_category = EqType.SUBDUCTION_INTERFACE
            else:
                eq_category = EqType.SUBDUCTION_INTRASLAB
        elif depth <= dlim and ddiff <= 20:
            eq_category = EqType.SUBDUCTION_INTERFACE
        else:
            eq_category = EqType.SUBDUCTION_INTRASLAB

    elif p_slab == p_int == p_crustal and p_slab > 0:
        # If MT exists, use thresholds to determine if interface
        # If no MT, then use depths
        diffMoho = np.abs(depth - mohoDepth)
        if s1 > 0 and sstr > 0:
            if kagan <= 30:
                eq_category = EqType.SUBDUCTION_INTERFACE
        else:
            if depth > sdep and depth > dlim:
                eq_category = EqType.SUBDUCTION_INTRASLAB
            elif depth <= dlim and ddiff <= 20:
                eq_category = EqType.SUBDUCTION_INTERFACE
            elif diffMoho < ddiff:
                eq_category = EqType.CRUSTAL
            else:
                eq_category = EqType.UNKNOWN
    return eq_category


def flag_mantle(s1, sstr, depth, dlim, sdep, sunc, Ppl, Tpl, eq_category):
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
        eq_category (str): most likely earthquake classification
    Returns:
        eq_category (str): most likely earthquake classification with 'm' (meaning 'mantle') appended to the end of the classification
    """
    if s1 > 0 and sstr > 0:
        if depth > dlim and depth <= (sdep - sunc) and Ppl > 45:
            eq_category = EqType.flag_mantle(eq_category)
        elif depth > dlim and depth <= (sdep - sunc) and Tpl > 20:
            eq_category = EqType.flag_mantle(eq_category)
    elif depth > dlim and depth <= (sdep - sunc):
        eq_category = EqType.flag_mantle(eq_category)
    else:
        eq_category = eq_category
    return eq_category


def determine_quality(eq_category, p_crustal, p_slab, p_interface, s1, sstr):
    """
    Determine quality (qual) of earthquake classification
    if kagan angle/MT is available: probability >=80% qual = A; 60-80% qual = B; 40-60% qual = C; <40% qual = D; if eq qual = D
    if kagan angle/MT not available: probability >=80% qual = B; 60-80% qual = C; <60% = D
    Args:
        eq_category (str): most likely earthquake classification
        p_crustal (float): probability the earthquake is crustal
        p_slab (float): probability the earthquake is intraslab
        p_int (float): probability the earthquake occurred along the subduction interface
        s1 (float): earthquakes strike at nodal plane 1
        sstr (float): slabs strike at the earthquakes location
    Returns:
        qual (str): quality/grade of earthquake classification (A, B, C, D)
    """
    if eq_category in [
        EqType.SUBDUCTION_INTERFACE_MANTLE,
        EqType.SUBDUCTION_INTRASLAB_MANTLE,
        EqType.SUBDUCTION_OUTERRISE_MANTLE,
    ]:
        qual = "D"
        return qual
    if eq_category in [EqType.SUBDUCTION_OUTERRISE, EqType.SUBDUCTION_OUTERRISE_MANTLE]:
        qual = "D"
        return qual

    prob = 0.0
    if eq_category == EqType.CRUSTAL:
        prob = p_crustal
    elif eq_category == EqType.SUBDUCTION_INTRASLAB:
        prob = p_slab
    elif eq_category == EqType.SUBDUCTION_INTERFACE:
        prob = p_interface
    if s1 > 0 and sstr > 0:
        if prob >= 80:
            qual = "A"
        elif prob >= 60 and prob < 80:
            qual = "B"
        elif prob >= 40 and prob < 60:
            qual = "C"
        else:
            qual = "D"
    else:
        if prob >= 80:
            qual = "B"
        elif prob >= 60 and prob < 80:
            qual = "C"
        else:
            qual = "D"
    return qual
