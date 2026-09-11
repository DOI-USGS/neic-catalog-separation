# stdlib imports
import heapq
import operator
from copy import deepcopy

import numpy as np

from .eqtype import EqType


def deg_to_rad(degree):
    return degree * np.pi / 180.0


def haversine_distance(points_lat, points_lon, target_lat, target_lon):
    """Compute great circle distance from target to points using Haversize formula.

    Args:
        points_lat (np.ndarray): Latitude (deg) of points.
        points_lon (np.ndarray): Longitude (deg) of points.
        target_lat (np.ndarray): Latitude (deg) of target.
        target_lon (np.ndarray): Longitude (deg) of points.
    Returns:
        array: Great circle distance (km) from target to points.
    """
    earth_radius = 6373.0  # km

    lat_r = deg_to_rad(points_lat)
    lon_r = deg_to_rad(points_lon)
    target_lat_r = deg_to_rad(target_lat)
    target_lon_r = deg_to_rad(target_lon)

    return (
        2
        * earth_radius
        * np.arcsin(
            np.sqrt(
                np.sin(0.5 * (target_lat_r - lat_r)) ** 2
                + (np.cos(lat_r) * np.cos(target_lat_r))
                * np.sin(0.5 * (target_lon_r - lon_r)) ** 2
            )
        )
    )


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
    deg_to_rad = np.pi / 180.0

    # get exponent and moment magnitude
    magpow = mag * 1.5 + 16.1
    mom = np.power(10, magpow)

    # get tensor components
    mrr = mom * np.sin(2 * dip * deg_to_rad) * np.sin(rake * deg_to_rad)
    mtt = -mom * (
        (
            np.sin(dip * deg_to_rad)
            * np.cos(rake * deg_to_rad)
            * np.sin(2 * strike * deg_to_rad)
        )
        + (
            np.sin(2 * dip * deg_to_rad)
            * np.sin(rake * deg_to_rad)
            * (np.sin(strike * deg_to_rad) * np.sin(strike * deg_to_rad))
        )
    )
    mpp = mom * (
        (
            np.sin(dip * deg_to_rad)
            * np.cos(rake * deg_to_rad)
            * np.sin(2 * strike * deg_to_rad)
        )
        - (
            np.sin(2 * dip * deg_to_rad)
            * np.sin(rake * deg_to_rad)
            * (np.cos(strike * deg_to_rad) * np.cos(strike * deg_to_rad))
        )
    )
    mrt = -mom * (
        (
            np.cos(dip * deg_to_rad)
            * np.cos(rake * deg_to_rad)
            * np.cos(strike * deg_to_rad)
        )
        + (
            np.cos(2 * dip * deg_to_rad)
            * np.sin(rake * deg_to_rad)
            * np.sin(strike * deg_to_rad)
        )
    )
    mrp = mom * (
        (
            np.cos(dip * deg_to_rad)
            * np.cos(rake * deg_to_rad)
            * np.sin(strike * deg_to_rad)
        )
        - (
            np.cos(2 * dip * deg_to_rad)
            * np.sin(rake * deg_to_rad)
            * np.cos(strike * deg_to_rad)
        )
    )
    mtp = -mom * (
        (
            np.sin(dip * deg_to_rad)
            * np.cos(rake * deg_to_rad)
            * np.cos(2 * strike * deg_to_rad)
        )
        + (
            0.5
            * np.sin(2 * dip * deg_to_rad)
            * np.sin(rake * deg_to_rad)
            * np.sin(2 * strike * deg_to_rad)
        )
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


def get_category(p_interface, p_intraslab, p_crustal, p_mantle, p_outerrise):
    probabilities = (
        (p_crustal, EqType.CRUSTAL),
        (p_interface, EqType.SUBDUCTION_INTERFACE),
        (p_intraslab, EqType.SUBDUCTION_INTRASLAB),
        (p_outerrise, EqType.SUBDUCTION_OUTERRISE),
        (p_mantle, EqType.MANTLE),
    )
    mixed_categories = {
        (
            EqType.CRUSTAL,
            EqType.SUBDUCTION_INTERFACE,
        ): EqType.CRUSTAL_SUBDUCTION_INTERFACE,
        (EqType.CRUSTAL, EqType.MANTLE): EqType.CRUSTAL_MANTLE,
        (
            EqType.SUBDUCTION_INTERFACE,
            EqType.CRUSTAL,
        ): EqType.SUBDUCTION_INTERFACE_CRUSTAL,
        (
            EqType.SUBDUCTION_INTERFACE,
            EqType.SUBDUCTION_INTRASLAB,
        ): EqType.SUBDUCTION_INTERFACE_INTRASLAB,
        (
            EqType.SUBDUCTION_INTERFACE,
            EqType.MANTLE,
        ): EqType.SUBDUCTION_INTERFACE_MANTLE,
        (
            EqType.SUBDUCTION_INTRASLAB,
            EqType.SUBDUCTION_INTERFACE,
        ): EqType.SUBDUCTION_INTRASLAB_INTERFACE,
        (
            EqType.SUBDUCTION_INTRASLAB,
            EqType.MANTLE,
        ): EqType.SUBDUCTION_INTRASLAB_MANTLE,
        (
            EqType.SUBDUCTION_OUTERRISE,
            EqType.MANTLE,
        ): EqType.SUBDUCTION_OUTERRISE_MANTLE,
        (EqType.MANTLE, EqType.CRUSTAL): EqType.MANTLE_CRUSTAL,
        (
            EqType.MANTLE,
            EqType.SUBDUCTION_INTERFACE,
        ): EqType.MANTLE_SUBDUCTION_INTERFACE,
        (
            EqType.MANTLE,
            EqType.SUBDUCTION_INTRASLAB,
        ): EqType.MANTLE_SUBDUCTION_INTRASLAB,
        (
            EqType.MANTLE,
            EqType.SUBDUCTION_OUTERRISE,
        ): EqType.MANTLE_SUBDUCTION_OUTERRISE,
    }
    top_categories = heapq.nlargest(2, probabilities, key=operator.itemgetter(0))
    if p_interface == p_intraslab == p_crustal == p_mantle == p_outerrise:
        category = EqType.UNKNOWN
    elif top_categories[1][0] == 0.0:
        category = top_categories[0][1]
    else:
        labels = tuple([c for _, c in top_categories])
        if labels in mixed_categories:
            category = mixed_categories[labels]
        else:
            category = labels[0]
    return category


def determine_quality(
    eq_category, p_crustal, p_intraslab, p_interface, eq_strike, slab_strike
):
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
        prob = p_intraslab
    elif eq_category == EqType.SUBDUCTION_INTERFACE:
        prob = p_interface
    if eq_strike > 0 and slab_strike > 0:
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
