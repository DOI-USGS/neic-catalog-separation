# stdlib imports
import logging

# third party imports
import numpy as np

# local imports
from . import data_sources, funcs
from .eqtype import NO_VALUE, EqType


def classify(
    earthquake,
    slab1,
    slab2,
    moho_depth,
    seismogenic_depth,
    nshm,
    config,
    tolerance,
) -> dict:
    """Determine probability that an event occurred in the upper plate (crustal), lower plate (intraslab), or along the subduction interface
    Args:
        slab1 ): slab2 model region
        slab2 (str or boolean): second slab2 model region (if applicable). Default is False.
        data_dir (str): users current working directory
        dataframe (pandas dataframe): input catalog dataframe
        config (dict): name of config file.
        tolerance (float): maximum distance (km) from interface to be considered an interface earthquake.
    Output:

    """
    logger = logging.getLogger(__name__)

    eq_id = earthquake["id"]
    eq_lon = float(earthquake["longitude"])
    eq_lat = float(earthquake["latitude"])
    eq_depth = float(earthquake["depth"])
    eq_strike = float(earthquake["S1"])
    eq_dip = float(earthquake["D1"])
    eq_rake = float(earthquake["R1"])
    eq_paxis_plunge = float(earthquake["Ppl"])
    eq_taxis_plunge = float(earthquake["Tpl"])

    if eq_lon < 0.0:
        eq_lon += 360.0

    # if a second slab2 region is provided, determine which slab this
    # event is closest to and use that slab2 region for classification:
    if slab2:
        slab_closest = data_sources.Slab.find_closest(
            lat=eq_lat, lon=eq_lon, depth=eq_depth, slab1=slab1, slab2=slab2
        )
        logger.info("Using the %s Slab2 region for %s.....", slab_closest.region, eq_id)
    else:
        slab_closest = slab1

    (
        slab_depth_eq,
        slab_strike_eq,
        slab_dip_eq,
        slab_uncertainty_eq,
    ) = slab_closest.get_values(lat=eq_lat, lon=eq_lon)
    moho_depth_eq = moho_depth.get_value(lat=eq_lat, lon=eq_lon)

    # determine deep_limit (deep seismogenic limit + tolerance (buffer/wiggle room))
    sz_deep, sz_rake = seismogenic_depth.get_values(
        slab_region=slab_closest.region, nshm=nshm
    )
    deep_limit = sz_deep + tolerance

    # set differences now (will be overwritten if slab values are not nan)
    ddiff = NO_VALUE
    kagan = NO_VALUE
    eq_category = EqType.UNKNOWN

    # If an EQ occurs in a slab region where there is a vertical/titled/overturned slab, then the published slab2 sup csv file must be used
    # find nearest lon,lat to EQ lon,lat and extract the corresponding tilted slab depth, strike, and dip
    if np.isnan(slab_depth_eq):
        if slab_closest.region in ["man", "ker", "izu", "sol"]:
            slab_depth_eq, slab_strike_eq, slab_dip_eq = slab_closest.get_overturned(
                lat=eq_lat, lon=eq_lon
            )
    # if depth is still nan, then assume this is crustal
    if not nshm and np.isnan(slab_depth_eq):
        p_interface = 0.0
        p_crustal = 1.0
        p_slab = 0.0
        # data below is written to output file, so setting to a nonsensical number here that may be overwritten
        moho_depth_eq = NO_VALUE
        slab_depth_eq = NO_VALUE
        slab_strike_eq = NO_VALUE
        slab_dip_eq = NO_VALUE
        slab_uncertainty_eq = NO_VALUE
    # For USGS NSHM catalogs, handle events outside of the SZ
    # here, slab depth does not exist & we are not in an overturned slab region
    # These events occur outside the 0 km slab contour, and can be assumed as outer-rise (eq_category="or")
    elif nshm and np.isnan(slab_depth_eq):
        p_interface = 0.0
        p_crustal = 1.0
        p_slab = 0.0
        # data below is written to output file, so setting to a nonsensical number here that may be overwritten
        moho_depth_eq = NO_VALUE
        slab_depth_eq = NO_VALUE
        slab_strike_eq = NO_VALUE
        slab_dip_eq = NO_VALUE
        slab_uncertainty_eq = NO_VALUE
        # band-aid for USGS PRVI NSHM - mark events outside/seaward of the 0 km slab grids as outer-rise, but events in the
        # specified bounding box below are outside the slab grids but crustal (i.e., not outer-rise), so mark those accordingly
        if slab_closest.region in ["mue", "car"]:
            if (
                eq_lon >= 285.0
                and eq_lon <= 289.0
                and eq_lat >= 18.0
                and eq_lat <= 20.0
            ):
                eq_category = EqType.CRUSTAL
            else:
                eq_category = EqType.SUBDUCTION_OUTERRISE
        else:
            eq_category = EqType.SUBDUCTION_OUTERRISE
    else:
        # get difference between slab & EQ
        ddiff = np.abs(eq_depth - slab_depth_eq)
        ddiff2 = eq_depth - slab_depth_eq

        # Determine kagan angle between fault plane and slab:
        if eq_strike > 0.0 and slab_strike_eq > 0.0:
            kagan = funcs.get_kagan_angle(
                slab_strike_eq, slab_dip_eq, sz_rake, eq_strike, eq_dip, eq_rake
            )
        else:
            kagan = np.nan

        # Determine interface probability:
        # Calculate the probability that the EQ occurred along the interface given the
        # EQ location and depth to slab +/- a tolerance (buffer) value around the slab
        x1 = slab_depth_eq + tolerance
        x2 = slab_depth_eq - tolerance
        # if EQ is between the tolerance zone depth and less than maximum seismogenic zone (sz) depth (deep_limit)
        if x2 <= eq_depth <= x1 and eq_depth <= deep_limit:
            p_interface_tolerance = 1.0
        # else if less than top of tolerance, but deeper than moho; this could be mantle wedge region then ramp from x1 = moho and x2 = x2, with p1 = 1 and p2 = 0
        elif eq_depth < x2 and eq_depth >= moho_depth_eq:
            x1 = moho_depth_eq
            x = eq_depth
            p1 = config["p_interface_fx"]["p1"]
            p2 = config["p_interface_fx"]["p2"]
            p_interface_tolerance = funcs.get_probability(x, x1, p1, x2, p2)
        # else if within tolerance, but less than moho, then interface
        elif x2 <= eq_depth <= x1 and eq_depth <= moho_depth_eq:
            p_interface_tolerance = 1.0
        else:
            p_interface_tolerance = 0.0

        # Calculate probability that event occurred between the seismogenic zone and maximum seismogenic zone (deep_limit)
        x1 = deep_limit - tolerance
        x2 = deep_limit
        p1 = config["p_interface_sz"]["p1"]
        p2 = config["p_interface_sz"]["p2"]
        p_interface_sz = funcs.get_probability(eq_depth, x1, p1, x2, p2)

        # Calculate probability of interface given Kagan's angle
        if eq_strike > 0 and slab_strike_eq > 0:
            x1 = config["p_interface_kagan"]["x1"]
            x2 = config["p_interface_kagan"]["x2"]
            p1 = config["p_interface_kagan"]["p1"]
            p2 = config["p_interface_kagan"]["p2"]
            p_interface_kagan = funcs.get_probability(kagan, x1, p1, x2, p2)
        # Calculate combined probability of interface
        if eq_strike > 0 and slab_strike_eq > 0:
            p_interface = p_interface_tolerance * p_interface_sz * p_interface_kagan
        else:
            p_interface = p_interface_tolerance * p_interface_sz

        # Calculate probability that the earthquake lies above the slab
        # and is thus crustal
        x1 = config["p_crust_diff"]["x1"]
        x2 = config["p_crust_diff"]["x2"]
        p1 = config["p_crust_diff"]["p1"]
        p2 = config["p_crust_diff"]["p2"]
        p_crust_diff = funcs.get_probability(ddiff2, x1, p1, x2, p2)

        # Calculate probability that earthquake is crustal
        # if greater than-equal to slab depth (x2), then p = 0
        x1 = moho_depth_eq
        x2 = slab_depth_eq
        p1 = config["p_crust_slab"]["p1"]  # 1
        p2 = config["p_crust_slab"]["p2"]  # 0
        p_crust_slab = funcs.get_probability(eq_depth, x1, p1, x2, p2)

        # Calculate probability of the event being crustal
        p_crustal = (1 - p_interface) * p_crust_slab * p_crust_diff

        # Calculate probability of the event being intraslab
        p_slab = 1 - (p_interface + p_crustal)

    # turn probabilities into percentages
    p_crustal *= 100
    p_interface *= 100
    p_slab *= 100

    # Based on probability determine if event is upper plate/crustal (u), lower plate/slab (l), or interface (i)
    if eq_category != EqType.SUBDUCTION_OUTERRISE:
        if p_crustal > p_interface and p_crustal > p_slab:
            eq_category = EqType.CRUSTAL
        elif p_interface > p_crustal and p_interface > p_slab:
            eq_category = EqType.SUBDUCTION_INTERFACE
        elif p_slab > p_interface and p_slab > p_crustal:
            eq_category = EqType.SUBDUCTION_INTRASLAB
        elif p_slab == p_interface == p_crustal == 0:
            eq_category = EqType.UNKNOWN
        # If there is an equal probability, determine most likely classification. Locations will also include 'eq' for 'equal' at the end to inform the user that the event may need further inspection
        elif (
            p_slab == p_interface == p_crustal
            or p_slab == p_interface
            or p_interface == p_crustal
            or p_crustal == p_slab
        ):
            eq_category = funcs.equal_prob(
                p_crustal,
                p_interface,
                p_slab,
                moho_depth_eq,
                eq_strike,
                slab_strike_eq,
                slab_depth_eq,
                kagan,
                ddiff,
                deep_limit,
                eq_depth,
            )

    # Flag events that may be mantle events (not determining probability, just flagging possible mantle events) - not used for USGS NSHM catalogs
    # This is tagged by adding a 'm' for mantle to eq_category, so that the highest probability is also still known
    # These events usually have extensional focal mechanisms. From a Frolich diagram, this would be Ppl > 45 or Tpl > 20 (also based on looking at test events)
    if not nshm and eq_category != EqType.SUBDUCTION_INTERFACE:
        eq_category = funcs.flag_mantle(
            eq_strike,
            slab_strike_eq,
            eq_depth,
            deep_limit,
            slab_depth_eq,
            slab_uncertainty_eq,
            eq_paxis_plunge,
            eq_taxis_plunge,
            eq_category,
        )
    # Determine quality (qual) of classification
    quality = funcs.determine_quality(
        eq_category, p_crustal, p_slab, p_interface, eq_strike, slab_strike_eq
    )

    classification = earthquake.copy()
    classification.update(
        {
            "p_crustal": p_crustal,
            "p_interface": p_interface,
            "p_slab": p_slab,
            "eq_category": EqType.to_string(eq_category),
            "quality": quality,
            "slab2_region": slab_closest.region,
        }
    )
    return classification
