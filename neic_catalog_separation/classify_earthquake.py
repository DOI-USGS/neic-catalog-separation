# stdlib imports
import logging
import typing

# third party imports
import numpy as np

# local imports
from . import data_sources, funcs
from .eqtype import NO_VALUE, EqType


def classify(
    earthquake: dict,
    slab1: str,
    slab2: typing.Union[str, None],
    moho_depth: float,
    seismogenic_depth: float,
    config: dict,
    nshm: bool,
) -> dict:
    """Determine probability that an event occurred in the upper plate (crustal), lower plate (intraslab), or along the subduction interface
    Args:
        earthquake (dict): dictionary of earthquake parameters (id, longitude, latitude, depth, S1, D1, R1, Ppl, Tpl)
        slab1: Slab2 model region
        slab2: second Slab2 model region (if applicable)
        moho_depth: Depth of Moho (km)
        seismogenic_depth: Seismogenic depth of interface (km).
        config (dict): name of config file.
        nshm (boolean): whether or not the input catalog is a USGS NSHM catalog (if True, then events outside of the slab2 regions will be classified as outer-rise instead of not being calssfied)
    Returns:
        Dictionary with classification values.
    """
    logger = logging.getLogger(__name__)

    eq_id = earthquake["id"]
    eq_lon = float(earthquake["longitude"])
    eq_lat = float(earthquake["latitude"])
    eq_depth = float(earthquake["depth"])
    eq_strike = float(earthquake["S1"])
    eq_dip = float(earthquake["D1"])
    eq_rake = float(earthquake["R1"])

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
        interface_depth_eq,
        slab_strike_eq,
        slab_dip_eq,
        slab_thickness_eq,
        slab_uncertainty_eq,
    ) = slab_closest.get_values(lat=eq_lat, lon=eq_lon)
    moho_depth_eq = moho_depth.get_value(lat=eq_lat, lon=eq_lon)

    # determine deep_limit for interface earthquakes (deep seismogenic limit + tolerance (buffer/wiggle room))
    sz_deep, sz_rake = seismogenic_depth.get_values(
        slab_region=slab_closest.region, nshm=nshm
    )

    # set differences now (will be overwritten if slab values are not nan)
    kagan = NO_VALUE

    # Set uncertainty to zero if turned off by config
    if not config.get("include_uncertainty", True):
        slab_uncertainty_eq = 0.0

    # If an EQ occurs in a slab region where there is a vertical/titled/overturned slab, then the published slab2 sup csv file must be used
    # find nearest lon,lat to EQ lon,lat and extract the corresponding tilted slab depth, strike, and dip
    if np.isnan(interface_depth_eq):
        if slab_closest.region in ["man", "ker", "izu", "sol"]:
            (
                interface_depth_eq,
                slab_strike_eq,
                slab_dip_eq,
                slab_thickness_eq,
            ) = slab_closest.get_overturned(lat=eq_lat, lon=eq_lon)
            slab_uncertainty_eq = 0.0

    # initialize probabilities and category
    p_interface = 0.0
    p_intraslab = 0.0
    p_crustal = 0.0
    p_outerrise = 0.0
    p_mantle = 0.0
    eq_category = EqType.UNKNOWN

    # if depth is still nan
    if np.isnan(interface_depth_eq):
        # data below is written to output file, so setting to a nonsensical number here that may be overwritten
        interface_depth_eq = NO_VALUE
        slab_strike_eq = NO_VALUE
        slab_dip_eq = NO_VALUE
        slab_uncertainty_eq = NO_VALUE

        if not nshm:
            # crust or mantle
            if np.isnan(moho_depth_eq):
                p_crustal = 1.0
                p_mantle = 0.0
                moho_depth_eq = NO_VALUE
            else:
                p_crustal = funcs.get_probability(
                    x=eq_depth - moho_depth_eq,
                    x1=config["p_crustal"]["x1"],
                    x2=config["p_crustal"]["x2"],
                    p1=1.0,
                    p2=0.0,
                )
                p_mantle = 1.0 - p_crustal

        else:
            (
                slab_depth,
                _,
                _,
                slab_thickness,
                _,
                min_slab_dist,
            ) = slab_closest.get_values_closest(lat=eq_lat, lon=eq_lon)
            if (
                min_slab_dist < config["outerrise_maxdist"]
                and slab_depth < config["outerrise_slab_maxdepth"]
            ):  # outer rise region
                p_outerrise = funcs.get_probability(
                    x=eq_depth - slab_thickness,
                    x1=config["p_outerrise"]["x1"],
                    x2=config["p_outerrise"]["x2"],
                    p1=1.0,
                    p2=0.0,
                )
                p_mantle = 1.0 - p_outerrise
            else:  # not outer rise region, so assume crustal
                if np.isnan(moho_depth_eq):
                    p_crustal = 1.0
                    p_mantle = 0.0
                    moho_depth_eq = NO_VALUE
                else:
                    p_crustal = funcs.get_probability(
                        x=eq_depth - moho_depth_eq,
                        x1=config["p_crustal"]["x1"],
                        x2=config["p_crustal"]["x2"],
                        p1=1.0,
                        p2=0.0,
                    )
                    p_mantle = 1.0 - p_crustal

    else:
        # Determine Kagan angle between fault plane and slab:
        if eq_strike > 0.0 and slab_strike_eq > 0.0:
            kagan = funcs.get_kagan_angle(
                slab_strike_eq, slab_dip_eq, sz_rake, eq_strike, eq_dip, eq_rake
            )
        else:
            kagan = np.nan

        # Determine interface probability:
        # Calculate the probability that the EQ occurred along the interface given the
        # EQ location and depth to slab +/- a tolerance (buffer) value around the slab
        p_interface_shallow = funcs.get_probability(
            x=eq_depth - interface_depth_eq,
            x1=-slab_uncertainty_eq + config["p_interface_shallow"]["x1"],
            x2=config["p_interface_shallow"]["x2"],
            p1=0.0,
            p2=1.0,
        )
        p_interface_deep = funcs.get_probability(
            x=eq_depth - interface_depth_eq,
            x1=config["p_interface_deep"]["x1"],
            x2=+slab_uncertainty_eq + config["p_interface_deep"]["x2"],
            p1=1.0,
            p2=0.0,
        )
        # Calculate probability that event occurred between the seismogenic zone and maximum seismogenic zone (deep_limit)
        p_interface_sz = funcs.get_probability(
            x=eq_depth - sz_deep,
            x1=config["p_interface_seismogenic"]["x1"],
            x2=+slab_uncertainty_eq + config["p_interface_seismogenic"]["x2"],
            p1=1.0,
            p2=0.0,
        )

        # Calculate probability of interface given Kagan's angle
        p_interface_kagan = 1.0
        if not np.isnan(eq_strike) and not np.isnan(slab_strike_eq):
            p_interface_kagan = funcs.get_probability(
                x=kagan,
                x1=config["p_interface_kagan"]["x1"],
                x2=config["p_interface_kagan"]["x2"],
                p1=config["p_interface_kagan"]["p1"],
                p2=config["p_interface_kagan"]["p2"],
            )
        p_interface = (
            p_interface_shallow * p_interface_deep * p_interface_sz * p_interface_kagan
        )

        # Crust: Depth relative to Moho.
        p_crustal_moho = funcs.get_probability(
            x=eq_depth - moho_depth_eq,
            x1=config["p_crustal"]["x1"],
            x2=config["p_crustal"]["x2"],
            p1=1.0,
            p2=0.0,
        )
        # Crust: Depth relative to interface (top of slab)
        p_crustal_interface = funcs.get_probability(
            x=eq_depth - interface_depth_eq,
            x1=-slab_uncertainty_eq + config["p_interface_shallow"]["x1"],
            x2=config["p_interface_shallow"]["x2"],
            p1=1.0,
            p2=0.0,
        )
        p_crustal = p_crustal_moho * p_crustal_interface

        # Intraslab: Depth relative to interface (top of slab)
        p_intraslab_shallow = funcs.get_probability(
            x=eq_depth - interface_depth_eq,
            x1=-slab_uncertainty_eq + config["p_intraslab_shallow"]["x1"],
            x2=+slab_uncertainty_eq + config["p_intraslab_shallow"]["x2"],
            p1=0.0,
            p2=1.0,
        )
        # Intraslab: Depth relative to bottom of slab (interface + slab thickness)
        p_intraslab_deep = funcs.get_probability(
            x=eq_depth - (interface_depth_eq + slab_thickness_eq),
            x1=-slab_uncertainty_eq + config["p_intraslab_deep"]["x1"],
            x2=+slab_uncertainty_eq + config["p_intraslab_deep"]["x2"],
            p1=1.0,
            p2=0.0,
        )
        p_intraslab_wedge = 1.0 - p_interface_deep

        # Maximum depth of interface probabilities
        interface_maxdepth = sz_deep + config["p_interface_seismogenic"]["x2"]
        p_intraslab = (
            # Above sz_deep + buffer = wedge.
            p_intraslab_wedge * p_intraslab_deep
            if eq_depth < interface_maxdepth
            # Below sz_deep + buffer = within slab.
            else p_intraslab_shallow * p_intraslab_deep
        )

        # Mantle: crust
        p_mantle_crust = 1.0 - p_crustal_moho
        p_mantle_interface_shallow = 1.0 - p_interface_shallow
        p_mantle_intraslab_deep = 1.0 - p_intraslab_deep
        if eq_depth < interface_depth_eq + config["p_interface_shallow"]["x2"]:
            p_mantle = p_mantle_crust * p_mantle_interface_shallow
        else:
            p_mantle = p_mantle_intraslab_deep

        p_sum = p_interface + p_crustal + p_intraslab + p_outerrise + p_mantle
        p_interface /= p_sum
        p_crustal /= p_sum
        p_intraslab /= p_sum
        p_outerrise /= p_sum
        p_mantle /= p_sum

    # turn probabilities into percentages
    p_crustal *= 100
    p_interface *= 100
    p_intraslab *= 100
    p_mantle *= 100
    p_outerrise *= 100

    # Set category
    eq_category = funcs.get_category(
        p_crustal=p_crustal,
        p_interface=p_interface,
        p_intraslab=p_intraslab,
        p_mantle=p_mantle,
        p_outerrise=p_outerrise,
    )

    # Determine quality (qual) of classification
    quality = funcs.determine_quality(
        eq_category=eq_category,
        p_crustal=p_crustal,
        p_intraslab=p_intraslab,
        p_interface=p_interface,
        eq_strike=eq_strike,
        slab_strike=slab_strike_eq,
    )

    # If a rename mapping is specified in the config file, remap this earthquake's category
    if "rename" in config:
        eq_category = EqType.rename_category(config).get(eq_category, eq_category)

    classification = earthquake.copy()
    classification.update(
        {
            "p_crustal": p_crustal,
            "p_interface": p_interface,
            "p_intraslab": p_intraslab,
            "p_mantle": p_mantle,
            "p_outerrise": p_outerrise,
            "eq_category": EqType.to_string(eq_category),
            "quality": quality,
            "slab2_region": slab_closest.region,
        }
    )
    return classification
