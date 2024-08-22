# stdlib imports
import configparser

# local imports
import classify_catalog_Funcs as funcs

# third party imports
import numpy as np


def classify_eqs(
    slab1,
    slab2,
    nshm,
    working_dir,
    dataframe,
    outfile,
    flex,
    i,
):
    """
    Determine probability that an event occurred in the upper plate (crustal), lower plate (intraslab), or along the subduction interface
    Args:
        slab1 (str): slab2 model region
        slab2 (str or boolean): second slab2 model region (if applicable). Default is False.
        working_dir (str): users current working directory
        dataframe (pandas dataframe): input catalog dataframe
        outfile (str): name of output file
        flex (int): flex value
        i (int): iteration
    Output:
        *_eqtype.csv: CSV file with catalog and probability information
    """
    lon = dataframe.lon[i]
    lat = dataframe.lat[i]

    if lon < 0:
        lon = lon + 360

    # get info from config file
    config = configparser.ConfigParser()
    config.sections()
    config.read(f"{working_dir}/catalog_sep/Input/config/subduction.conf")

    # if a second slab2 region is provided, determine which slab this
    # event is closest to and use that slab2 region for classification:
    if slab2 is not False:
        smod = funcs.determine_closest_slab(
            slab1, slab2, working_dir, lat, lon, dataframe.depth[i]
        )
        print(f"Using the {smod} slab2 region for {i}.....")
    else:
        smod = slab1
    # determine dlim (deep seismogenic limit + flex (buffer/wiggle room))
    sz_deep, srake = funcs.get_seismogenic_depth(working_dir, smod, nshm)
    dlim = sz_deep + flex

    # get slab2 info & moho depth at earthquake location
    sdep, sstr, sdip, sunc, mohoDepth = funcs.get_slab_moho_info(
        smod, working_dir, lat, lon
    )

    # set differences now (will be overwritten is slab values are not nan)
    ddiff = 999
    kagan = 999

    # If an EQ occurs in a slab region where there is a vertical/titled/overturned slab, then the published slab2 sup csv file must be used
    # find nearest lon,lat to EQ lon,lat and extract the corresponding tilted slab depth, strike, and dip
    if np.isnan(sdep):
        if smod == "man" or smod == "ker" or smod == "izu" or smod == "sol":
            sdep, sstr, sdip = funcs.overturned_slab(smod, working_dir, lon, lat)
    # if depth is still nan, then assume this is crustal
    if np.isnan(sdep):
        p_int = 0
        p_crustal = 1
        p_slab = 0
        # data below is written to output file, so setting to a nonsensical number here that may be overwritten
        mohoDepth = 999
        sdep = 999
        sstr = 999
        sdip = 999
        sunc = 999
        s1 = dataframe.S1[i]
    # For USGS NSHM catalogs, handle events outside of the SZ
    # here, slab depth does not exist & we are not in an overturned slab region
    # These events occur outside the 0 km slab contour, and can be assumed as outer-rise (eqloc="or")
    if nshm and np.isnan(sdep):
        p_int = 0
        p_crustal = 1
        p_slab = 0
        # data below is written to output file, so setting to a nonsensical number here that may be overwritten
        mohoDepth = 999
        sdep = 999
        sstr = 999
        sdip = 999
        sunc = 999
        s1 = dataframe.S1[i]
        # band-aid for USGS PRVI NSHM - mark events outside/seaward of the 0 km slab grids as outer-rise, but events in the
        # specified bounding box below are outside the slab grids but crustal (i.e., not outer-rise), so mark those accordingly
        if smod == "mue" or smod == "car":
            if lon >= 285 and lon <= 289 and lat >= 18 and lat <= 20:
                eqloc = "u"
            else:
                eqloc = "or"
        else:
            eqloc = "u"
    else:
        # set eqloc to empty since it is not determined yet, but needed for an if statement below
        eqloc = ""
        # get difference between slab & EQ
        ddiff = np.abs(dataframe.depth[i] - sdep)
        ddiff2 = dataframe.depth[i] - sdep
        # set strike, dip, and rake of EQ
        s1 = dataframe.S1[i]
        d1 = dataframe.D1[i]
        r1 = dataframe.R1[i]
        fpStr = dataframe.S1[i]
        fpDip = dataframe.D1[i]
        fpRake = dataframe.R1[i]

        # get fault plane info used for kagan angle calculation
        fpStr == s1
        fpDip == d1
        fpRake == r1

        # Determine kagan angle between fault plane and slab:
        if s1 > 0 and sstr > 0:
            kagan = funcs.get_kagan_angle(sstr, sdip, srake, fpStr, fpDip, fpRake)
        else:
            kagan = np.nan

        # Determine interface probability:
        # Calculate the probability that the EQ occurred along the interface given the
        # EQ location and depth to slab +/- a flex (buffer) value around the slab
        x1 = sdep + flex
        x2 = sdep - flex
        # if EQ is between the flex zone depth and less than maximum seismogenic zone (sz) depth (dlim)
        if x2 <= dataframe.depth[i] <= x1 and dataframe.depth[i] <= dlim:
            p_int_flex = 1.0
        # else if less than top of flex, but deeper than moho; this could be mantle wedge region then ramp from x1 = moho and x2 = x2, with p1 = 1 and p2 = 0
        elif dataframe.depth[i] < x2 and dataframe.depth[i] >= mohoDepth:
            x1 = mohoDepth
            x = dataframe.depth[i]
            p1 = float(config["p_int_fx"]["p1"])
            p2 = float(config["p_int_fx"]["p2"])
            p_int_flex = funcs.get_probability(x, x1, p1, x2, p2)
        # else if within flex, but less than moho, then interface
        elif x2 <= dataframe.depth[i] <= x1 and dataframe.depth[i] <= mohoDepth:
            p_int_flex = 1.0
        else:
            p_int_flex = 0.0

        # Calculate probability that event occurred between the seismogenic zone and maximum seismogenic zone (dlim)
        x1 = dlim - flex
        x2 = dlim
        p1 = float(config["p_int_sz"]["p1"])
        p2 = float(config["p_int_sz"]["p2"])
        p_int_sz = funcs.get_probability(dataframe.depth[i], x1, p1, x2, p2)

        # Calculate probability of interface given Kagan's angle
        if s1 > 0 and sstr > 0:
            x1 = float(config["p_int_kagan"]["x1"])
            x2 = float(config["p_int_kagan"]["x2"])
            p1 = float(config["p_int_kagan"]["p1"])
            p2 = float(config["p_int_kagan"]["p2"])
            p_int_kagan = funcs.get_probability(kagan, x1, p1, x2, p2)
        # Calculate combined probability of interface
        if s1 > 0 and sstr > 0:
            p_int = p_int_flex * p_int_sz * p_int_kagan
        else:
            p_int = p_int_flex * p_int_sz

        # Calculate probability that the earthquake lies above the slab
        # and is thus crustal
        x1 = float(config["p_crust_diff"]["x1"])
        x2 = float(config["p_crust_diff"]["x2"])
        p1 = float(config["p_crust_diff"]["p1"])
        p2 = float(config["p_crust_diff"]["p2"])
        p_crust_diff = funcs.get_probability(ddiff2, x1, p1, x2, p2)

        # Calculate probability that earthquake is crustal
        # if greater than-equal to slab depth (x2), then p = 0
        x1 = mohoDepth
        x2 = sdep
        p1 = float(config["p_crust_slab"]["p1"])  # 1
        p2 = float(config["p_crust_slab"]["p2"])  # 0
        p_crust_slab = funcs.get_probability(dataframe.depth[i], x1, p1, x2, p2)

        # Calculate probability of the event being crustal
        p_crustal = (1 - p_int) * p_crust_slab * p_crust_diff

        # Calculate probability of the event being intraslab
        p_slab = 1 - (p_int + p_crustal)

    # turn probabilities into percentages
    p_crustal = p_crustal * 100
    p_int = p_int * 100
    p_slab = p_slab * 100
    print(
        f"EQ {dataframe.id_no[i]}, {i}: Crustal = {p_crustal}, Interface = {p_int}, Intraslab = {p_slab}"
    )

    # Based on probability determine if event is upper plate/crustal (u), lower plate/slab (l), or interface (i)
    if eqloc != "or":
        if p_crustal > p_int and p_crustal > p_slab:
            eqloc = "u"
        elif p_int > p_crustal and p_int > p_slab:
            eqloc = "i"
        elif p_slab > p_int and p_slab > p_crustal:
            eqloc = "l"
        elif p_slab == p_int == p_crustal == 0:
            eqloc = "unknown"
        # If there is an equal probability, determine most likely classification. Locations will also include 'eq' for 'equal' at the end to inform the user that the event may need further inspection
        elif (
            p_slab == p_int == p_crustal
            or p_slab == p_int
            or p_int == p_crustal
            or p_crustal == p_slab
        ):
            eqloc = funcs.equal_prob(
                p_crustal,
                p_int,
                p_slab,
                mohoDepth,
                s1,
                sstr,
                sdep,
                kagan,
                ddiff,
                dlim,
                dataframe.depth[i],
            )

    # Flag events that may be mantle events (not determining probability, just flagging possible mantle events) - not used for USGS NSHM catalogs
    # This is tagged by adding a 'm' for mantle to eqloc, so that the highest probability is also still known
    # These events usually have extensional focal mechanisms. From a Frolich diagram, this would be Ppl > 45 or Tpl > 20 (also based on looking at test events)
    if not nshm and eqloc != "i":
        eqloc = funcs.flag_mantle(
            s1,
            sstr,
            dataframe.depth[i],
            dlim,
            sdep,
            sunc,
            dataframe.Ppl[i],
            dataframe.Tpl[i],
            eqloc,
        )
    # Determine quality (qual) of classification
    qual = funcs.determine_quality(eqloc, p_crustal, p_slab, p_int, s1, sstr)

    # add columns determined here to dataframe:
    dataframe.at[i, "p_crustal"] = p_crustal
    dataframe.at[i, "p_interface"] = p_int
    dataframe.at[i, "p_slab"] = p_slab
    dataframe.at[i, "eqloc"] = eqloc
    dataframe.at[i, "quality"] = qual
    dataframe.at[i, "slab2_region"] = smod

    # write dataframe to csv file:
    funcs.write_file(dataframe, outfile)
