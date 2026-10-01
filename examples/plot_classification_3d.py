#!/usr/bin/env python3
"""Plot the results of `classify_catalog` as an interactive 3D (longitude,
latitude, depth) scatter plot in HTML format, colored by earthquake category.

Optionally overlays the Slab2 surface for a given region as a 3D surface, and/or
adds a 2D geo map panel (with coastlines and country borders) showing the same
events colored by category, and/or adds a focal mechanism ("beachball") map
panel showing each event's nodal-plane solution (from the S1/D1/R1 columns).
Written with AI-assisted coding tool Kiro CLI 2.16.2 using claude-sonnet-5.
"""
import argparse
import base64
import json
import pathlib
import sys

import numpy
import plotly.graph_objects
import polars
from plotly.subplots import make_subplots

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
import focal_mechanism  # noqa: E402

from neic_catalog_separation import data_sources

# Same category/color mapping used in plot_classification_slice.py.
CATEGORY_COLORS = {
    "crustal": "orange",
    "subduction_interface": "orchid",
    "subduction_intraslab": "darkgray",
    "subduction_outerrise": "red",
    "mantle": "wheat",
    "crustal_mantle": "salmon",
    "subduction_interface_mantle": "purple",
    "subduction_intraslab_mantle": "lightslategray",
    "subduction_outerrise_mantle": "darkred",
    "unknown": "pink",
}


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--catalog",
        dest="catalog_filename",
        required=True,
        help="Path to a classified catalog CSV file (output of classify_catalog), "
        "must contain longitude, latitude, depth, and eq_category columns.",
    )
    parser.add_argument(
        "--filename",
        default="classification_3d.html",
        help="Filename for the HTML plot output.",
    )
    parser.add_argument(
        "--slab2-region",
        dest="slab2_region",
        default=None,
        help="3-letter Slab2 region code. If given, the Slab2 "
        "depth surface for this region is overlaid on the 3D scatter plot.",
    )
    parser.add_argument(
        "--slab-data-dir",
        dest="slab_data_dir",
        default=None,
        help="Directory containing Slab2 grid files. Defaults to the Slab2 data "
        "included in this repository.",
    )
    parser.add_argument(
        "--coastlines",
        action="store_true",
        help="Add a 2D geo map panel with coastlines and country borders.",
    )
    parser.add_argument(
        "--focal-mechanisms",
        dest="focal_mechanisms",
        action="store_true",
        help="Add a map panel plotting each event's focal mechanism ('beachball') "
        "at its longitude/latitude, using the S1/D1/R1 nodal-plane columns. Note: "
        "Plotly cannot anchor images to a geo/coastline map, so this panel uses a "
        "plain longitude/latitude cartesian axis instead.",
    )
    parser.add_argument(
        "--focal-mechanism-size",
        dest="focal_mechanism_size_deg",
        type=float,
        default=None,
        help="Size (degrees longitude) of each beachball symbol on the focal "
        "mechanism panel. Defaults to roughly 3%% of the catalog's longitude span.",
    )
    parser.add_argument(
        "--max-focal-mechanisms",
        dest="max_focal_mechanisms",
        type=int,
        default=500,
        help="Maximum number of focal mechanisms to render (to keep the plot "
        "file size and load time reasonable for large catalogs). Events beyond "
        "this count are skipped.",
    )
    parser.add_argument(
        "--outerrise-geojson",
        dest="outerrise_geojson",
        default=None,
        help="Path to a GeoJSON file (e.g. amsam2026_outerrise.json) whose "
        "polygon boundary/boundaries are drawn as an outline on the focal "
        "mechanism map panel (requires --focal-mechanisms).",
    )
    args = parser.parse_args()

    plot_classification_3d(
        catalog_filename=args.catalog_filename,
        filename=pathlib.Path(args.filename),
        slab2_region=args.slab2_region,
        slab_data_dir=args.slab_data_dir,
        coastlines=args.coastlines,
        focal_mechanisms=args.focal_mechanisms,
        focal_mechanism_size_deg=args.focal_mechanism_size_deg,
        max_focal_mechanisms=args.max_focal_mechanisms,
        outerrise_geojson=args.outerrise_geojson,
    )

    return 0


def _get_slab_surface(region: str, data_dir):
    """Load the Slab2 depth surface for a region as (longitude, latitude, depth) grids."""
    slab_data = data_sources.slab(region=region, data_dir=data_dir)
    geo_dict = slab_data.depth_grid.getGeoDict()
    latitude, longitude = slab_data.depth_grid.getLatLonMesh(geo_dict)
    # getLatLonMesh builds latitude ascending from ymin, but getData()'s rows
    # run from the north edge (ymax) down to the south edge (ymin), so the
    # latitude mesh must be flipped to align with the data array (see the
    # same np.flipud(latitude) correction in data_sources.Slab.load()).
    latitude = numpy.flipud(latitude)
    depth = numpy.abs(slab_data.depth_grid.getData())
    return longitude, latitude, depth


def _slab_uses_0_360_longitude(region: str, data_dir) -> bool:
    """Determine whether a Slab2 region's grid uses 0-360 longitude that spans
    the dateline (e.g. Kermadec: 169 to 188 degrees), as opposed to the -180/180
    convention. Regions crossing the dateline store xmax < xmin in the geo dict
    (see mapio Grid2D.getLatLonMesh), so getLatLonMesh returns longitudes above
    180 to keep the grid contiguous.
    """
    slab_data = data_sources.slab(region=region, data_dir=data_dir)
    geo_dict = slab_data.depth_grid.getGeoDict()
    return geo_dict.xmax < geo_dict.xmin


def _normalize_longitude_to_0_360(longitude):
    """Convert longitude(s) from -180/180 to 0-360 convention where negative."""
    return numpy.where(longitude < 0.0, longitude + 360.0, longitude)


def _add_scatter3d(figure, data, row, col, longitude=None):
    plot_longitude = numpy.asarray(longitude if longitude is not None else data["longitude"])
    for category, color in CATEGORY_COLORS.items():
        mask = (data["eq_category"] == category).to_numpy()
        if mask.sum() == 0:
            continue
        subset = data.filter(mask)
        subset_longitude = plot_longitude[mask]
        customdata, hovertemplate = _probability_customdata_and_template(
            subset, lon_key="x", lat_key="y"
        )
        figure.add_trace(
            plotly.graph_objects.Scatter3d(
                x=subset_longitude,
                y=subset["latitude"],
                z=subset["depth"],
                mode="markers",
                marker=dict(size=3, color=color),
                name=category,
                customdata=customdata,
                hovertemplate=hovertemplate + "<extra>" + category + "</extra>",
                hoverlabel=dict(bgcolor=color),
            ),
            row=row,
            col=col,
        )


def _add_slab_surface(figure, region, data_dir, row, col):
    longitude, latitude, depth = _get_slab_surface(region=region, data_dir=data_dir)
    figure.add_trace(
        plotly.graph_objects.Surface(
            x=longitude,
            y=latitude,
            z=depth,
            colorscale="Greys",
            opacity=0.5,
            showscale=False,
            name=f"{region} slab surface",
        ),
        row=row,
        col=col,
    )


def _hover_columns(subset):
    """Probability columns to include in hover text, matching the 3D panel."""
    return [
        c
        for c in ("p_crustal", "p_interface", "p_intraslab", "p_mantle", "p_outerrise")
        if c in subset.columns
    ]


def _probability_customdata_and_template(subset, lon_key: str, lat_key: str):
    """Build custom data array and hover template showing id/lon/lat/depth.
    """
    hover_columns = _hover_columns(subset)
    has_id = "id" in subset.columns
    columns = (["id"] if has_id else []) + ["depth"] + hover_columns
    # Use dtype=object since "id" is a string column and would otherwise force
    # the whole array to string dtype, breaking the ":.1f" format specifiers
    # used for the numeric (depth/probability) columns below.
    customdata = numpy.empty((len(subset), len(columns)), dtype=object)
    for i, c in enumerate(columns):
        customdata[:, i] = subset[c].to_numpy()

    depth_index = 1 if has_id else 0
    hovertemplate = f"lon=%{{{lon_key}}}<br>lat=%{{{lat_key}}}<br>"
    if has_id:
        hovertemplate += "id=%{customdata[0]}<br>"
    hovertemplate += f"depth=%{{customdata[{depth_index}]:.1f}} km<br>" + "<br>".join(
        f"{c}=%{{customdata[{depth_index + 1 + i}]:.1f}}"
        for i, c in enumerate(hover_columns)
    )
    return customdata, hovertemplate


def _add_geo_scatter(figure, data, row, col):
    for category, color in CATEGORY_COLORS.items():
        mask = data["eq_category"] == category
        if mask.sum() == 0:
            continue
        subset = data.filter(mask)
        customdata, hovertemplate = _probability_customdata_and_template(
            subset, lon_key="lon", lat_key="lat"
        )
        figure.add_trace(
            plotly.graph_objects.Scattergeo(
                lon=subset["longitude"],
                lat=subset["latitude"],
                mode="markers",
                marker=dict(size=4, color=color),
                name=category,
                showlegend=False,
                customdata=customdata,
                hovertemplate=hovertemplate + "<extra>" + category + "</extra>",
                hoverlabel=dict(bgcolor=color),
            ),
            row=row,
            col=col,
        )


def _geojson_polygon_rings(geojson_path):
    """Yield (longitudes, latitudes) coordinate rings for every Polygon /
    MultiPolygon in a GeoJSON file, so they can be drawn as outlines.
    """
    with open(geojson_path) as handle:
        geojson = json.load(handle)

    if geojson.get("type") == "FeatureCollection":
        geometries = [feature.get("geometry") for feature in geojson.get("features", [])]
    elif geojson.get("type") == "Feature":
        geometries = [geojson.get("geometry")]
    else:
        geometries = [geojson]

    for geometry in geometries:
        if not geometry:
            continue
        geometry_type = geometry.get("type")
        if geometry_type == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry_type == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            for ring in polygon:
                longitudes = [coord[0] for coord in ring]
                latitudes = [coord[1] for coord in ring]
                yield longitudes, latitudes


def _add_geojson_outline(figure, geojson_path, row, col, xaxis=None, yaxis=None):
    """Draw the polygon boundary/boundaries from a GeoJSON file as a line
    outline on a cartesian (longitude/latitude) subplot panel.
    """
    for index, (longitudes, latitudes) in enumerate(_geojson_polygon_rings(geojson_path)):
        trace = plotly.graph_objects.Scatter(
            x=longitudes,
            y=latitudes,
            mode="lines",
            line=dict(color="black", width=2),
            name="outer-rise region",
            legendgroup="outerrise-region",
            # Only show one legend entry for potentially many rings.
            showlegend=index == 0,
            hoverinfo="skip",
        )
        if xaxis is not None and yaxis is not None:
            trace.update(xaxis=xaxis, yaxis=yaxis)
            figure.add_trace(trace)
        else:
            figure.add_trace(trace, row=row, col=col)


def _add_focal_mechanism_panel(
    figure,
    data,
    row,
    col,
    size_deg: float = None,
    max_mechanisms: int = 500,
    outerrise_geojson=None,
):
    """Add a plain longitude/latitude panel plotting each event's focal
    mechanism ("beachball") image at its location, plus a category-colored
    marker for events without S1/D1/R1 data.

    Plotly cannot anchor `layout.images` to a `geo` map's lon/lat axes (only
    to cartesian x/y axes or paper coordinates), so this panel uses a plain
    cartesian longitude/latitude axis rather than the coastline geo map.
    """
    # The S1/D1/R1 columns may be read as strings (e.g. when the CSV has empty
    # or non-numeric entries for events without a moment tensor), in which case
    # `is_not_nan()` raises InvalidOperationError. Cast them to Float64 first
    # (non-numeric values become null) so both null and NaN entries are treated
    # as "no mechanism".
    data = data.with_columns(
        polars.col(column).cast(polars.Float64, strict=False)
        for column in ("S1", "D1", "R1")
    )

    has_mechanism = (
        data["S1"].is_not_null()
        & data["D1"].is_not_null()
        & data["R1"].is_not_null()
        & data["S1"].is_not_nan()
        & data["D1"].is_not_nan()
        & data["R1"].is_not_nan()
    )
    with_mechanism = data.filter(has_mechanism)
    without_mechanism = data.filter(~has_mechanism)

    if len(with_mechanism) > max_mechanisms:
        with_mechanism = with_mechanism[:max_mechanisms]

    if size_deg is None:
        lon_span = float(data["longitude"].max() - data["longitude"].min())
        size_deg = max(lon_span * 0.03, 0.05)

    # Add one (invisible) marker trace per category first via add_trace(row=,
    # col=), which lets plotly assign the correct subplot axis IDs; then read
    # those IDs back off the trace so the beachball images (added via
    # add_layout_image, which does not take row/col) reference the same axes.
    # Splitting by category (rather than one trace for all events) lets each
    # trace's hoverlabel color match that category, same as the other panels.
    xaxis = yaxis = None
    for category, color in CATEGORY_COLORS.items():
        subset = with_mechanism.filter(with_mechanism["eq_category"] == category)
        if len(subset) == 0:
            continue
        customdata, hovertemplate = _probability_customdata_and_template(
            subset, lon_key="x", lat_key="y"
        )
        marker_trace = plotly.graph_objects.Scatter(
            x=subset["longitude"],
            y=subset["latitude"],
            mode="markers",
            marker=dict(size=1, opacity=0.0),
            name=category,
            showlegend=False,
            customdata=customdata,
            hovertemplate=hovertemplate + "<extra>" + category + "</extra>",
            hoverlabel=dict(bgcolor=color),
        )
        figure.add_trace(marker_trace, row=row, col=col)
        if xaxis is None:
            added_trace = figure.data[-1]
            xaxis = "x" + added_trace.xaxis[1:] if added_trace.xaxis else "x"
            yaxis = "y" + added_trace.yaxis[1:] if added_trace.yaxis else "y"

    if xaxis is None:
        # No events with a focal mechanism; still need axis IDs for the
        # "no focal mechanism" trace added below to attach to the right subplot.
        placeholder = plotly.graph_objects.Scatter(x=[], y=[], mode="markers", showlegend=False)
        figure.add_trace(placeholder, row=row, col=col)
        added_trace = figure.data[-1]
        xaxis = "x" + added_trace.xaxis[1:] if added_trace.xaxis else "x"
        yaxis = "y" + added_trace.yaxis[1:] if added_trace.yaxis else "y"

    # Cache rendered beachball PNGs by rounded (strike, dip, rake) so events
    # sharing a mechanism (e.g. the synthetic test catalogs) don't re-render.
    image_cache = {}
    for row_data in with_mechanism.iter_rows(named=True):
        strike, dip, rake = row_data["S1"], row_data["D1"], row_data["R1"]
        key = (round(strike, 1), round(dip, 1), round(rake, 1))
        if key not in image_cache:
            color = CATEGORY_COLORS.get(row_data["eq_category"], "black")
            png_bytes = focal_mechanism.render_beachball_png(
                strike, dip, rake, size_px=60, compressional_color=color
            )
            image_cache[key] = "data:image/png;base64," + base64.b64encode(png_bytes).decode(
                "ascii"
            )

        figure.add_layout_image(
            dict(
                source=image_cache[key],
                xref=xaxis,
                yref=yaxis,
                x=row_data["longitude"],
                y=row_data["latitude"],
                xanchor="center",
                yanchor="middle",
                sizex=size_deg,
                sizey=size_deg,
                sizing="contain",
                layer="above",
            )
        )

    for category, color in CATEGORY_COLORS.items():
        subset = without_mechanism.filter(without_mechanism["eq_category"] == category)
        if len(subset) == 0:
            continue
        customdata, hovertemplate = _probability_customdata_and_template(
            subset, lon_key="x", lat_key="y"
        )
        figure.add_trace(
            plotly.graph_objects.Scatter(
                x=subset["longitude"],
                y=subset["latitude"],
                mode="markers",
                marker=dict(size=3, color=color),
                name=category,
                showlegend=False,
                customdata=customdata,
                hovertemplate=hovertemplate
                + "(no focal mechanism data)<extra>"
                + category
                + "</extra>",
                hoverlabel=dict(bgcolor=color),
            ),
            row=row,
            col=col,
        )

    if outerrise_geojson is not None:
        _add_geojson_outline(
            figure, outerrise_geojson, row=row, col=col, xaxis=xaxis, yaxis=yaxis
        )


def plot_classification_3d(
    catalog_filename: str,
    filename: pathlib.Path,
    slab2_region: str = None,
    slab_data_dir: str = None,
    coastlines: bool = False,
    focal_mechanisms: bool = False,
    focal_mechanism_size_deg: float = None,
    max_focal_mechanisms: int = 500,
    outerrise_geojson=None,
):
    """Render a 3D (longitude, latitude, depth) scatter plot of a classified
    earthquake catalog, colored by eq_category, and write it to an HTML file.

    Optionally overlays the Slab2 depth surface for `slab2_region` in the 3D
    scene, adds a 2D geo map panel with coastlines/country borders when
    `coastlines` is True, and/or adds a focal mechanism ("beachball") map
    panel when `focal_mechanisms` is True.
    """
    # infer_schema_length=None scans the whole file before picking column
    # dtypes, rather than just the first N rows
    data = polars.read_csv(
        catalog_filename, null_values=["NaN", "nan"], infer_schema_length=None
    )

    panels = ["scene"]
    titles = ["3D Viewer"]
    if coastlines:
        panels.append("geo")
        titles.append("Map View")
    if focal_mechanisms:
        panels.append("xy")
        titles.append("Interactive Map View with Focal Mechanisms")

    specs = [[{"type": panel} for panel in panels]]
    column_widths = [1.0 / len(panels)] * len(panels)
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        specs=specs,
        column_widths=column_widths,
        subplot_titles=titles,
    )

    _add_scatter3d_kwargs = {}
    slab_data_dir_path = None
    if slab2_region:
        top_dir = pathlib.Path(__file__).parent.parent
        slab_data_dir_path = (
            pathlib.Path(slab_data_dir)
            if slab_data_dir
            else top_dir / "neic_catalog_separation" / "data" / "Slab2"
        )
        # Some Slab2 regions cross the antimeridian and store
        # their grid using 0-360 longitude (e.g. 169 to 188 degrees) rather
        # than -180/180, matching the convention classify_earthquake.py uses
        # when looking up slab values (it adds 360 to negative longitudes
        # before the lookup). The catalog's raw longitude column is in
        # -180/180, so it must be converted the same way here for the 3D
        # scatter points to line up with the slab surface.
        if _slab_uses_0_360_longitude(slab2_region, slab_data_dir_path):
            _add_scatter3d_kwargs["longitude"] = _normalize_longitude_to_0_360(
                data["longitude"].to_numpy()
            )

    _add_scatter3d(figure, data, row=1, col=1, **_add_scatter3d_kwargs)

    if slab2_region:
        _add_slab_surface(figure, region=slab2_region, data_dir=slab_data_dir_path, row=1, col=1)

    next_col = 2
    if coastlines:
        _add_geo_scatter(figure, data, row=1, col=next_col)
        # Zoom the map tightly to the catalog's extent (with a small margin)
        # rather than relying on fitbounds, which adds more generous padding.
        lon_min = float(data["longitude"].min())
        lon_max = float(data["longitude"].max())
        lat_min = float(data["latitude"].min())
        lat_max = float(data["latitude"].max())
        lon_margin = max((lon_max - lon_min) * 0.08, 0.5)
        lat_margin = max((lat_max - lat_min) * 0.08, 0.5)
        figure.update_geos(
            resolution=50,
            showcoastlines=True,
            coastlinecolor="black",
            coastlinewidth=1.5,
            showcountries=True,
            countrycolor="gray",
            showland=True,
            landcolor="whitesmoke",
            showocean=True,
            oceancolor="lightblue",
            lonaxis_range=[lon_min - lon_margin, lon_max + lon_margin],
            lataxis_range=[lat_min - lat_margin, lat_max + lat_margin],
            row=1,
            col=next_col,
        )
        next_col += 1

    if focal_mechanisms:
        _add_focal_mechanism_panel(
            figure,
            data,
            row=1,
            col=next_col,
            size_deg=focal_mechanism_size_deg,
            max_mechanisms=max_focal_mechanisms,
            outerrise_geojson=outerrise_geojson,
        )
        figure.update_xaxes(title_text="Longitude", row=1, col=next_col)
        figure.update_yaxes(
            title_text="Latitude", scaleratio=1, scaleanchor="x", row=1, col=next_col
        )
        next_col += 1

    figure.update_scenes(zaxis_autorange="reversed")
    figure.update_layout(
        title="Earthquake Classification Results",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth, km",
        ),
    )
    figure.write_html(filename, config={"scrollZoom": True})


if __name__ == "__main__":
    raise SystemExit(cli())
