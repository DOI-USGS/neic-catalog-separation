#!/usr/bin/env python3
"""Application to show earthquake classification based on position.

# :TODO: Add Moho depth on plot.
# :TODO: Add slab interface on plot.
# :TODO: Add Moho 'buffer' on plot.
# :TODO: Add slab thickness on plot.
# :TODO: Add slab interface 'buffer' on plot.
# :TODO: Add slab thickness 'buffer' on plot.
"""
import dataclasses
import pathlib

import numpy
import polars


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--latitude",
        type=float,
        dest="latitude",
        required=True,
        help="Latitude for slice.",
    )
    parser.add_argument(
        "--min-longitude",
        type=float,
        dest="min_longitude",
        required=True,
        help="Minimum longitude for slice.",
    )
    parser.add_argument(
        "--max-longitude",
        type=float,
        dest="max_longitude",
        required=True,
        help="Maximum longitude for slice.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        dest="max_depth_km",
        required=True,
        help="Maximum depth (km) for slice.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Horizontal resolution for slice (vertical resolution will match).",
    )

    parser.add_argument("--regions", required=True, help="Slab2 regions")

    parser.add_argument(
        "--backend",
        default="plotly",
        choices=("matplotlib", "plotly"),
        help="Plotting backend.",
    )
    parser.add_argument(
        "--filename", default="slice.html", help="Filename for plot output."
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        dest="n_processes",
        help="Number of processes to use for classification.",
    )

    parser.add_argument(
        "--create-catalog",
        dest="create_catalog",
        action="store_true",
        help="Create catalog of fake earthquakes on vertical slice.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Classify earthquakes on vertical slice.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot classification of earthquakes on vertical slice.",
    )

    args = parser.parse_args()

    app = PlotSlice()
    if args.create_catalog:
        app.create_catalog(
            latitude=args.latitude,
            min_longitude=args.min_longitude,
            max_longitude=args.max_longitude,
            max_depth_km=args.max_depth_km,
            horiz_resolution=args.resolution,
        )
    if args.classify:
        app.classify(args.regions.split(","), args.n_processes)
    if args.plot:
        app.plot(backend=args.backend, filename=pathlib.Path(args.filename))

    return 0


@dataclasses.dataclass
class SliceData:
    longitude: numpy.ndarray
    depth: numpy.ndarray
    category: str
    p_crustal: numpy.ndarray
    p_interface: numpy.ndarray
    p_intraslab: numpy.ndarray
    p_mantle: numpy.ndarray
    p_outerrise: numpy.ndarray

    slab: numpy.ndarray
    moho: numpy.ndarray

    @staticmethod
    def from_dataframe(df: polars.DataFrame, slab: numpy.ndarray, moho: numpy.ndarray):
        data = SliceData(
            longitude=df["longitude"],
            depth=df["depth"],
            category=df["eq_category"],
            p_crustal=df["p_crustal"],
            p_interface=df["p_interface"],
            p_intraslab=df["p_intraslab"],
            p_mantle=df["p_mantle"],
            p_outerrise=df["p_outerrise"],
            slab=slab,
            moho=moho,
        )
        return data


class PlotSlice:

    _tmp_dir = pathlib.Path("tmp")
    _catalog_filename = _tmp_dir / "slice_catalog.csv"
    _classify_filename = _tmp_dir / "slice_classified.csv"

    def __init__(self):
        self.longitude_1d = None

    def create_catalog(
        self,
        latitude: float,
        min_longitude: float,
        max_longitude: float,
        max_depth_km: float,
        horiz_resolution: float,
    ):
        vert_resolution_km = horiz_resolution * 111.0
        longitude_1d = numpy.arange(min_longitude, max_longitude, horiz_resolution)
        depth_1d = numpy.arange(0.0, max_depth_km, vert_resolution_km)
        self.longitude_1d = longitude_1d

        longitude, depth = numpy.meshgrid(longitude_1d, depth_1d)
        longitude = longitude.ravel()
        depth = depth.ravel()

        unc = 5.0 * numpy.ones_like(depth)
        eq_id = [f"xx{i:08d}" for i in range(len(depth))]
        etype = ["EQ"] * len(depth)
        mag = 7.0 * numpy.ones_like(depth)
        time = numpy.nan * numpy.ones_like(depth)
        Paz = numpy.nan * numpy.ones_like(depth)
        Ppl = numpy.nan * numpy.ones_like(depth)
        Taz = numpy.nan * numpy.ones_like(depth)
        Tpl = numpy.nan * numpy.ones_like(depth)
        S1 = numpy.nan * numpy.ones_like(depth)
        D1 = numpy.nan * numpy.ones_like(depth)
        R1 = numpy.nan * numpy.ones_like(depth)
        S2 = numpy.nan * numpy.ones_like(depth)
        D2 = numpy.nan * numpy.ones_like(depth)
        R2 = numpy.nan * numpy.ones_like(depth)
        mlon = numpy.nan * numpy.ones_like(depth)
        mlat = numpy.nan * numpy.ones_like(depth)
        mdep = numpy.nan * numpy.ones_like(depth)
        id_no = numpy.nan * numpy.ones_like(depth)
        src = numpy.nan * numpy.ones_like(depth)

        df = polars.DataFrame(
            data={
                "latitude": latitude,
                "longitude": longitude,
                "depth": depth,
                "unc": unc,
                "id": eq_id,
                "etype": etype,
                "mag": mag,
                "time": time,
                "Paz": Paz,
                "Ppl": Ppl,
                "Taz": Taz,
                "Tpl": Tpl,
                "S1": S1,
                "D1": D1,
                "R1": R1,
                "S2": S2,
                "D2": D2,
                "R2": R2,
                "mlon": mlon,
                "mlat": mlat,
                "mdep": mdep,
                "id_no": id_no,
                "src": src,
            }
        )
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        df.write_csv(self._catalog_filename)

    def classify(self, regions: list, n_processes: int = 1):
        from neic_catalog_separation import classify_catalog

        classify_catalog.classify_catalog(
            output_filename=self._classify_filename,
            slab_region1=regions[0],
            slab_region2=regions[1] if len(regions) > 1 else None,
            catalog_filename=self._catalog_filename,
            config_filename=pathlib.Path(__file__).parent.parent
            / "neic_catalog_separation"
            / "config"
            / "subduction.yml",
            num_processes=n_processes,
            tolerance=15.0,
            nshm=True,
            no_mantle=False,
            int_cutoff=False,
        )

    def plot(self, filename: pathlib.Path, backend: str = "plotly"):

        data = polars.read_csv(self._classify_filename)
        # :TODO: Get slab and moho profiles and add to plot.
        slab = None
        moho = None
        plot_data = SliceData.from_dataframe(df=data, slab=slab, moho=moho)

        if backend == "plotly":
            self._render_plotly(filename=filename, data=plot_data)
        else:
            self._render_matplotlib(filename=filename, data=plot_data)

    def _get_slab(self):
        from neic_catalog_separation import data_sources

        longitude = self.longitude_1d
        latitude = self.latitude_1d

        slab_data = data_sources.Slab(self.region)
        slab_data.load()
        slab = slab_data.get_values(lat=latitude, lon=longitude)
        slab = numpy.stack((longitude, slab)).T
        return slab

    def _get_moho(self):
        from neic_catalog_separation import data_sources

        longitude = self.longitude_1d
        latitude = self.latitude_1d

        moho_data = data_sources.MohoDepth(self.region)
        moho_data.load(data_dir=XX)
        moho = moho_data.get_value(lat=latitude, lon=longitude)
        moho = numpy.stack((longitude, moho)).T
        return moho

    def _get_seismogenic_depth(self):
        from neic_catalog_separation import data_sources

        longitude = self.longitude_1d
        latitude = self.latitude_1d

        moho_data = data_sources.MohoDepth(self.region)
        moho_data.load(data_dir=XX)
        moho = moho_data.get_value(lat=latitude, lon=longitude)
        moho = numpy.stack((longitude, moho)).T
        return moho

    def _render_plotly(self, filename: pathlib.Path, data: SliceData):
        import plotly.express

        figure = plotly.express.scatter(
            x=data.longitude,
            y=data.depth,
            color=data.category,
            size=8 * numpy.ones_like(data.depth),
            hover_data={
                "category": data.category,
                "p_crustal": data.p_crustal,
                "p_interface": data.p_interface,
                "p_intraslab": data.p_intraslab,
                "p_outerrise": data.p_outerrise,
                "p_mantle": data.p_mantle,
            },
            color_discrete_map={
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
            },
            title="Vertical cross section",
            labels={
                "x": "Longitude",
                "y": "Depth, km",
            },
        )
        figure.update_yaxes(autorange="reversed")
        figure.write_html(filename)

    def _render_matplotlib(self, filename: pathlib.Path, data: SliceData):
        return


if __name__ == "__main__":
    raise SystemExit(cli())
