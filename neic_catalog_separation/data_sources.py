"""Module for managing data sources for classifying earthquakes."""

import pathlib

import numpy as np
from mapio import reader

from . import funcs


class Slab:
    """Interface for Slab2 data."""

    def __init__(self, region: str):
        self.region = region
        self.depth_grid = None
        self.error_grid = None
        self.strike_grid = None
        self.dip_grid = None
        self.thickness_grid = None
        self.sup = None
        self.valid_values = None

    def load(self, data_dir: pathlib.Path):
        """Load data.

        Args:
            data_dir: Path to directory with Slab2 files.
        """

        def _get_file(pattern: str, data_dir: pathlib.Path):
            files = list(data_dir.glob(pattern))
            if len(files) != 1:
                raise IOError(f"Expected one file matching {pattern}. Found {files}.")
            return files[0]

        self.depth_grid = reader.read(
            _get_file(pattern=self.region + "_slab2_dep*.grd", data_dir=data_dir)
        )
        self.error_grid = reader.read(
            _get_file(pattern=self.region + "_slab2_unc*.grd", data_dir=data_dir)
        )
        self.strike_grid = reader.read(
            _get_file(pattern=self.region + "_slab2_str*.grd", data_dir=data_dir)
        )
        self.dip_grid = reader.read(
            _get_file(pattern=self.region + "_slab2_dip*.grd", data_dir=data_dir)
        )
        self.thickness_grid = reader.read(
            _get_file(pattern=self.region + "_slab2_thk*.grd", data_dir=data_dir)
        )
        if self.region in ["man", "ker", "izu", "sol"]:
            self.sup = np.genfromtxt(
                data_dir / f"{self.region}_slab2_sup.csv",
                delimiter=",",
                names=True,
                encoding="utf-8",
            )

        # Set up grid with valid values used in getting nearest valid data
        latitude, longitude = self.depth_grid.getLatLonMesh(
            self.depth_grid.getGeoDict()
        )
        latitude = np.flipud(latitude)
        depth = self.depth_grid.getData()
        mask = ~np.isnan(depth)
        depth = depth[mask]
        latitude = latitude[mask]
        longitude = longitude[mask]
        self.valid_values = {
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
        }

    def get_values(self, lat: float, lon: float) -> tuple:
        """Get slab values at location.

        Args:
            lat: Latitude of location (degrees).
            lon: Longitude of location (degrees).
        Returns:
            Tuple (depth, strike, dip, thickness, error) of slab information at location.
        """
        try:
            depth = self.depth_grid.getValue(lat, lon)
            error = self.error_grid.getValue(lat, lon)
            strike = self.strike_grid.getValue(lat, lon)
            dip = self.dip_grid.getValue(lat, lon)
            thickness = self.thickness_grid.getValue(lat, lon)
            if strike < 0.0:
                strike += 360.0
        except Exception:
            depth = error = dip = strike = thickness = np.nan
        return (abs(depth), strike, dip, thickness, error)

    def get_values_closest(self, lat: float, lon: float) -> tuple:
        """Get information for closest point in grid that contains values.

        Args:
            lat: Latitude of location (degrees).
            lon: Longitude of location (degrees).
        Returns:
            Tuple (depth, strike, dip, thickness, error, min_dist) of slab information at location.
        """
        latitude = self.valid_values["latitude"]
        longitude = self.valid_values["longitude"]
        geo_distance = funcs.haversine_distance(
            points_lat=latitude,
            points_lon=longitude,
            target_lat=lat,
            target_lon=lon,
        )
        min_index = np.argmin(geo_distance)
        nearest_lat = latitude[min_index]
        nearest_lon = longitude[min_index]
        (depth, strike, dip, thickness, error) = self.get_values(
            lat=nearest_lat, lon=nearest_lon
        )
        min_dist = geo_distance[min_index]
        return (depth, strike, dip, thickness, error, min_dist)

    def get_overturned(self, lat: float, lon: float) -> tuple:
        """Get depth, strike, and dip in region with overturned slab.

        Args:
            lat: Latitude of location (degrees).
            lon: Longitude of location (degrees).
        Returns:
            Tuple (depth, strike, dip) of overturned slab information at location.
        """
        depth = np.nan
        strike = np.nan
        dip = np.nan
        thickness = np.nan

        if self.sup is not None:
            sup_nearest = self.find_nearest_sup(lon=lon, lat=lat)
            # dont want to classify events that are far from the tilted slab (such as outer rise or
            # events that may not have been filtered out), so correct for that here (in km; 1 degree lon/lat is
            # ~111 km. Slabs can span 10s of degrees lon/lat, so lets assume anything greater than 250 km/2.5 degrees here)
            if sup_nearest["distance"] <= 250.0:
                depth = sup_nearest["depth"]
                strike = sup_nearest["strike"]
                dip = sup_nearest["dip"]
                thickness = sup_nearest["thickness"]

        return (depth, strike, dip, thickness)

    def find_nearest_sup(self, lon, lat):
        """
        Determine nearest lon,lat from tilted/vertical slab data to earthquake location and take tilted slab depth, strike, and dip at that point
        Args:
            lon (float): Earthquake longitude (deg)
            lat (float): Earthquake latitude (deg)
        Returns:
            nearest_lon (float): closest lon
            nearest_lat (float): closest lat
            nearest_depth (float): closest depth
            nearest_strike (float): closest strike
            nearest_dip (float): closest dip
            nearest_thickness (float): closest thickness
            nearest_dist (float): Distance of closest point (km)
        """
        geo_distance = funcs.haversine_distance(
            points_lat=self.sup["lat"],
            points_lon=self.sup["lon"],
            target_lat=lat,
            target_lon=lon,
        )
        min_index = np.argmin(geo_distance)

        sup_nearest = dict(zip(self.sup.dtype.names, self.sup[min_index]))
        sup_nearest["distance"] = geo_distance[min_index]
        return sup_nearest

    @staticmethod
    def find_closest(lat: float, lon: float, depth: float, slab1, slab2):
        """Find closest slab.

        Args:
            lat: Latitude (degrees) of location.
            lon: Longitude (degrees) of location.
            depth: Depth (km) of location.
            slab1: First Slab2 model.
            slab2: Second Slab2 model.
        Returns:
            Slab2 model that is closest to location.
        """
        closest = slab1

        slab1_depth, _, _, _, _ = slab1.get_values(lat=lat, lon=lon)
        slab1_ddiff = np.abs(depth - slab1_depth)

        slab2_depth, _, _, _, _ = slab2.get_values(lat=lat, lon=lon)
        slab2_ddiff = np.abs(depth - slab2_depth)

        # if both depths are nan, then the earthquake is outside of slab2 region, and is likely crustal or outer rise
        # this will be classified accordingly, so we can default to the first slab region supplied
        if np.isnan(slab1_depth) and np.isnan(slab2_depth):
            closest = slab1
        # if depth from first slab (slab1) is not nan and depth of earthquake is closest to depth of the slab1, then use slab1
        elif not np.isnan(slab1_depth):
            if np.isnan(slab2_ddiff):
                closest = slab1
            elif slab1_ddiff <= slab2_ddiff:
                closest = slab1
            else:
                closest = slab2
        # if depth from second slab (slab2) is not nan and depth of earthquake is closest to depth of the slab2, then use slab2
        elif not np.isnan(slab2_depth):
            if np.isnan(slab1_ddiff):
                closest = slab2
            elif slab2_ddiff < slab1_ddiff:
                closest = slab2
        else:
            raise NotImplementedError(
                f"Unknown case for finding closest slab slab1_ddiff={slab1_ddiff} slab2_ddiff={slab2_ddiff}."
            )

        return closest


class MohoDepth:
    """Interface for depth to Moho data."""

    def __init__(self):
        self.depth_grid = None

    def load(self, data_dir: pathlib.Path):
        """Load data.

        Args:
            data_dir: Path to directory with Moho depth file.
        """
        self.depth_grid = reader.read(data_dir / "depthtomoho.grd")

    def get_value(self, lat: float, lon: float) -> float:
        """Get Moho depth at location.

        Args:
            lat: Latitude of location (degrees).
            lon: Longitude of location (degrees).
        Returns:
            Moho depth (km) at location.
        """
        try:
            lon = lon if lon <= 180.0 else lon - 360.0
            depth = self.depth_grid.getValue(lat, lon)
        except Exception:
            depth = np.nan
        return abs(depth)


class SeismogenicDepth:
    """Interface for depth to seismogenic depth data."""

    def __init__(self):
        self.data = None

    def load(self, data_dir: pathlib.Path):
        """Load data.

        Args:
            data_dir: Path to directory with seismogenic depth file.
        """
        values = np.genfromtxt(
            data_dir / "seismogenic_depth.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        self.data = dict(
            [(d["Slab2_Code"], dict(zip(values.dtype.names, d))) for d in values]
        )
        for d in self.data.values():
            del d["Slab2_Code"]

    def get_values(self, slab_region: str, nshm: bool = False) -> tuple:
        """Get depth to the deep limit of the seismogenic zone based on Slab2.

        Args:
            slab_region: 3-letter code for Slab2 region.
            nshm: Flag to use NSHM limits to depth of seismogenic zone.
        Returns:
            Tuple of depth (km) and rake angle (degrees).
        """
        depth = 40.0
        rake = 90.0

        if slab_region == "car" and nshm:
            depth = 50.0
        elif slab_region == "mue" and nshm:
            depth = 40.0
        elif slab_region in self.data:
            values = self.data[slab_region]
            depth = values["Sd_km"]
            rake = values["λ_º"]
        else:
            pass  # use defaults

        return (depth, rake)


def slab(region: str, data_dir: pathlib.Path) -> Slab:
    """Create object holding Slab2 data.

    Args:
        region: 3-letter code for Slab2 region.
        data_dir: Directory containing Slab2 data files.
    Returns:
        Object with Slab2 data.
    """
    data = Slab(region=region)
    data.load(data_dir=data_dir)
    return data


def moho_depth(data_dir: pathlib.Path) -> MohoDepth:
    """Create object holding depth to Moho data.

    Args:
        data_dir: Directory containing Moho depth data files.
    Returns:
        Object with Moho depth data.
    """
    data = MohoDepth()
    data.load(data_dir=data_dir)
    return data


def seismogenic_depth(data_dir: pathlib.Path) -> SeismogenicDepth:
    """Create object holding seismogenic depth data.

    Args:
        data_dir: Directory containing seismogenic depth data files.
    Returns:
        Object with seismogenic depth data.
    """
    data = SeismogenicDepth()
    data.load(data_dir=data_dir)
    return data
