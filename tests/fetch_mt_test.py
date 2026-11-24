import io
import json
import logging
import math
import sys

import polars
import pytest

from neic_catalog_separation.utils import fetch_mt

TENSOR = {
    "tensor-mrr": "-2.04e+17",
    "tensor-mtt": "9.01e+18",
    "tensor-mpp": "-8.80e+18",
    "tensor-mrt": "3.03e+18",
    "tensor-mrp": "6.38e+18",
    "tensor-mtp": "9.30e+18",
}


def comcat_event(properties=None, products=None):
    """Event in the same structure as the ComCat GeoJSON response."""
    if products is None:
        products = {"moment-tensor": [{"properties": properties}]}
    return {"type": "Feature", "properties": {"products": products}}


class FakeResponse(io.BytesIO):
    """Stand-in for the object returned by urlopen()."""


@pytest.fixture
def catalog(tmp_path):
    filename = tmp_path / "catalog.csv"
    polars.DataFrame(
        {
            "id": ["us0001", "us0002", "us0003"],
            "magnitude": [6.1, 6.2, 6.3],
            # existing columns are replaced
            "mrr": [1.0, 1.0, 1.0],
        }
    ).write_csv(filename)
    return filename


class TestParseMomentTensor:
    def test_components(self):
        tensor = fetch_mt.parse_moment_tensor(comcat_event(TENSOR))

        assert set(tensor) == {"mrr", "mtt", "mpp", "mrt", "mrp", "mtp"}
        assert tensor["mrr"] == pytest.approx(-2.04e17)
        assert tensor["mtt"] == pytest.approx(9.01e18)
        assert tensor["mpp"] == pytest.approx(-8.80e18)
        assert tensor["mrt"] == pytest.approx(3.03e18)
        assert tensor["mrp"] == pytest.approx(6.38e18)
        assert tensor["mtp"] == pytest.approx(9.30e18)

    @pytest.mark.parametrize(
        "event",
        [
            {},
            {"properties": {}},
            {"properties": {"products": {}}},
            {"properties": {"products": {"moment-tensor": []}}},
            comcat_event(products={"origin": [{"properties": {}}]}),
        ],
    )
    def test_no_moment_tensor(self, event):
        tensor = fetch_mt.parse_moment_tensor(event)

        assert len(tensor) == 6
        assert all(math.isnan(value) for value in tensor.values())

    def test_missing_and_invalid_components(self):
        properties = dict(TENSOR)
        del properties["tensor-mtp"]
        properties["tensor-mrt"] = "not a number"
        tensor = fetch_mt.parse_moment_tensor(comcat_event(properties))

        assert math.isnan(tensor["mtp"])
        assert math.isnan(tensor["mrt"])
        assert tensor["mrr"] == pytest.approx(-2.04e17)


class TestFetchEvent:
    def test_request(self, monkeypatch):
        event = comcat_event(TENSOR)
        requests = []

        def urlopen(url, timeout):
            requests.append((url, timeout))
            return FakeResponse(json.dumps(event).encode("utf-8"))

        monkeypatch.setattr(fetch_mt, "urlopen", urlopen)

        assert fetch_mt.fetch_event("us0001") == event
        url, timeout = requests[0]
        assert url == (
            "https://earthquake.usgs.gov/fdsnws/event/1/query"
            "?eventid=us0001&format=geojson"
        )
        assert timeout == fetch_mt.REQUEST_TIMEOUT

    def test_event_id_is_quoted(self, monkeypatch):
        requests = []

        def urlopen(url, timeout):
            requests.append(url)
            return FakeResponse(b"{}")

        monkeypatch.setattr(fetch_mt, "urlopen", urlopen)
        fetch_mt.fetch_event("us0001&format=csv#x")

        assert "eventid=us0001%26format%3Dcsv%23x&format=geojson" in requests[0]


class TestFetchMtFromComcat:
    def test_fetch(self, catalog, tmp_path, monkeypatch):
        events = {
            "us0001": comcat_event(TENSOR),
            "us0002": comcat_event(products={}),  # no moment tensor
        }

        def fetch_event(event_id):
            if event_id == "us0003":
                raise OSError("connection failed")
            return events[event_id]

        monkeypatch.setattr(fetch_mt, "fetch_event", fetch_event)
        output = tmp_path / "output.csv"
        fetch_mt.fetch_mt_from_comcat(catalog, output, sleep_seconds=0)

        result = polars.read_csv(output)
        assert result["id"].to_list() == ["us0001", "us0002", "us0003"]
        # other columns are preserved, existing mrr column is replaced
        assert result["magnitude"].to_list() == [6.1, 6.2, 6.3]
        assert result["mrr"][0] == pytest.approx(-2.04e17)
        for name in fetch_mt.MT_COMPONENTS:
            assert result[name].dtype == polars.Float64
            assert result[name].is_nan().to_list() == [False, True, True]
        assert result["mtp"][0] == pytest.approx(9.30e18)

    def test_failed_request_logs_warning(self, catalog, tmp_path, monkeypatch, caplog):
        def fetch_event(event_id):
            raise OSError("connection failed")

        monkeypatch.setattr(fetch_mt, "fetch_event", fetch_event)
        with caplog.at_level(logging.WARNING):
            fetch_mt.fetch_mt_from_comcat(
                catalog, tmp_path / "output.csv", sleep_seconds=0
            )

        assert "us0001" in caplog.text
        assert "connection failed" in caplog.text

    def test_sleep_between_requests(self, catalog, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_mt, "fetch_event", lambda event_id: {})
        delays = []
        monkeypatch.setattr(fetch_mt.time, "sleep", delays.append)
        fetch_mt.fetch_mt_from_comcat(
            catalog, tmp_path / "output.csv", sleep_seconds=0.5
        )

        assert delays == [0.5, 0.5]  # not before the first request


class TestCli:
    def test_cli(self, catalog, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_mt, "fetch_event", lambda event_id: {})
        monkeypatch.setattr(fetch_mt.time, "sleep", lambda seconds: None)
        output = tmp_path / "cli_output.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify_fetch_mt",
                "--catalog",
                str(catalog),
                "--output",
                str(output),
            ],
        )
        fetch_mt.cli()

        result = polars.read_csv(output)
        assert result.height == 3
        assert set(fetch_mt.MT_COMPONENTS).issubset(result.columns)

    def test_cli_missing_arguments(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["classify_fetch_mt"])
        with pytest.raises(SystemExit):
            fetch_mt.cli()
