import json
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import polars  # noqa: E402
import pytest  # noqa: E402

from neic_catalog_separation.utils import calculate_bvalue  # noqa: E402

B_VALUE = 1.0
MC = 4.0
BIN_SIZE = 0.1


def gutenberg_richter_magnitudes(n_events=20000, seed=1234):
    """Magnitudes that are complete above MC and follow Gutenberg-Richter with b=1."""
    rng = np.random.default_rng(seed)
    mags = MC - BIN_SIZE / 2.0 + rng.exponential(1.0 / (B_VALUE * np.log(10)), n_events)
    return np.around(mags, 1)


@pytest.fixture(scope="module")
def magnitudes():
    return gutenberg_richter_magnitudes()


@pytest.fixture
def catalog(tmp_path, magnitudes):
    filename = tmp_path / "catalog.csv"
    polars.DataFrame({"id": range(len(magnitudes)), "magnitude": magnitudes}).write_csv(
        filename
    )
    return filename


class TestCalculateWW2000:
    def test_b_value(self, magnitudes):
        stats, mag_bins = calculate_bvalue.calculate_ww2000(
            mags=magnitudes, mc_est=MC, bin_size=BIN_SIZE
        )

        assert stats["b_value"] == pytest.approx(B_VALUE, abs=0.1)
        assert stats["mc"] == pytest.approx(MC, abs=0.3)
        assert stats["a_value"] > 0.0
        assert stats["std_dev"] == pytest.approx(0.01, abs=0.01)
        assert 0.0 <= stats["residual"] <= 100.0

    def test_stats_are_scalars(self, magnitudes):
        """All statistics are plain numbers (and can be written to JSON)."""
        stats, _ = calculate_bvalue.calculate_ww2000(
            mags=magnitudes, mc_est=MC, bin_size=BIN_SIZE
        )

        assert set(stats) == {
            "mc",
            "b_value",
            "a_value",
            "std_dev",
            "residual",
            "error_sb",
        }
        for value in stats.values():
            assert np.ndim(value) == 0
        json.dumps({key: float(value) for key, value in stats.items()})

    def test_mag_bins(self, magnitudes):
        _, mag_bins = calculate_bvalue.calculate_ww2000(
            mags=magnitudes, mc_est=MC, bin_size=BIN_SIZE
        )

        assert mag_bins[0] == 0.0
        assert np.diff(mag_bins) == pytest.approx(BIN_SIZE)
        assert mag_bins[-1] >= magnitudes.max() - BIN_SIZE

    def test_nan_magnitudes_ignored(self, magnitudes):
        with_nan = np.concatenate([magnitudes, np.full(100, np.nan)])
        expected, _ = calculate_bvalue.calculate_ww2000(
            mags=magnitudes, mc_est=MC, bin_size=BIN_SIZE
        )
        stats, _ = calculate_bvalue.calculate_ww2000(
            mags=with_nan, mc_est=MC, bin_size=BIN_SIZE
        )

        assert stats == expected


class TestCalculateBvalue:
    def test_output_json(self, catalog, capsys):
        calculate_bvalue.calculate_bvalue(
            catalog_filename=catalog, mc_est=MC, bin_size=BIN_SIZE, show_plot=False
        )

        output = catalog.with_name("catalog_bvalue.json")
        assert output.exists()
        with open(output, encoding="utf-8") as infile:
            stats = json.load(infile)
        assert stats["b_value"] == pytest.approx(B_VALUE, abs=0.1)
        assert stats["gof"] == pytest.approx(100.0 - stats["residual"])
        assert "b-value" in capsys.readouterr().out
        assert not catalog.with_name("catalog_bvalue.pdf").exists()

    def test_string_filename(self, catalog):
        calculate_bvalue.calculate_bvalue(
            catalog_filename=str(catalog), mc_est=MC, bin_size=BIN_SIZE, show_plot=False
        )

        assert catalog.with_name("catalog_bvalue.json").exists()

    def test_plot(self, catalog):
        calculate_bvalue.calculate_bvalue(
            catalog_filename=catalog, mc_est=MC, bin_size=BIN_SIZE, show_plot=True
        )

        plot = catalog.with_name("catalog_bvalue.pdf")
        assert plot.exists()
        assert plot.read_bytes().startswith(b"%PDF")

    def test_mag_column(self, tmp_path, magnitudes):
        """Output of classify_catalog uses column 'mag'."""
        filename = tmp_path / "classified.csv"
        polars.DataFrame({"id": range(len(magnitudes)), "mag": magnitudes}).write_csv(
            filename
        )
        calculate_bvalue.calculate_bvalue(
            catalog_filename=filename, mc_est=MC, bin_size=BIN_SIZE, show_plot=False
        )

        assert (tmp_path / "classified_bvalue.json").exists()

    def test_no_magnitude_column(self, tmp_path):
        filename = tmp_path / "bad.csv"
        polars.DataFrame({"id": [1, 2], "depth": [10.0, 20.0]}).write_csv(filename)

        with pytest.raises(ValueError, match="magnitude"):
            calculate_bvalue.calculate_bvalue(
                catalog_filename=filename, mc_est=MC, bin_size=BIN_SIZE
            )

    def test_all_nan_magnitudes(self, tmp_path, capsys):
        filename = tmp_path / "nan.csv"
        polars.DataFrame({"magnitude": [float("nan")] * 3}).write_csv(filename)

        calculate_bvalue.calculate_bvalue(
            catalog_filename=filename, mc_est=MC, bin_size=BIN_SIZE
        )

        assert "No magnitudes" in capsys.readouterr().out
        assert not (tmp_path / "nan_bvalue.json").exists()


class TestCli:
    def test_cli(self, catalog, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify_calculate_bvalue",
                "--catalog",
                str(catalog),
                "--completeness-magnitude",
                str(MC),
                "--bin-size",
                str(BIN_SIZE),
                "--plot",
            ],
        )
        calculate_bvalue.cli()

        assert catalog.with_name("catalog_bvalue.json").exists()
        assert catalog.with_name("catalog_bvalue.pdf").exists()

    def test_cli_without_plot(self, catalog, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify_calculate_bvalue",
                "--catalog",
                str(catalog),
                "--completeness-magnitude",
                str(MC),
            ],
        )
        calculate_bvalue.cli()

        assert catalog.with_name("catalog_bvalue.json").exists()
        assert not catalog.with_name("catalog_bvalue.pdf").exists()

    def test_cli_missing_arguments(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["classify_calculate_bvalue"])
        with pytest.raises(SystemExit):
            calculate_bvalue.cli()
