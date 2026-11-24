import sys

import polars
import pytest

from neic_catalog_separation.utils import separate_csv

CATEGORIES = [
    "crustal",
    "subduction_interface",
    "subduction_intraslab",
    "subduction_outerrise",
    "mantle",
    "unknown",
    "crustal-mantle",
    "subduction_interface-mantle",
    "subduction_interface-subduction_intraslab",
    "mantle-subduction_intraslab",
]


@pytest.fixture
def catalog(tmp_path):
    """Classified catalog with one event per category plus extra crustal events.

    Events with an even index have a focal mechanism (S1), the others do not.
    """
    categories = CATEGORIES + ["crustal", "crustal"]
    strike = [float(10 * i) if i % 2 == 0 else float("nan") for i in range(12)]
    df = polars.DataFrame(
        {
            "id": [f"ev{i}" for i in range(len(categories))],
            "eq_category": categories,
            "S1": strike,
        }
    )
    filename = tmp_path / "classified.csv"
    df.write_csv(filename)
    return filename


class TestCategoryFilenameLabel:
    @pytest.mark.parametrize(
        "category, label",
        [
            ("crustal", "crustal"),
            ("subduction_interface", "interface"),
            ("subduction_intraslab", "intraslab"),
            ("subduction_outerrise", "outerrise"),
            ("mantle", "mantle"),
            ("unknown", "unknown"),
            ("crustal-mantle", "crustal_mantle"),
            ("subduction_intraslab-mantle", "intraslab_mantle"),
            ("subduction_interface-mantle", "interface_mantle"),
            ("subduction_interface-subduction_intraslab", "interface_intraslab"),
        ],
    )
    def test_label(self, category, label):
        assert separate_csv.category_filename_label(category) == label


class TestSeparateInput:
    def test_all_events_written_once(self, catalog, tmp_path):
        """Every event, including combined categories, is in exactly one file."""
        prefix = str(tmp_path / "out")
        filenames = separate_csv.separate_input(
            catalog_filename=catalog, output_file=prefix, moment_tensors=False
        )

        assert len(filenames) == len(set(CATEGORIES))
        ids = []
        for filename in filenames:
            ids += polars.read_csv(filename)["id"].to_list()
        assert sorted(ids) == sorted(f"ev{i}" for i in range(12))

    def test_file_names_and_contents(self, catalog, tmp_path):
        prefix = str(tmp_path / "out")
        separate_csv.separate_input(
            catalog_filename=catalog, output_file=prefix, moment_tensors=False
        )

        crustal = polars.read_csv(tmp_path / "out_crustal.csv")
        assert crustal["id"].to_list() == ["ev0", "ev10", "ev11"]
        assert set(crustal["eq_category"]) == {"crustal"}

        combined = polars.read_csv(tmp_path / "out_interface_intraslab.csv")
        assert combined["eq_category"].to_list() == [
            "subduction_interface-subduction_intraslab"
        ]
        assert (tmp_path / "out_crustal_mantle.csv").exists()
        assert (tmp_path / "out_mantle_intraslab.csv").exists()
        # no files for categories that are not in the catalog
        assert len(list(tmp_path.glob("out_*.csv"))) == len(set(CATEGORIES))

    def test_moment_tensors(self, catalog, tmp_path):
        prefix = str(tmp_path / "out")
        filenames = separate_csv.separate_input(
            catalog_filename=catalog, output_file=prefix, moment_tensors=True
        )

        assert len(filenames) == 2 * len(set(CATEGORIES))
        crustal_mt = polars.read_csv(tmp_path / "out_crustal_mt.csv")
        # ev0, ev10 have a strike; ev11 does not
        assert crustal_mt["id"].to_list() == ["ev0", "ev10"]
        # category without a mechanism gives an empty (header only) file
        unknown_mt = polars.read_csv(tmp_path / "out_unknown_mt.csv")
        assert unknown_mt.height == 0
        assert unknown_mt.columns == ["id", "eq_category", "S1"]


class TestCli:
    def test_cli(self, catalog, tmp_path, monkeypatch):
        prefix = str(tmp_path / "cli")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify_separate_csv",
                "--catalog",
                str(catalog),
                "--output-file",
                prefix,
            ],
        )
        separate_csv.cli()

        assert (tmp_path / "cli_crustal.csv").exists()
        assert not (tmp_path / "cli_crustal_mt.csv").exists()

    def test_cli_moment_tensors(self, catalog, tmp_path, monkeypatch):
        prefix = str(tmp_path / "cli")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify_separate_csv",
                "--catalog",
                str(catalog),
                "--output-file",
                prefix,
                "--moment-tensors",
            ],
        )
        separate_csv.cli()

        assert (tmp_path / "cli_crustal_mt.csv").exists()

    def test_cli_missing_arguments(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["classify_separate_csv"])
        with pytest.raises(SystemExit):
            separate_csv.cli()
