import sys

import polars
import pytest

from neic_catalog_separation.utils import redistribute_unknowns

PROB_COLS = ["p_crustal", "p_interface", "p_intraslab", "p_mantle", "p_outerrise"]


def make_catalog(rows: list) -> polars.DataFrame:
    """Create classified catalog from tuples of
    (depth, eq_category, p_crustal, p_interface, p_intraslab, p_mantle, p_outerrise).
    """
    columns = ["depth", "eq_category"] + PROB_COLS
    data = {name: [row[i] for row in rows] for i, name in enumerate(columns)}
    data["id"] = [f"ev{i}" for i in range(len(rows))]
    data["quality"] = ["D"] * len(rows)
    return polars.DataFrame(data).with_columns(
        polars.col(PROB_COLS).cast(polars.Float64)
    )


def redistribute(rows: list) -> polars.DataFrame:
    return redistribute_unknowns.redistribute_unknowns(make_catalog(rows))


class TestUnknownEvents:
    def test_shallow_proportional(self):
        """Shallow unknown: p_mantle is split in proportion to the other probabilities."""
        out = redistribute([(30, "unknown", 50, 25, 5, 20, 0)])
        row = out.row(0, named=True)

        # total of the three targets is 80; mantle 20 is distributed in proportion
        assert row["p_crustal"] == pytest.approx(50 + 20 * 50 / 80)
        assert row["p_interface"] == pytest.approx(25 + 20 * 25 / 80)
        assert row["p_intraslab"] == pytest.approx(5 + 20 * 5 / 80)
        assert row["p_mantle"] == 0.0
        assert row["eq_category"] == "crustal"

    def test_shallow_all_mantle_is_crustal(self):
        out = redistribute([(30, "unknown", 0, 0, 0, 100, 0)])
        row = out.row(0, named=True)

        assert row["p_crustal"] == 100.0
        assert row["p_interface"] == 0.0
        assert row["p_intraslab"] == 0.0
        assert row["p_mantle"] == 0.0
        assert row["eq_category"] == "crustal"

    def test_shallow_category_follows_largest(self):
        out = redistribute([(30, "unknown", 10, 60, 10, 20, 0)])
        assert out["eq_category"].to_list() == ["subduction_interface"]

        out = redistribute([(30, "unknown", 10, 10, 60, 20, 0)])
        assert out["eq_category"].to_list() == ["subduction_intraslab"]

    @pytest.mark.parametrize("depth", [100.0, 250.0])
    def test_deep_is_intraslab(self, depth):
        """Unknown at or below the depth threshold is forced to 100% intraslab."""
        out = redistribute([(depth, "unknown", 20, 20, 20, 20, 20)])
        row = out.row(0, named=True)

        assert row["p_intraslab"] == 100.0
        for col in ["p_crustal", "p_interface", "p_mantle", "p_outerrise"]:
            assert row[col] == 0.0
        assert row["eq_category"] == "subduction_intraslab"

    def test_depth_threshold(self):
        out = redistribute(
            [
                (99.9, "unknown", 0, 0, 0, 100, 0),
                (100.0, "unknown", 0, 0, 0, 100, 0),
            ]
        )
        assert out["eq_category"].to_list() == ["crustal", "subduction_intraslab"]


class TestOtherEventsWithMantle:
    def test_category_is_reset(self):
        """Events with p_mantle > 0 get a new category without 'mantle'."""
        out = redistribute(
            [
                (30, "crustal-mantle", 60, 10, 10, 20, 0),
                (30, "mantle-subduction_intraslab", 10, 10, 50, 30, 0),
                (30, "subduction_interface-mantle", 10, 50, 10, 30, 0),
                (30, "mantle", 20, 30, 10, 40, 0),
            ]
        )

        assert out["eq_category"].to_list() == [
            "crustal",
            "subduction_intraslab",
            "subduction_interface",
            "subduction_interface",
        ]
        assert out["p_mantle"].to_list() == [0.0] * 4
        assert not any("mantle" in category for category in out["eq_category"])

    def test_proportional(self):
        out = redistribute([(30, "crustal-mantle", 60, 10, 10, 20, 0)])
        row = out.row(0, named=True)

        assert row["p_crustal"] == pytest.approx(60 + 20 * 60 / 80)
        assert row["p_interface"] == pytest.approx(10 + 20 * 10 / 80)
        assert row["p_intraslab"] == pytest.approx(10 + 20 * 10 / 80)

    def test_equal_split_when_targets_zero(self):
        out = redistribute([(30, "mantle", 0, 0, 0, 90, 10)])
        row = out.row(0, named=True)

        for col in ["p_crustal", "p_interface", "p_intraslab"]:
            assert row[col] == pytest.approx(30.0)
        assert row["p_outerrise"] == 10.0
        # tie: crustal is chosen before interface and intraslab
        assert row["eq_category"] == "crustal"

    def test_no_depth_threshold(self):
        """Known events with mantle probability are not forced to intraslab at depth."""
        out = redistribute([(300, "crustal-mantle", 60, 10, 10, 20, 0)])
        assert out["eq_category"].to_list() == ["crustal"]


class TestUnchangedEvents:
    def test_no_mantle_unchanged(self):
        rows = [
            (10, "crustal", 100, 0, 0, 0, 0),
            (150, "subduction_intraslab", 0, 10, 90, 0, 0),
            (5, "subduction_interface-subduction_intraslab", 0, 55, 45, 0, 0),
        ]
        before = make_catalog(rows)
        after = redistribute_unknowns.redistribute_unknowns(before)

        assert after.equals(before)

    def test_columns_and_order_preserved(self):
        before = make_catalog(
            [
                (10, "crustal", 100, 0, 0, 0, 0),
                (30, "mantle", 20, 30, 10, 40, 0),
            ]
        )
        after = redistribute_unknowns.redistribute_unknowns(before)

        assert after.columns == before.columns  # no temporary columns left
        assert after["id"].to_list() == before["id"].to_list()
        assert after["quality"].to_list() == before["quality"].to_list()

    def test_probabilities_sum_unchanged(self):
        rows = [
            (30, "unknown", 50, 25, 5, 20, 0),
            (30, "mantle", 20, 30, 10, 30, 10),
            (200, "unknown", 20, 20, 20, 20, 20),
            (30, "crustal-mantle", 0, 0, 0, 100, 0),
        ]
        out = redistribute(rows)

        totals = out.select(PROB_COLS).sum_horizontal().to_list()
        assert totals == pytest.approx([100.0] * len(rows))


class TestMain:
    def run(self, monkeypatch, input_filename, output_filename):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify_redistribute_unknowns",
                "--input",
                str(input_filename),
                "--output",
                str(output_filename),
            ],
        )
        redistribute_unknowns.main()

    def test_main(self, tmp_path, monkeypatch, capsys):
        catalog = make_catalog(
            [
                (30, "unknown", 50, 25, 5, 20, 0),
                (200, "unknown", 20, 20, 20, 20, 20),
                (30, "mantle", 20, 30, 10, 40, 0),
                (10, "crustal", 100, 0, 0, 0, 0),
            ]
        )
        input_filename = tmp_path / "in.csv"
        output_filename = tmp_path / "out.csv"
        catalog.write_csv(input_filename)

        self.run(monkeypatch, input_filename, output_filename)

        out = polars.read_csv(output_filename)
        assert out["eq_category"].to_list() == [
            "crustal",
            "subduction_intraslab",
            "subduction_interface",
            "crustal",
        ]
        assert out["p_mantle"].to_list() == [0.0] * 4
        assert out.columns == catalog.columns

        message = capsys.readouterr().out
        assert "4 rows" in message
        assert "1 shallow" in message
        assert "1 deep" in message
        assert "1 other" in message

    def test_main_missing_arguments(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["classify_redistribute_unknowns"])
        with pytest.raises(SystemExit):
            redistribute_unknowns.main()
