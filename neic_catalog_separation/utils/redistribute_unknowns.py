import argparse

import polars as pl

# Columns that receive the redistributed mass
TARGETS = ["p_crustal", "p_interface", "p_intraslab"]

# All probability columns (used when forcing deep events to intraslab)
PROB_COLS = ["p_crustal", "p_interface", "p_intraslab", "p_mantle", "p_outerrise"]

CATEGORY_MAP = {
    "p_crustal": "crustal",
    "p_interface": "subduction_interface",
    "p_intraslab": "subduction_intraslab",
}

# Temporary column flagging events whose p_mantle is redistributed
HAS_MANTLE = "_has_mantle"

# Depth threshold (km) separating shallow redistribution from deep intraslab
MAX_DEPTH_KM = 100


def redistribute_unknowns(df: pl.DataFrame) -> pl.DataFrame:
    """Reclassify 'unknown' events, and redistribute any leftover p_mantle.

    Shallow (depth < MAX_DEPTH_KM): redistribute p_mantle into the target
    categories proportionally, unless mantle is 100%, then make crustal 100%.
    Deep (depth >= MAX_DEPTH_KM): force p_intraslab = 100, all other
    probabilities = 0, category = subduction_intraslab.

    Additionally, for any event (regardless of eq_category or depth) that
    still has p_mantle > 0 after the above (e.g. non-"unknown" events that
    simply have a nonzero p_mantle), p_mantle is redistributed into the
    target categories proportionally using the same equation, with no
    is_deep check applied. eq_category is updated via CATEGORY_MAP if the
    largest target probability implies a different category.
    """
    is_unknown = pl.col("eq_category") == "unknown"
    is_shallow = is_unknown & (pl.col("depth") < MAX_DEPTH_KM)
    is_deep = is_unknown & (pl.col("depth") >= MAX_DEPTH_KM)

    # Sum of the three remaining target categories (the redistribution base)
    remaining_total = (
        pl.col("p_crustal") + pl.col("p_interface") + pl.col("p_intraslab")
    )

    def redistribute(col_name: str) -> pl.Expr:
        # new value = ((current / remaining_total) * p_mantle) + current
        proportional = (
            (pl.col(col_name) / remaining_total) * pl.col("p_mantle")
        ) + pl.col(col_name)
        # If remaining total is 0 (i.e., mantle is 100), for shallow unknowns make 100% crustal
        shallow_val = (
            pl.when(remaining_total > 0)
            .then(proportional)
            .otherwise(100.0 if col_name == "p_crustal" else 0.0)
        )
        # Deep unknowns: p_intraslab -> 100, other targets -> 0
        deep_val = 100.0 if col_name == "p_intraslab" else 0.0
        return (
            pl.when(is_shallow)
            .then(shallow_val)
            .when(is_deep)
            .then(pl.lit(deep_val))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

    df = df.with_columns([redistribute(c) for c in TARGETS])

    # p_mantle: 0 for both shallow (redistributed) and deep (forced) unknowns
    df = df.with_columns(
        pl.when(is_shallow | is_deep)
        .then(pl.lit(0.0))
        .otherwise(pl.col("p_mantle"))
        .alias("p_mantle")
    )

    # p_outerrise: 0 for deep unknowns (all non-intraslab categories zeroed)
    df = df.with_columns(
        pl.when(is_deep)
        .then(pl.lit(0.0))
        .otherwise(pl.col("p_outerrise"))
        .alias("p_outerrise")
    )

    # Assign eq_category based on the largest of the three target probabilities.
    # For deep events this naturally resolves to subduction_intraslab since
    # p_intraslab == 100 and the others are 0.
    new_category = (
        pl.when(
            (pl.col("p_crustal") >= pl.col("p_interface"))
            & (pl.col("p_crustal") >= pl.col("p_intraslab"))
        )
        .then(pl.lit(CATEGORY_MAP["p_crustal"]))
        .when(pl.col("p_interface") >= pl.col("p_intraslab"))
        .then(pl.lit(CATEGORY_MAP["p_interface"]))
        .otherwise(pl.lit(CATEGORY_MAP["p_intraslab"]))
    )

    df = df.with_columns(
        pl.when(is_shallow | is_deep)
        .then(new_category)
        .otherwise(pl.col("eq_category"))
        .alias("eq_category")
    )

    # Any remaining events with p_mantle > 0 (e.g. non-"unknown" events that
    # simply carry a nonzero p_mantle) get that mass redistributed into the
    # target categories using the same equation, with no is_deep check.
    # Flag these events in a column now: p_mantle is set to zero below, so the
    # condition cannot be re-evaluated later to update eq_category.
    df = df.with_columns((pl.col("p_mantle") > 0).alias(HAS_MANTLE))
    has_mantle = pl.col(HAS_MANTLE)
    remaining_total = (
        pl.col("p_crustal") + pl.col("p_interface") + pl.col("p_intraslab")
    )

    def redistribute_mantle(col_name: str) -> pl.Expr:
        proportional = (
            (pl.col(col_name) / remaining_total) * pl.col("p_mantle")
        ) + pl.col(col_name)
        equal_split = pl.col(col_name) + pl.col("p_mantle") / len(TARGETS)
        new_val = pl.when(remaining_total > 0).then(proportional).otherwise(equal_split)
        return (
            pl.when(has_mantle)
            .then(new_val)
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

    df = df.with_columns([redistribute_mantle(c) for c in TARGETS])

    df = df.with_columns(
        pl.when(has_mantle)
        .then(pl.lit(0.0))
        .otherwise(pl.col("p_mantle"))
        .alias("p_mantle")
    )

    mantle_new_category = (
        pl.when(
            (pl.col("p_crustal") >= pl.col("p_interface"))
            & (pl.col("p_crustal") >= pl.col("p_intraslab"))
        )
        .then(pl.lit(CATEGORY_MAP["p_crustal"]))
        .when(pl.col("p_interface") >= pl.col("p_intraslab"))
        .then(pl.lit(CATEGORY_MAP["p_interface"]))
        .otherwise(pl.lit(CATEGORY_MAP["p_intraslab"]))
    )

    df = df.with_columns(
        pl.when(has_mantle)
        .then(mantle_new_category)
        .otherwise(pl.col("eq_category"))
        .alias("eq_category")
    )

    return df.drop(HAS_MANTLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reclassify 'unknown' earthquake events: shallow events "
        f"(< {MAX_DEPTH_KM} km) have p_mantle redistributed proportionally into "
        f"p_crustal/p_interface/p_intraslab; deep events (>= {MAX_DEPTH_KM} km) "
        "are forced to p_intraslab = 100."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output CSV file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pl.read_csv(args.input, infer_schema_length=None)

    is_unknown = pl.col("eq_category") == "unknown"
    n_shallow = df.filter(is_unknown & (pl.col("depth") < MAX_DEPTH_KM)).height
    n_deep = df.filter(is_unknown & (pl.col("depth") >= MAX_DEPTH_KM)).height
    n_mantle = df.filter(~is_unknown & (pl.col("p_mantle") > 0)).height

    df = redistribute_unknowns(df)

    df.write_csv(args.output)
    print(
        f"Wrote {args.output} ({df.height} rows, "
        f"{n_shallow} shallow 'unknown' redistributed, "
        f"{n_deep} deep 'unknown' set to intraslab, "
        f"{n_mantle} other events with p_mantle > 0 redistributed)"
    )


if __name__ == "__main__":
    main()
