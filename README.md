# Introduction

This project contains code for classifying (or separating) an earthquake catalog into various tectonic regions: crustal, interface, intraslab, mantle, and outer-rise. If there an equal probability among these categories, the earthquak is assgined an unknown category.
Code for calculating b-value for the resulting catalog(s) is also provided.
Please send any inquires to Kirstie Haynie (<khaynie@usgs.gov>).

Note: the version of the code associated with the USGS Puerto Rico-U.S. Virgin Islands National Seismic Hazard Model (see [Haynie et al., 2025, SRL](https://pubs.geoscienceworld.org/ssa/srl/article/97/2A/1191/661894/Subduction-Zone-Earthquake-Catalog-Separation-Tool)) is avaiable at <https://code.usgs.gov/ghsc/neic/utilities/neic-catalog-segregation/-/tree/1.0.0?ref_type=tags>. To cite the code, refer to [How to Cite](#how-to-cite).

## How it works

For each earthquake, the classifier looks up the Slab2 surface (depth, strike, dip, thickness, and depth uncertainty) and the CRUST1.0 Moho depth at the event's location, then computes several independent probabilities using smooth ramp functions defined by the parameters in the config file:

- Probability of being on the interface, based on depth relative to the slab surface (+/- tolerance), depth relative to the seismogenic zone, and the Kagan angle between the earthquake's focal mechanism and the slab's strike/dip/rake (a measure of how closely the fault plane orientation matches the subduction interface).
- Probability of being crustal, based on depth relative to the Moho and relative to the slab surface.
- Probability of being intraslab, based on depth relative to the interface and relative to the slab thickness.
- Probability of being outer-rise (only computed for events outside the slab region when `--nshm` is used), based on depth relative to the slab thickness near the trench.

These probabilities are normalized to sum to 1 (with any remainder assigned to mantle), and the earthquake is assigned to whichever category has the highest probability. A tie between categories results in an `unknown` classification. A quality grade (A-D) is also assigned to each classification, reflecting how well-supported the category assignment is.

By default, the code produces "sub-mantle" categories (e.g., `crustal_mantle`) for events near the boundary with the mantle. These can be re-mapped to their primary category (e.g. `crustal`) using the `rename` section of the config file.

Vertical, tilted, or overturned slab regions (Manila, Kermadec, Izu-Bonin, and Solomon Islands) are handled separately using the published Slab2 "sup" grids and a nearest-neighbor lookup, since the standard Slab2 grids do not have valid depth values in these regions.

# Installation and Dependencies

This software uses Python version 3.10 or later.
The package can be installed using [Poetry](https://python-poetry.org/)

```bash
poetry install
```

or using pip

```bash
pip install .
```

# How to use

This code was designed to work with U.S. Geological Survey (USGS) Slab2 subduction zones and Slab2 data from Hayes et. al., (2018).

The catalog input file requires the following columns:

- `id` (earthquake id)
- `longitude` (longitude of earthquake in degrees)
- `latitude` (latitude of earthquake in degrees)
- `depth` (depth of earthquake in km)
- `S1` (strike angle of first nodal plane in degrees)
- `D1` (dip angle of first nodal plane in degrees)
- `R1` (rake angle of first nodal plane in degrees)
- `Ppl` (Plunge angle of p axis in degrees)
- `Tpl` (Plunge angle of t axis in degrees)
  Additional columns are ignored.

Please note that this is a heuristic code, such that certain parameters were set according to testing with a global catalog.

You are encouraged to update values to best reflect the subduction zone(s) for your use case.
The parameters can be updated in [config/subduction.yml](neic_catalog_separation/config/subduction.yml).

```bash
usage: classify_catalog [-h] [--slab2-regions SLAB2_REGIONS] [--catalog CATALOG_FILENAME] [--output OUTPUT] [--config CONFIG_FILE] [--nshm] [--num-processes NUM_PROCESSES]

    Classify an Earthquake Catalog within a Slab2 Region. Use the 3 letter abbreviations below for the Slab2 region(s) the eartqhuake catalog covers.

    Slab2 regions include:

        Aleutians               alu
        Calabria                cal
        Central America         cam
        Caribbean               car
        Cascadia                cas
        Cotabato                cot
        Halmahera               hal
        Hellenic                hel
        Himalaya                him
        Hindu Kush              hin
        Izu-Bonin               izu
        Kermadec                ker
        Kuril                   kur
        Makran                  mak
        Manila                  man
        Muertos                 mue
        Pamir                   pam
        New Guinea              png
        Philippines             phi
        Puysegur                puy
        Ryukyu                  ryu
        South America           sam
        Scotia                  sco
        Solomon Islands         sol
        Sulawesi                sul
        Sumatra/Java            sum
        Vanuatu                 van


options:

  -h, --help            show this help message and exit
  --slab2-regions SLAB2_REGIONS
                        Comma separated list of names of Slab2 regions. Currently limited to a maximum of two regions.
  --catalog CATALOG_FILENAME
                        Path and file name for the input catalog to separate (must be CSV file format).
  --output OUTPUT       Path and file name for the output file.
  --config CONFIG_FILE  Path and file name for the configuration file. If not provided, the default configuration is used.
  --num-processes NUM_PROCESSES
                        Number of processes to use in classification.
  --nshm                Classify earthquakes outside of the specified Slab2 region as outer-rise.
```

### Examples

To classify a Slab2 subduction region's earthquake catalog into upper plate/crustal, interface, or lower plate/intraslab, open your terminal and run the following command (run within the home directory (neic-catalog-separation)):

```bash
classify_catalog --slab2-regions=SLAB_REGION --catalog=CATALOG_FILENAME
```

If your earthquake catalog files spans two subduction zones/slabs run

```bash
classify_catalog --slab2-regions=SLAB_REGION1,SLAB_REGION2 --catalog=CATALOG_FILENAME
```

The `SLAB_REGION` should be the three-letter abbreviation associated with the Slab2 subduction zone of interest.

The code classifies sub-mantle categories (e.g., crustal mantle) by default. To re-map these categories to their primary category, uncomment the `rename` section in [config/subduction.yml](neic_catalog_separation/config/subduction.yml) before running `classify_catalog`. The mappings specified here can also be edited based on the user's needs.

To see a full list of the Slab2 subduction zones with abbreviations and command line arguments run

```bash
classify_catalog --help
```

An example script is also provided that will create a test earthquake catalog for very longitude, latitude, and depth along a user supplied cross-section, classify the cross-section, and plot the results. To use this example script, navigate to the `examples` directory and run

```bash
plot_classification_slice.py --help
```

For example, the following will create a catalog for a cross-section along -17&deg; from -172&deg; to -178&deg; in the Kermadec Slab2 region to a depth of 450 km, classify that catalog, then plot the result as a HTML file that can be opened in a web browser

```bash
plot_classification_slice.py --latitude -17 --max-longitude -172 --min-longitude -178 --max-depth 450 --regions ker --create-catalog --classify --plot
```

A second example script, [`plot_classification_3d.py`](examples/plot_classification_3d.py), renders the classified output of `classify_catalog` (e.g., the columns `longitude`, `latitude`, `depth`, and `eq_category`) as an interactive 3D scatter plot in HTML format. It can optionally overlay the Slab2 depth surface for a region (`--slab2-region`) in the 3D scene, add a 2D map panel with coastlines and country borders (`--coastlines`), and/or add a focal mechanism ("beachball") map panel (`--focal-mechanisms`) plotting each event's nodal-plane solution from the `S1`/`D1`/`R1` columns.

```bash
plot_classification_3d.py --catalog path/to/separated/csv --filename output_filname.html --slab2-region SLAB_REGION --coastlines --focal-mechanisms
```

Run `plot_classification_3d.py --help` for the full list of options.

## Helpful utilities

### `clasify_fetch_mt`

Fetch moment tensors from ANSS ComCat for use in the classification process.

```bash
usage: classify_fetch_mt [-h] --catalog CATALOG --output OUTPUT

Query ComCat for moment tensors for event ID's provided in a catalog CSV file


options:
  -h, --help         show this help message and exit
  --catalog CATALOG  Name of input earthquake catalog file containing earthquake ids.
  --output OUTPUT    Name of output file.
```

### `classify_separate_csv`

Separate the output comma-separated values (CSV) file from `classify_catalog.py` into separate files based on classification.

```bash
usage: classify_separate_csv [-h] --catalog CATALOG --moment-tensors

    Separate the output from running classify_catalog into files based on classification.


options:
  -h, --help         show this help message and exit
  --catalog CATALOG  Name of catalog file to separate into classified sub-files (crustal, interface, slab, mantle, outerrise)
  --moment-tensors   Separate files for data that moment tensor information
```

### `clasify_calculate_bvalue`

Calculate the b-value of the earthquake catalog from the earthquake catalog.

```bash
usage: classify_calculate_bvalue [-h] --catalog CATALOG --completeness-magnitude COMPLETENESS_MAGNITUDE [--bin-size BIN_SIZE] [--plot]

Calculate b-value for a given catalog

options:
  -h, --help            show this help message and exit
  --catalog CATALOG     Name of earthquake cata log file with a column 'magnitude'
  --completeness-magnitude COMPLETENESS_MAGNITUDE
                        Starting/estimated Mc (magnitude of completion)
  --bin-size BIN_SIZE   Earthquake bin size given prevision of the magnitudes in the catalog, usually 0.1
  --plot                Generate plot showing fit of b_value and magnitude of completeness.
```

```bash
classify_calculate_bvalue --catalog=CATALOG_FILENAME --completeness-magnitude=4.5 --bin-size=0.1
```

# Project structure

- [`neic_catalog_separation/classify_catalog.py`](neic_catalog_separation/classify_catalog.py) &mdash; `classify_catalog` CLI entry point. Loads the catalog and Slab2/Moho/seismogenic-depth data, then classifies each earthquake (serially or in parallel via `--num-processes`) and writes the results to a CSV.
- [`neic_catalog_separation/classify_earthquake.py`](neic_catalog_separation/classify_earthquake.py) &mdash; Core per-earthquake classification algorithm described in [How it works](#how-it-works).
- [`neic_catalog_separation/funcs.py`](neic_catalog_separation/funcs.py) &mdash; Supporting math/geometry functions: probability ramp functions, Kagan angle calculation, category selection, and quality grading.
- [`neic_catalog_separation/data_sources.py`](neic_catalog_separation/data_sources.py) &mdash; Loaders for the Slab2 grids (depth, strike, dip, thickness, uncertainty, including nearest-neighbor lookups for overturned slabs), CRUST1.0 Moho depth grids, and per-region seismogenic depth/rake values.
- [`neic_catalog_separation/eqtype.py`](neic_catalog_separation/eqtype.py) &mdash; `EqType` enum for the classification categories and the category rename mechanism.
- [`neic_catalog_separation/io.py`](neic_catalog_separation/io.py) &mdash; Configuration file reader.
- [`neic_catalog_separation/config/subduction.yml`](neic_catalog_separation/config/subduction.yml) &mdash; Tunable heuristic parameters used by the ramp functions, plus the optional category `rename` mappings.
- [`neic_catalog_separation/data/`](neic_catalog_separation/data) &mdash; Bundled Slab2 and CRUST1.0 grid data used by the data source loaders.
- [`neic_catalog_separation/utils/`](neic_catalog_separation/utils) &mdash; The `classify_fetch_mt`, `classify_separate_csv`, and `classify_calculate_bvalue` utilities described above.
- [`examples/plot_classification_slice.py`](examples/plot_classification_slice.py) &mdash; Generates, classifies, and plots a synthetic earthquake catalog along a user-specified cross-section, useful for visually checking classification boundaries for a given slab region.
- [`examples/plot_classification_3d.py`](examples/plot_classification_3d.py) &mdash; Renders a classified catalog's `longitude`, `latitude`, and `depth` as an interactive 3D scatter plot, colored by `eq_category`.
- [`examples/focal_mechanism.py`](examples/focal_mechanism.py) &mdash; Standalone, dependency-free (matplotlib only) rendering of focal mechanism ("beachball") diagrams from strike/dip/rake, used by `plot_classification_3d.py`'s `--focal-mechanisms` option.
- `tests/` &mdash; Pytest coverage for the probability/Kagan-angle math, Slab2/Moho/seismogenic-depth data lookups, and an end-to-end `classify_catalog` smoke test.

# How to cite

If you use this code or find it useful for your own purposes, please use the following citation:

Haynie, K. L., 2024, Subduction zone earthquake catalog separation code, version 1.0.0: U.S. Geological Survey software release, [https://doi.org/10.5066/P13E7CAY](https://doi.org/10.5066/P13E7CAY)

# Links/References

Disclaimer: [DISCLAIMER.md](./DISCLAIMER.md)

License: [LICENSE.md](./LICENSE.md)

Gavin P. Hayes, Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M., 2018, Slab2, a comprehensive subduction zone geometry model. Science 362, 58-61. [https://doi.org/10.1126/science.aat4723](https://doi.org/10.1126/science.aat4723).
