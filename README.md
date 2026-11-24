# Introduction

This project contains code for classifying (or separating) an earthquake catalog into various tectonic regions: crustal, interface, intraslab, mantle, and outer-rise. Utility scripts for analyzing the classified results are also included. Inquires can be sent to Kirstie Haynie (<khaynie@usgs.gov>).

Version 2.0.0 is the latest version of this code and was used in the U.S. Geological Survey (USGS) U.S. Pacific territories of Guam, the Commonwealth of the Northern Mariana Islands, and American Samoa National Seismic Hazard Models (<https://code.usgs.gov/ghsc/neic/utilities/neic-catalog-segregation/-/tree/2.0.0?ref_type=tags>). To cite the code, refer to [How to Cite](#how-to-cite).

Version 1.0.0 of this code is associated with the USGS Puerto Rico-U.S. Virgin Islands National Seismic Hazard Model (see [Haynie et al., 2025, SRL](https://pubs.geoscienceworld.org/ssa/srl/article/97/2A/1191/661894/Subduction-Zone-Earthquake-Catalog-Separation-Tool)) and is available at <https://code.usgs.gov/ghsc/neic/utilities/neic-catalog-segregation/-/tree/1.0.0?ref_type=tags>.

## How it works

For each earthquake, the classifier looks up the Slab2 surface (depth, strike, dip, thickness, and depth uncertainty) and the CRUST1.0 Moho depth at the event's location. It then computes several independent plausibility scores using smooth ramp functions defined by parameters in a config file. When two Slab2 regions are given, the region whose slab depth is closest to the event's depth is used.

- Probability of being on the interface is the product of four terms: depth relative to the slab surface (shallow and deep ramps, widened by the local Slab2 depth uncertainty), depth relative to the deep limit of the seismogenic zone (a per-region value with a region-specific reference rake), and the Kagan angle between the earthquake's first nodal plane (`S1`/`D1`/`R1`) and the slab's strike/dip with that reference rake. The Kagan angle measures how closely the fault plane orientation matches the subduction interface. If the event has no moment tensor, the Kagan term is not applied.
- Probability of being crustal is the product of depth relative to the Moho and depth relative to the top of the slab surface.
- Probability of being intraslab depends on depth relative to the bottom of the slab (interface depth plus slab thickness). Above the seismogenic zone limit (plus the config tolerance value), it also depends on the event being below the interface. Below that limit, it depends on depth relative to the top of the slab.
- Probability of being mantle depends on whether the event is above or below the top of the slab surface (plus a tolerance). If above, the probability is the product of depth below the Moho and depth relative to the top of the slab surface. If below, the probability depends on depth below the bottom of the slab.
- Probability of being outer-rise is only computed for events outside the slab region when `--nshm` is used. The event must be within `outerrise_maxdist` of the slab, and the slab depth at the nearest slab point must be less than `outerrise_slab_maxdepth`. The probability is based on depth relative to the slab thickness near the trench. Other events outside the slab region are classified as crustal or mantle based on the Moho depth.

The interface, crustal, intraslab, outer-rise, and mantle plausibility scores are normalized to sum to 100 percent and are written to the output as `p_interface`, `p_crustal`, `p_intraslab`, `p_outerrise`, and `p_mantle`. The earthquake is assigned to the category with the highest score. If the second-highest score is not zero and the pair is one of a defined combinations, the label combines both categories as `primary-secondary` (e.g., `crustal-mantle`, `subduction_interface-subduction_intraslab`, `mantle-crustal`). If all five probabilities are equal, the category is `unknown`.

A quality grade (A-D) is assigned to each classification. For crustal, interface, and intraslab events, the grade is based on the plausibility score of the assigned category. With a focal mechanism and a valid slab strike, 80% or more is given a quality grade of A, 60-80% is B, 40-60% is C, and below 40% is D. Without these, the grades shift down one level (80% or more is B, 60-80% is C, below 60% is D). Combined categories, outer-rise events, and `unknown` events are always graded D.

The combined categories (e.g., `crustal-mantle`) can be re-mapped to a single category (e.g., `crustal`) using the `rename` section of the config file. Quality is calculated before renaming.

Vertical, tilted, or overturned slab regions (Manila, Kermadec, Izu-Bonin, and Solomon Islands) are handled separately using the published Slab2 "sup" grids and a nearest-neighbor lookup, since the standard Slab2 grids do not have valid depth values in these regions. Events farther than 250 km from the nearest point in the "sup" grid are treated as outside the slab.

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

This code was designed to work with USGS Slab2 subduction zones (Hayes et. al., (2018)).

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
usage: classify_catalog [-h] [--slab2-regions SLAB2_REGIONS] [--catalog CATALOG_FILENAME] [--output OUTPUT] [--config CONFIG_FILENAME] [--nshm] [--num-processes NUM_PROCESSES]

    Classify an Earthquake Catalog within a Slab2 Region. Use the 3 letter abbreviations below for the Slab2 region(s) the earthquake catalog covers.

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
  --config CONFIG_FILENAME  Path and file name for the configuration file. If not provided, the default configuration is used.
  --num-processes NUM_PROCESSES
                        Number of processes to use in classification.
  --nshm                Classify earthquakes outside of the specified Slab2 region.
```

### Examples

To classify a Slab2 subduction region's earthquake catalog, open your terminal and run the following command (run within the home directory (neic-catalog-separation)):

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

The script reads and writes the catalog and classification (`slice_catalog.csv` and `slice_classified.csv`) in a `tmp` directory within the directory where it is run, so `--plot` without `--create-catalog` and `--classify` plots the results of the previous run. Use `--backend matplotlib` (with `--filename` set to an image file name, e.g., `slice.png`) for a static figure instead of HTML.

A second example script, [`plot_classification_3d.py`](examples/plot_classification_3d.py), renders the classified output of `classify_catalog` (e.g., the columns `longitude`, `latitude`, `depth`, and `eq_category`) as an interactive 3D scatter plot in HTML format. It can optionally overlay the Slab2 depth surface for a region (`--slab2-region`) in the 3D scene, add a 2D map with focal mechanisms plotted (`--focal-mechanisms`) plotting each event's nodal-plane solution from the `S1`/`D1`/`R1` columns.

```bash
plot_classification_3d.py --catalog path/to/separated/csv --filename output_filename.html --slab2-region SLAB_REGION --focal-mechanisms
```

Run `plot_classification_3d.py --help` for the full list of options.

## Helpful utilities

### `classify_fetch_mt`

Fetch moment tensors from ANSS ComCat for use in the classification process. The input catalog requires an `id` column with the ComCat event ids. The output is a copy of the catalog with the moment tensor components `mrr`, `mtt`, `mpp`, `mrt`, `mrp`, and `mtp` (any existing columns with these names are replaced). The components are `NaN` for events without a moment tensor or when the request to ComCat fails (a warning is logged). One request is made per second to stay within the ComCat rate limit.

```bash
usage: classify_fetch_mt [-h] --catalog CATALOG --output OUTPUT

Query ComCat for moment tensors for event ID's provided in a catalog CSV file


options:
  -h, --help         show this help message and exit
  --catalog CATALOG  Name of input earthquake catalog file containing earthquake ids.
  --output OUTPUT    Name of output file.
```

### `classify_separate_csv`

Separate the output comma-separated values (CSV) file from `classify_catalog` into separate files based on classification.

```bash
usage: classify_separate_csv [-h] --catalog CATALOG --output-file OUTPUT_FILE [--moment-tensors]

    Separate the output from running classify_catalog into files based on classification.


options:
  -h, --help            show this help message and exit
  --catalog CATALOG     Name of catalog file to separate into classified sub-files (crustal, interface, intraslab, mantle, outerrise, and combined categories)
  --output-file OUTPUT_FILE
                        Name to prepend to output files
  --moment-tensors      Separate files for data that moment tensor information
```

A file named `OUTPUT_FILE_<category>.csv` is written for each category in the catalog. If specifying `--moment-tensors`, an additional `OUTPUT_FILE_<category>_mt.csv` file is written for each category for events that have a focal mechanism.

### `classify_redistribute_unknowns`

To remove `unknown` and mantle categories from a classified catalog, run `classify_redistribute_unknowns` on the output of `classify_catalog` to re-distribute any mantle plausibility scores among the crustal, interface, and intraslab scores:

```bash
classify_redistribute_unknowns --input CATALOG --output OUTPUT_CATALOG_NAME
```

This script writes a new CSV file and prints how many events were changed. It does the following:

- `unknown` events shallower than 100 km have `p_mantle` redistributed proportionally into `p_crustal`, `p_interface`, and `p_intraslab`. If those three are all zero (mantle is 100 percent), the event becomes 100 percent crustal.
- `unknown` events at or below 100 km are set to 100 percent `p_intraslab`, with all other probabilities set to zero.
- Any other event with a nonzero `p_mantle` also has it redistributed proportionally into the same three probabilities. If those three are all zero, it is split equally.
- `eq_category` is reset to the category with the largest of the three probabilities for every event that changed (including events with a mantle category, such as `mantle` or `crustal-mantle`). If probabilities are tied, the order of preference is crustal, interface, intraslab. The `quality` column is not recomputed.

### `classify_calculate_bvalue`

Calculate the b-value of the earthquake catalog using the method of Wiemer and Wyss (2000). The catalog requires a column `magnitude` (or `mag`, as in the output of `classify_catalog`).

```bash
usage: classify_calculate_bvalue [-h] --catalog CATALOG --completeness-magnitude COMPLETENESS_MAGNITUDE [--bin-size BIN_SIZE] [--plot]

Calculate b-value for a given catalog

options:
  -h, --help            show this help message and exit
  --catalog CATALOG     Name of earthquake catalog file with a column 'magnitude'
  --completeness-magnitude COMPLETENESS_MAGNITUDE
                        Starting/estimated Mc (magnitude of completion)
  --bin-size BIN_SIZE   Earthquake bin size given prevision of the magnitudes in the catalog, usually 0.1
  --plot                Generate plot showing fit of b_value and magnitude of completeness.
```

```bash
classify_calculate_bvalue --catalog=CATALOG_FILENAME --completeness-magnitude=4.5 --bin-size=0.1
```

The results (`mc`, `b_value`, `a_value`, `std_dev`, `residual`, `error_sb`, and the goodness-of-fit `gof`) are printed and written to `CATALOG_FILENAME_bvalue.json`. `--plot` saves the figure as `CATALOG_FILENAME_bvalue.pdf`.

# How to cite

If you use this code or find it useful for your own purposes, please use the following citation:

Haynie, K. L. and Aagaard, B. T., 2026, Subduction zone earthquake catalog separation code, version 2.0.0: U.S. Geological Survey software release, [https://doi.org/10.5066/P132QMF7](https://doi.org/10.5066/P132QMF7)

# Links/References

Disclaimer: [DISCLAIMER.md](./DISCLAIMER.md)

License: [LICENSE.md](./LICENSE.md)

Haynie, K. L., 2024, Subduction zone earthquake catalog separation code, version 1.0.0: U.S. Geological Survey software release, [https://doi.org/10.5066/P13E7CAY](https://doi.org/10.5066/P13E7CAY)

Gavin P. Hayes, Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M., 2018, Slab2, a comprehensive subduction zone geometry model. Science 362, 58-61. [https://doi.org/10.1126/science.aat4723](https://doi.org/10.1126/science.aat4723).
