# Introduction

This project contains code for classifying (or separating) an earthquake catalog into three tectonic regions: upper plate, interface, and lower plate.
Code for calculating b-value for the resulting catalog(s) is also provided.
Please send any inquires to Kirstie Haynie <khaynie@usgs.gov>.

Note: the version of the code associated with the USGS Puerto Rice-U.S. Virgin Islands National Seismic Hazard Model (Haynie et al., 2025, SRL, accepted) is avaiable at https://code.usgs.gov/ghsc/neic/utilities/neic-catalog-segregation/-/tree/1.0.0?ref_type=tags. To cite this, see [How to Cite](#how-to-cite).

# Installation and Dependencies

This software uses Python version 3.9 or later.
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
usage: classify_catalog [-h] [--slab2-regions SLAB2_REGIONS] [--catalog CATALOG_FILENAME] [--output OUTPUT] [--config CONFIG_FILE] [--nshm]
                        [--interface-tolerance TOLERANCE] [--num-processes NUM_PROCESSES]

    Classify a Slab2 Region Earthquake Catalog. Use one of the 3 letter abbreviations for the Slab2 region catalog to classify

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
  --nshm                Classify earthquakes outside of the specified Slab2 region.
  --interface-tolerance TOLERANCE
                        Maximum distance from slab interface for earthquakes to be considered on the interface.
  --num-processes NUM_PROCESSES
                        Number of processes to use in classification.
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
To see a full list of the Slab2 subduction zones with abbreviations and command line arguments run

```bash
classify_catalog --help
```

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

Separate the output comma-separated values (CSV) file from `classify_catalog.py` into three files, one for each separated region (crustal, interface, and intraslab) use

```bash
usage: classify_separate_csv [-h] --catalog CATALOG --moment-tensors

    Separate the output from running classify_catalog into files based on classification.
    

options:
  -h, --help         show this help message and exit
  --catalog CATALOG  Name of catalog file to separate into classified sub-files (crustal, interface, slab, mantle)
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

# How to cite

If you use this code or find it useful for your own purposes, please use the following citation:

Haynie, K. L., 2024, Subduction zone earthquake catalog separation code, version 1.0.0: U.S. Geological Survey software release, [https://doi.org/10.5066/P13E7CAY](https://doi.org/10.5066/P13E7CAY)

# Links/References

Disclaimer: [DISCLAIMER.md](./DISCLAIMER.md)

License: [LICENSE.md](./LICENSE.md)

Gavin P. Hayes, Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M., 2018, Slab2, a comprehensive subduction zone geometry model. Science 362, 58-61. [https://doi.org/10.1126/science.aat4723](https://doi.org/10.1126/science.aat4723).
