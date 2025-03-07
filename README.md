# Introduction
This project contains code for classifying (or separating) an earthquake catalog into three tectonic regions: upper plate, interface, and lower plate. Code for calculating b-value for the resulting catalog(s) is also provided. Please send any inquires to Kirstie Haynie <khaynie@usgs.gov>.

# Installation and Dependencies
This software uses python version 3.10. The python environment can be installed via [Poetry](https://python-poetry.org/) using:

    - poetry install

# How to use
This code was designed to specifically work with U.S. Geological Survey (USGS) Slab2 subduction zones and Slab2 data from Hayes et. al., (2018). Thus, the code requires all input files to be in the Slab2 input file format. Included under catalog_sep/Input/Slab2Catalogs are the Slab2 input files associated with the published Slab2 code and models from Hayes et al., (2018). New databases have been added to the [Slab2 GitLab repository](https://code.usgs.gov/ghsc/esi/slab2) since 2018 and input files created from newer databases can easily be made by downloading the Slab2 repository and following along with the necessary documentation. One can also use their own earthquake catalog provided that it covers a Slab2 subduction zone. The user is referred to the Slab2 GitLab repository and documentation for details on how to convert an earthquake catalog into the Slab2 input format. Please note that this is a heuristic code, such that certain variables were set according to testing with a global catalog. It is encouraged to update values to best reflect the subduction zone(s) for your use case. Such values can be updated in [subduction.conf](catalog_sep/Input/config/subduction.conf). The flex zone (variable FLEX in the code) used as a buffer around the subduction interface region can be adjusted in [classify_catalog.py](catalog_sep/src/classify_catalog.py) by setting the constant FLEX to your desired value in kilometers.

To classify a Slab2 subduction region's earthquake catalog into upper plate/crustal, interface, or lower plate/intraslab, open your terminal and run the following command (run within the home directory (neic-catalog-separation)):

    - python catalog_sep/src/classify_catalog.py $Slab2Region $PathToInputCatalog

If your earthquake catalog files spans two subduction zones/slabs run

    - python catalog_sep/src/classify_catalog.py $Slab2Region $PathToInputCatalog --second_slab $Slab2Region

The $Slab2Region should be the three-letter abbreviation associated with the Slab2 subduction zone of interest. To see a full list of the Slab2 subduction zones and abbreviations run:

    - python catalog_sep/src/classify_catalog.py -h

Moment tensor information is used during the classification process. To query the USGS earthquake catalog for moment tensors use:

    - python catalog_sep/src/utils/get_mt.py -h

To separate the output Comma-separated Values (CSV) file from classify_catalog.py into three files, one for each separated region (crustal, interface, and intraslab) use:

    - python catalog_sep/src/utils/separate_csv.py -h
      - Note: You may need to update the header names to match your input data.

To calculate the b-value of the earthquake catalog, prepare a text file containing only magnitudes in one column and run:

    - python catalog_sep/src/utils/calculate_bValue.py -h

Arguments required are name of the input file containing magnitudes, a starting/estimated magnitude of completeness (Mc), and a bin size. For example,

    - python catalog_sep/src/utils/calculate_bValue.py MagFile.in 4.5 0.1
  
# How to cite

If you use this code or find it useful for your own purposes, please use the following citation:

Haynie, K. L., 2024, Subduction zone earthquake catalog separation code, version 1.0.0: U.S. Geological Survey software release, [https://doi.org/10.5066/P13E7CAY](https://doi.org/10.5066/P13E7CAY)

# Links/References
Disclaimer: [DISCLAIMER.md](./DISCLAIMER.md)

License: [LICENSE.md](./LICENSE.md)

Gavin P. Hayes, Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M., 2018, Slab2, a comprehensive subduction zone geometry model. Science 362, 58-61. [https://doi.org/10.1126/science.aat4723](https://doi.org/10.1126/science.aat4723)