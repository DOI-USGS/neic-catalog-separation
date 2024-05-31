# Introduction
This project contains code for classifying (or separating) an earthquake catalog into three tectonic regions: upper plate, interface, and lower plate. Code for calculating b-value for the resulting catalog(s) is also provided. Please send any inquires to Kirstie Haynie <khaynie@usgs.gov>.

# Installation and Dependencies
This software uses python version 3.6+. The dependencies are:
  - numpy Used for array calculations
  - matplotlib Used for making figures
  - pandas Used for dataframe calculations

Use the included install.sh shell script to create a separate conda environment
with pytest and ipython installed for debugging and test script development.

On a Linux or OSX system, open a Terminal window and create the conda (Anaconda) environment by typing the following command:

    - bash install.sh
    - conda activate catalog_sep

 The conda environment should now be activated and code from this repository can be run.

# How to use
This code was designed to specifically work with USGS Slab2 subduction zones and Slab2 data. Thus, the code requires all input files to be in the Slab2 input file format. Included under catalog_sep/Input/Slab2Catalogs are the Slab2 input files associated with the published Slab2 code and models from Hayes et al., 2018. New databases have been added to the [Slab2 GitLab repository] (https://code.usgs.gov/ghsc/esi/slab2) since 2018 and input files created from newer databases can easily be made by downloading the Slab2 repository and following along with the necessary documentation. One can also use their own earthquake catalog provided that it covers a Slab2 subduction zone. The user is referred to the Slab2 GitLab repository and documentation for details on doing so.

To classify a Slab2 subduction region's earthquake catalog into upper plate/crustal, interface, or lower plate/intraslab, open your terminal and run the following command (run within the home directory (neic-catalog-segregation)):

    - python catalog_sep/src/classify_catalog.py $Slab2Region

If you earthquake catalog files spans two subduction zones/slabs run

    - python catalog_sep/src/classify_catalog.py $Slab2Region --second_slab $Slab2Region

The $Slab2Region should be the three-letter abbreviation associated with the Slab2 subduction zone of interest. To see a full list of the Slab2 subduction zones and abbreviations run:

    - python catalog_sep/src/classify_catalog.py -h

Moment tensor information is used during the classification process. To query the USGS earthquake catalog for moment tensors use:

    - python catalog_sep/src/utils/get_mt.py -h

To separate the output CSV from classify_catalog.py into three files, one for each separated region (crustal, interface, and intraslab) use:

    - python catalog_sep/src/utils/separate_csv.py -h
      - Note: You may need to update the header names to match your input data.

To calculate the b-value of the earthquake catalog, prepare an ASCII file containing only magnitudes in one column and run:

    - python catalog_sep/src/utils/calculate_bValue.py -h

Arguments required are name of the input file containing magnitudes, a starting/estimated magnitude of completeness (Mc), and a bin size. For example,

    - python catalog_sep/src/utils/calculate_bValue.py MagFile.in 4.5 0.1