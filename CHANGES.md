# Changes

## main

- **Changed**
  - Output earthquake location as a meaningful string instead of letters.
  - Read data once (outside loop over earthquakes).
  - Reorganize code layout to facilitate use as a library.
    - Separate source and tests.
    - Load data files using path relative to main driver.
  - Use concurent.futures.ProcessPoolExecutor for parallel processing with single earthquake passed to each worker. This also maintains the same order for the earthquakes in the input catalog and output classification.
  - Switch configuration file from config syntax to YAML (automatic type recognition).
  - Switch from pandas to polars for reading/writing CSV files.
  - Changed catalog header (id_no -> id, lat -> latitude, lon -> longitude).
  - Update tests.
    - Use pytest fixture so data is only read once.
    - Organize tests by module and use classes to separate tests.
- **Added**
  - Add interface tolerance as a command line parameter.
  - Add number of processes as a command line parameter.
  - Add configuration file as optional command line parameter.
  - Add progress bar using tqdm.
