"""Input and output utility finctions."""

# stdlib
import pathlib


def read_config(filename: pathlib.Path) -> dict:
    """Read configuration file.

    Args:
      filename: Name of configuration file.
    """
    import yaml

    config = None
    with open(filename, "r", encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
    return config
