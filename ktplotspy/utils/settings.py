#!/usr/bin/env python
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("ktplotspy").version
except:  # pragma: no cover
    __version__ = "dev"

DEFAULT_SEP = ">@<"
DEFAULT_SPEC_PAT = "/|:|\\?|\\*|\\+|\\|\\(|\\)|\\/"
DEFAULT_CELLSIGN_ALPHA = 0.5
DEFAULT_COLUMNS = ["interaction_group", "celltype_group"]
DEFAULT_V5_COL_START = 13
DEFAULT_COL_START = 11
DEFAULT_CLASS_COL = 12
DEFAULT_CPDB_SEP = "|"
