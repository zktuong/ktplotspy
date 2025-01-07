#!/usr/bin/env python
from importlib.metadata import version

__version__ = version("ktplotspy").split("+")[0]

DEFAULT_SEP = ">@<"
DEFAULT_SPEC_PAT = "/|:|\\?|\\*|\\+|\\(|\\)|\\/|\\[|\\]"
DEFAULT_CELLSIGN_ALPHA = 0.5
DEFAULT_COLUMNS = ["interaction_group", "celltype_group"]
DEFAULT_V5_COL_START = 13
DEFAULT_COL_START = 11
DEFAULT_CLASS_COL = 12
DEFAULT_CPDB_SEP = "|"
# DEFAULT_PAL = cycle(plt.cm.tab20.colors)
