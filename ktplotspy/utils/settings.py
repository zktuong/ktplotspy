#!/usr/bin/env python
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("ktplotspy").version
except:  # pragma: no cover
    __version__ = "dev"

DEFAULT_SEP = ">@<"
DEFAULT_SPEC_PAT = "/|:|\\?|\\*|\\+|\\|\\(|\\)|\\/"
