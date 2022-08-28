#!/usr/bin/env python
import pkg_resources

__version__ = pkg_resources.get_distribution("ktplotspy").version

DEFAULT_SEP = ">@<"
DEFAULT_SPEC_PAT = "/|:|\\?|\\*|\\+|\\|\\(|\\)|\\/"
