#!/usr/bin/env python
import toml
from pathlib import Path

from ktplotspy.plot import *

__version__ = toml.load(Path(__file__).parent.parent / "pyproject.toml")["project"]["version"]
