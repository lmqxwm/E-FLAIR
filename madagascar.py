#!/usr/bin/env python3
"""Run the FLAIR++ Madagascar main results (thin wrapper around madagascar.R)."""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call(["Rscript", os.path.join(here, "madagascar.R")] + sys.argv[1:]))
