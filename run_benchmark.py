#!/usr/bin/env python3
"""Run SafeActBench from this package, regardless of the current directory."""
from pathlib import Path
import os
import runpy
import sys

if sys.version_info < (3, 11):
    raise SystemExit('This distribution requires Python 3.11 or newer.')

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root / 'scripts'))
runpy.run_path(str(root / 'scripts/run_safeact_agent_batch.py'), run_name='__main__')
