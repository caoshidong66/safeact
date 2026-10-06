#!/usr/bin/env python3
"""Run one V2/V3 case through the interactive tool interface."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent
if '--snapshot' not in sys.argv:
    sys.argv += ['--snapshot', str(root)]
runpy.run_path(str(root / 'scripts/run_english_interactive_rollout.py'), run_name='__main__')
