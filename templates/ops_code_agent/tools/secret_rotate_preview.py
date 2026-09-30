#!/usr/bin/env python3
"""Run the shared-world entrypoint under the secret_rotate_preview tool name."""

from __future__ import annotations

import sys

from config_apply_preview import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
