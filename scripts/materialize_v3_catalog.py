#!/usr/bin/env python3
"""Materialize and verify the optional plain-JSON V3 catalog inspection copy."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/v3_public_observation_template_catalog.json.gz"
DEFAULT_OUTPUT = ROOT / "scripts/v3_public_observation_template_catalog.json"
EXPECTED_SHA256 = "a872ef8fdbd7df02512a2e4448a0c3f80cccb84d2902f5c25c9c4a0c46162578"
EXPECTED_BYTES = 379_759_283


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the compressed resource without retaining a plain copy",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".v3-catalog-", suffix=".json", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with gzip.open(SOURCE, "rb") as source:
            shutil.copyfileobj(source, temporary, length=1024 * 1024)
    try:
        observed_bytes = temporary_path.stat().st_size
        observed_sha256 = digest(temporary_path)
        if observed_bytes != EXPECTED_BYTES or observed_sha256 != EXPECTED_SHA256:
            raise SystemExit(
                "catalog integrity check failed: "
                f"bytes={observed_bytes} sha256={observed_sha256}"
            )
        if args.check_only:
            print(
                f"catalog verified: bytes={observed_bytes} sha256={observed_sha256}"
            )
        else:
            temporary_path.replace(output)
            print(f"catalog materialized: {output}")
    finally:
        temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
