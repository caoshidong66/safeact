#!/usr/bin/env python3
"""Verify the immutable files shipped with this SafeActBench distribution."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    expected = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    errors = []
    for relative, wanted in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            errors.append({'path': relative, 'error': 'invalid path'})
            continue
        if not path.is_file():
            errors.append({'path': relative, 'error': 'missing file'})
            continue
        with path.open('rb') as handle:
            observed = hashlib.file_digest(handle, 'sha256').hexdigest()
        if observed != wanted:
            errors.append({'path': relative, 'error': 'checksum mismatch'})
    result = {'passed': not errors, 'files_checked': len(expected), 'errors': errors}
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
