"""Module entrypoint for `python -m vld`."""

from __future__ import annotations

import sys

from vld.cli import main

if __name__ == "__main__":
    sys.exit(main())
