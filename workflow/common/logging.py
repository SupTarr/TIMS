#!/usr/bin/env python3
"""
Logging setup for all workflow scripts.
"""

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure standard root logger format for all utils."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
