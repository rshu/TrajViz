"""Shared CLI scope for the report, dossier, digest and pipeline entry points."""

from __future__ import annotations

import argparse
from . import config


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def pipeline_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate AWE diagnosis artifacts.")
    parser.add_argument("--agent", choices=config.AGENTS)
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--workers", type=_positive, default=1)
    parser.add_argument("--max-md", type=_nonnegative, default=30)
    return parser.parse_args(argv)


def selected_inputs(args):
    instances = config.list_instances()
    if args.limit is not None:
        instances = instances[: args.limit]
    return ([args.agent] if args.agent is not None else list(config.AGENTS), config.require_instances(instances))
