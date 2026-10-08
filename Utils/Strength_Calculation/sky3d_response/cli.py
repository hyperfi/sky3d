"""Command-line interface for Sky3D response analysis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .io import parse_sky3d_input
from .pipeline import run_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Fourier.py",
        description="Reproducible linear-response analysis for Sky3D *.res files",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="run a configured analysis")
    analyze.add_argument("--config", required=True, type=Path, help="TOML configuration")
    analyze.add_argument(
        "--output",
        type=Path,
        help="override [output].directory (useful for validation runs)",
    )
    analyze.add_argument(
        "--run-directory",
        type=Path,
        help="override [run].directory while preserving relative input/channel paths",
    )
    inspect = subparsers.add_parser("inspect", help="print parsed Sky3D namelist metadata")
    inspect.add_argument("--input", required=True, type=Path, help="Sky3D for005 input")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "inspect":
            print(json.dumps(parse_sky3d_input(arguments.input), indent=2, sort_keys=True))
            return 0
        config = load_config(arguments.config)
        if arguments.run_directory is not None:
            new_run = arguments.run_directory.resolve()
            old_run = config.run_directory
            input_relative = config.input_file.relative_to(old_run)
            channels = tuple(
                replace(channel, file=new_run / channel.file.relative_to(old_run))
                for channel in config.channels
            )
            config = replace(
                config,
                run_directory=new_run,
                input_file=new_run / input_relative,
                channels=channels,
            )
        if arguments.output is not None:
            config = replace(config, output_directory=arguments.output.resolve())
        result = run_analysis(config)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Analyzed {len(config.channels)} channel(s) with {len(config.windows)} window(s).")
    print(f"Output directory: {config.output_directory}")
    for artifact in result.artifacts:
        print(f"  {artifact.name}")
    return 0
