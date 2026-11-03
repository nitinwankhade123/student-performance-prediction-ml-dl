"""Explicit command-line entry point for dataset preparation tasks.

This module never trains or evaluates a model and does not start Streamlit.
Run ``python main.py --help`` to see the available data commands.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from datasets.load_dataset import inspect_dataset, load_dataset
from preprocessing.preprocess import preprocess_dataset


def _print_dataset_report(data) -> None:
    """Print the loader's descriptive checks and a compact data preview."""
    report = inspect_dataset(data)
    print(f"Dataset dimensions: {report['dimensions']}")
    print(f"Columns: {report['columns']}")
    print(f"Missing values: {report['missing_values_by_column']}")
    print(f"Duplicate records: {report['duplicate_records']}")
    print("Preview (first 5 rows):")
    print(data.head().to_string(index=False))


def inspect_command() -> int:
    """Load and inspect the source CSV without writing any artifacts."""
    data = load_dataset()
    print("Synthetic demonstration dataset loaded.")
    _print_dataset_report(data)
    return 0


def prepare_command() -> int:
    """Load and inspect the source dataset, then explicitly preprocess it."""
    data = load_dataset()
    print("Source dataset loaded; validating it before preprocessing.")
    _print_dataset_report(data)

    report = preprocess_dataset(data)
    print("Preprocessing completed.")
    print(f"Input features: {report['feature_names']}")
    for split_name in ("train", "validation", "test"):
        print(
            f"{split_name}: {report['split_sizes'][split_name]} records; "
            f"class counts (0=fail, 1=pass): {report['class_counts'][split_name]}"
        )
    print(f"Saved scaler: {report['scaler_path']}")
    print(f"Saved processed splits: {report['processed_dir']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit data-only CLI; no command runs by default."""
    parser = argparse.ArgumentParser(
        description="Inspect the synthetic dataset or explicitly prepare data splits."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "inspect",
        help="load and inspect datasets/student_data.csv without writing files",
    )
    commands.add_parser(
        "prepare",
        help="load and inspect the source CSV, then run preprocessing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one explicitly requested data command."""
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return inspect_command()
    if args.command == "prepare":
        return prepare_command()
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
