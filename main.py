"""Command line interface to run the four programming problems."""
from __future__ import annotations

import argparse
from pathlib import Path

from solutions import problem1_diabetes, problem2_breast_cancer, problem3_mnist, problem4_pretrained


PROBLEMS = {
    "problem1": problem1_diabetes.run,
    "problem2": problem2_breast_cancer.run,
    "problem3": problem3_mnist.run,
    "problem4": problem4_pretrained.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explainability experiments.")
    parser.add_argument(
        "problem",
        choices=sorted(PROBLEMS.keys()) + ["all"],
        help="Which problem pipeline to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Base directory for generated reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.problem == "all":
        for key in sorted(PROBLEMS.keys()):
            print(f"Running {key}...")
            PROBLEMS[key](args.output_dir / key)
    else:
        PROBLEMS[args.problem](args.output_dir / args.problem)


if __name__ == "__main__":
    main()
