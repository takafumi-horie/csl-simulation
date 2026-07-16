"""Command-line interface for the standalone CSL package.

スタンドアローン CSL パッケージのコマンドラインインターフェース。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_config
from .data import write_compact_fixture


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free command parser.

    副作用のないコマンドパーサーを構築する。

    Returns:
        Configured argument parser.
        設定済み引数パーサー。
    """
    parser = argparse.ArgumentParser(
        prog="legacy-csl",
        description="Standalone categorical Uni-CSL / Inter-CSL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("learn", "predict", "analyze", "validate-data"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
    fixture_parser = subparsers.add_parser("make-fixture")
    fixture_parser.add_argument("destination", type=Path)
    fixture_parser.add_argument("--documents", type=int, default=8)
    fixture_parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI command.

    CLI コマンドを1つ実行する。

    Args:
        argv: Optional arguments excluding the executable name.
            実行ファイル名を除く任意の引数。

    Returns:
        Process exit code.
        プロセス終了コード。
    """
    args = build_parser().parse_args(argv)
    if args.command == "make-fixture":
        write_compact_fixture(
            args.destination,
            num_documents=args.documents,
            seed=args.seed,
        )
        print(f"fixture written to {args.destination.resolve()}")
        return 0

    config = load_config(args.config)
    if args.command == "learn":
        from .runner import learn

        outputs = learn(config)
        for output in outputs:
            print(output)
        return 0
    if args.command == "predict":
        from .runner import predict

        predict(config)
        return 0
    if args.command == "analyze":
        from .runner import analyze

        analyze(config)
        return 0
    from .runner import validate_data

    validate_data(config)
    print("input data is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
