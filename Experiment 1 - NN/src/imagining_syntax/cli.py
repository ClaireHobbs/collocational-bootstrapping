"""imsyn — single console entry point for the imagining_syntax package.

Subcommand groups (`run`, `gen`) are populated as their underlying modules
move into the package. Until then, each group's parser exists but exposes
no subcommands.
"""
import argparse
import sys


def _add_run_group(subparsers: argparse._SubParsersAction) -> None:
    from imagining_syntax.runners import run as run_module
    run_module.add_parser(subparsers)


def _add_gen_group(subparsers: argparse._SubParsersAction) -> None:
    from imagining_syntax.data import generate, minimal_pairs
    gen = subparsers.add_parser("gen", help="Generate datasets and minimal-pair files.")
    gen_sub = gen.add_subparsers(dest="gen_subcommand", metavar="SUBCOMMAND")
    generate.add_parser(gen_sub)
    minimal_pairs.add_parser(gen_sub)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imsyn")
    subparsers = parser.add_subparsers(dest="group", metavar="GROUP")
    _add_run_group(subparsers)
    _add_gen_group(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "group", None):
        from imagining_syntax.runners import paper
        return paper.main(args) or 0
    func = getattr(args, "func", None)
    if func is None:
        # Group selected but no subcommand wired up yet (or none provided).
        print(f"imsyn {args.group}: no subcommands available yet")
        return 0
    return func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
