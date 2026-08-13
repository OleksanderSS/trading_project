from __future__ import annotations

import argparse

from dean_os.command_index import (
    build_command_index,
    load_retired,
    render_markdown,
    undocumented_drift,
    write_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate dean_os/COMMAND_INDEX.md from the run_agent_*.py wrappers on disk."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without writing the index. Exits 1 if the index is stale.",
    )
    parser.add_argument("--print-markdown", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index = build_command_index()
    retired = load_retired()
    drift = undocumented_drift()

    if args.print_markdown:
        print(render_markdown(index, retired))
        return 0

    if args.check:
        from dean_os.command_index import INDEX_PATH

        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        stale = current != render_markdown(index, retired)
        print(f"Commands on disk: {len(index)}")
        print(f"Recorded as retired/missing: {len(retired)}")
        print(f"Index stale: {stale}")
        if drift:
            print(f"Undocumented drift: {len(drift)} command(s) named in prose docs but neither present nor retired:")
            for name, docs in sorted(drift.items()):
                print(f"  {name}  <- {', '.join(docs)}")
        return 1 if (stale or drift) else 0

    path = write_index()
    print(f"Commands on disk: {len(index)}")
    print(f"Recorded as retired/missing: {len(retired)}")
    print(f"Wrote: {path}")
    if drift:
        print(f"\nWARNING: {len(drift)} command(s) named in prose docs are neither present nor retired:")
        for name, docs in sorted(drift.items()):
            print(f"  {name}  <- {', '.join(docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
