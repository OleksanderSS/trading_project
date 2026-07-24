from __future__ import annotations

import argparse
import json
import sys

from dean_os.current_architecture_map import CurrentArchitectureMap, render_current_architecture_map_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the current architecture map report.")
    parser.add_argument("--output-dir", default="reports/dean_os/current_architecture_map")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = CurrentArchitectureMap(args.output_dir).build(save=not args.no_save)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_current_architecture_map_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
