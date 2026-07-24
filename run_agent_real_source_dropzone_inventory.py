from __future__ import annotations

import argparse
import sys

from dean_os.real_source_dropzone_inventory import (
    RealSourceDropzoneInventory,
    render_real_source_dropzone_inventory_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory operator-supplied research files in a dropzone.")
    parser.add_argument("--dropzone", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/real_source_dropzone_inventory_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    kwargs = {"recursive": args.recursive, "save": not args.no_save}
    if args.dropzone is not None:
        kwargs["dropzone"] = args.dropzone

    payload = RealSourceDropzoneInventory(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_real_source_dropzone_inventory_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
