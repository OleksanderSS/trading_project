from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dean_os.review_only_real_source_normalized_packet_validation_gate import (
    build_validation_gate,
    render_validation_gate_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the review-only real source normalized packet validation gate.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", default="reports/dean_os/real_source_normalized_packet_validation_gate")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    upstream_payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = build_validation_gate(upstream_payload)

    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "latest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "latest.md").write_text(
            render_validation_gate_markdown(result), encoding="utf-8"
        )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_validation_gate_markdown(result).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
