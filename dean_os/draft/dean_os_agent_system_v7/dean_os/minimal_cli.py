from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system import create_minimal_system
from dean_os.schemas import MarketContext


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimally complete, review-only DEAN-OS composition: "
            "bounded pipeline control plus one portable domain analyst."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--domain", default="semiconductor_ai_infrastructure")
    parser.add_argument("--horizon-days", type=int, default=180)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--timeframes", nargs="*", default=["1d"])
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--disable-pipeline", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save-world-model-artifacts", action="store_true")
    parser.add_argument("--disable-world-state-store", action="store_true")
    parser.add_argument("--world-state-store", type=Path)
    parser.add_argument("--historical-analog-limit", type=int, default=5)
    return parser


def _load_input(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--input-json must contain one JSON object")
    return payload


def _build_context(args: argparse.Namespace) -> MarketContext:
    payload = _load_input(args.input_json)
    payload.setdefault("as_of", args.as_of or datetime.now(UTC).isoformat())
    if args.tickers:
        payload["tickers"] = args.tickers
    else:
        payload.setdefault("tickers", [])
    if args.timeframes:
        payload["timeframes"] = args.timeframes
    else:
        payload.setdefault("timeframes", ["1d"])
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("MarketContext.metadata must be a JSON object")
    metadata.setdefault("entrypoint", "dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_cli")
    return MarketContext.model_validate(payload)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    system = create_minimal_system(
        project_root=args.project_root,
        domain_id=args.domain,
        horizon_days=args.horizon_days,
        pipeline_enabled=not args.disable_pipeline,
        soft_mode=not args.strict,
        save_world_model_artifacts=args.save_world_model_artifacts,
        save_world_state_snapshots=not args.disable_world_state_store,
        world_state_store_path=args.world_state_store,
        historical_analog_limit=args.historical_analog_limit,
    )
    result = await system.run(_build_context(args))
    return result.model_dump(mode="json")


def main() -> int:
    args = _parser().parse_args()
    payload = asyncio.run(_run(args))
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
