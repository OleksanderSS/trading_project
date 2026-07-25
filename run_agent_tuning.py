from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dean_os.agents.model_performance import inspect_model_performance
from dean_os.agents.tuning import TuningAgent
from dean_os.schemas import MarketContext
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TuningAgent as a proposal-only planner with optional PipelineControlSurface gating.",
    )
    parser.add_argument("performance_path", nargs="?", default=None, help="Model performance JSON/CSV artifact.")
    parser.add_argument("--regime-context-json", default=None)
    parser.add_argument("--control-surface-json", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--timeframes", nargs="*", default=None)
    parser.add_argument("--require-control-surface", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/tuning")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if args.performance_path:
        metadata["model_performance"] = _load_model_performance(args.performance_path)
    if args.regime_context_json:
        metadata["regime_context"] = _load_json(args.regime_context_json)
    control_surface_path = args.control_surface_json
    if control_surface_path:
        metadata["pipeline_control_surface"] = _load_json(control_surface_path)

    context = MarketContext(
        tickers=[ticker.upper() for ticker in args.tickers or [] if str(ticker).strip()],
        timeframes=[str(timeframe) for timeframe in args.timeframes or [] if str(timeframe).strip()],
        metadata=metadata,
    )
    report = await TuningAgent(
        name="tuning",
        config={
            "tickers": context.tickers,
            "timeframes": context.timeframes,
            "require_control_surface": args.require_control_surface,
        },
    ).run(context)
    payload = {
        "run_id": "tuning_" + report.timestamp.replace(":", "").replace("-", "").replace(".", "_"),
        "created_at": report.timestamp,
        "mode": "tuning_agent",
        "inputs": {
            "performance_path": args.performance_path,
            "regime_context_json": args.regime_context_json,
            "control_surface_json": control_surface_path,
            "tickers": context.tickers,
            "timeframes": context.timeframes,
            "require_control_surface": args.require_control_surface,
        },
        "report": report.model_dump(mode="json"),
        "tuning": context.metadata.get("tuning", {}),
        "action_proposals": [proposal.model_dump(mode="json") for proposal in context.action_proposals],
    }
    _save_report(payload, Path(args.output_dir))
    return payload


def _load_model_performance(path: str | Path) -> dict[str, Any]:
    payload = _load_json_or_csv(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("metrics_snapshot"), dict):
            return payload["metrics_snapshot"]
        if isinstance(payload.get("model_performance"), dict):
            return payload["model_performance"]
        if "threshold_failures" in payload:
            return payload
    return inspect_model_performance(performance_path=path)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = _load_json_or_csv(path)
    return payload if isinstance(payload, dict) else {"items": payload}


def _load_json_or_csv(path: str | Path) -> Any:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        return json.loads(resolved.read_text(encoding="utf-8"))
    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(resolved)
        if frame.empty:
            return {}
        return frame.iloc[-1].to_dict()
    raise ValueError(f"Unsupported artifact type: {resolved.suffix}. Use .json or .csv.")


def _save_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    json_path = output_dir / f"{run_id}.json"
    md_path = output_dir / f"{run_id}.md"
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"
    paths = {"json": json_path, "markdown": md_path, "latest_json": latest_json, "latest_markdown": latest_md}
    payload["saved_paths"] = {key: str(value) for key, value in paths.items()}
    rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
    rendered_md = _render_markdown(payload)
    json_path.write_text(rendered_json, encoding="utf-8")
    latest_json.write_text(rendered_json, encoding="utf-8")
    md_path.write_text(rendered_md, encoding="utf-8")
    latest_md.write_text(rendered_md, encoding="utf-8")


def _render_markdown(payload: dict[str, Any]) -> str:
    report = payload.get("report", {})
    tuning = payload.get("tuning", {})
    lines = [
        "# DEAN-OS Tuning Agent",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Verdict: `{report.get('verdict')}`",
        f"- Tuning status: `{tuning.get('status')}`",
        f"- Proposal count: {tuning.get('proposal_count')}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in report.get("reasons", []))
    lines.extend(["", "## Proposals", ""])
    for proposal in payload.get("action_proposals", []):
        lines.append(f"- `{proposal.get('action_type')}` -> `{proposal.get('target')}`")
    return "\n".join(lines).strip() + "\n"


def print_summary(payload: dict[str, Any]) -> None:
    report = payload.get("report", {})
    tuning = payload.get("tuning", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {report.get('verdict')} | tuning status={tuning.get('status')}")
    print(f"Proposal count: {tuning.get('proposal_count')}")
    for proposal in payload.get("action_proposals", []):
        print(f"- {proposal.get('action_type')} -> {proposal.get('target')}")
        if proposal.get("command_preview"):
            print(f"  {proposal.get('command_preview')}")
    saved = payload.get("saved_paths", {})
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")
        print(f"Report Markdown: {saved.get('latest_markdown') or saved.get('markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
