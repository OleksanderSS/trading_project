from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dean_os.scaffold import create_domain_profile, generate_registry_entry  # noqa: E402


def _cmd_check() -> None:
    """Domain Readiness Check — validate all domain profiles."""
    import os
    import yaml

    profiles_dir = Path("config/domain_profiles")
    if not profiles_dir.is_dir():
        print("No domain profiles directory found.")
        return

    registry_path = Path("dean_os/config/agent_registry.yaml")
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.safe_load(f) or {}
    agents = registry.get("agents", {})

    profile_files = sorted(profiles_dir.glob("*.yaml"))
    if not profile_files:
        print("No domain profiles found.")
        return

    print(f"{'Domain':25} {'YAML':6} {'Agent':7} {'Keywords':9} {'Tickers':8} {'Enabled':8}")
    print("-" * 70)

    all_ok = True
    for pf in profile_files:
        domain_id = pf.stem
        yaml_ok = "OK"
        try:
            with open(pf) as f:
                prof = yaml.safe_load(f) or {}
        except Exception:
            prof = {}
            yaml_ok = "ERR"

        keywords = prof.get("sector_keywords", prof.get("keywords", []))
        kw_count = len(keywords) if isinstance(keywords, (list, dict)) else 0

        tickers = prof.get("ticker_universe_hint", prof.get("tickers", []))
        tk_count = len(tickers) if isinstance(tickers, (list, dict)) else 0

        agent_cfg = next((a for n, a in agents.items() if a.get("domain_id") == domain_id), None)
        agent_name = next((n for n, a in agents.items() if a.get("domain_id") == domain_id), "")
        agent_ok = "OK" if agent_cfg else "MISS"
        enabled = "ON" if agent_cfg and agent_cfg.get("enabled") else "OFF"

        if yaml_ok != "OK" or kw_count == 0:
            all_ok = False

        print(f"{domain_id:25} {yaml_ok:6} {agent_ok:7} {str(kw_count):9} {str(tk_count):8} {enabled:8}")

    disabled = [n for n, a in agents.items() if a.get("domain_id") and not a.get("enabled")]
    no_domain = [n for n, a in agents.items() if a.get("enabled") and not a.get("domain_id") and a.get("branch") == "pipeline"]

    if disabled:
        print(f"\nDisabled domain agents: {', '.join(disabled)}")
    if no_domain:
        print(f"Enabled pipeline agents without domain_id: {', '.join(no_domain)}")

    print(f"\nTotal: {len(profile_files)} profiles, {len([n for n,a in agents.items() if a.get('domain_id')])} registered domain agents")
    print("All checks pass!" if all_ok else "Some checks failed — see above.")


def _cmd_coherence() -> None:
    """Run coherence scan offline."""
    from dean_os.agents.coherence_scan import CoherenceScanAgent, AGENT_DOMAIN_MAP, OVERLAP_PAIRS
    print(f"Coherence Scan — {len(AGENT_DOMAIN_MAP)} mapped agents, {len(OVERLAP_PAIRS)} overlap pairs")
    print()
    print("Overlap pairs to watch:")
    for a, b in OVERLAP_PAIRS:
        print(f"  {a:30} <-> {b}")
    print()

    # Try loading tracker stats as context
    from dean_os.outcome_tracker import OutcomeTracker
    try:
        cal = OutcomeTracker().calibrate()
        if cal.total_outcomes > 0:
            print(f"Tracker calibration available:")
            print(f"  Outcomes: {cal.total_outcomes}, Accuracy: {cal.accuracy_rate:.0%}, Brier: {cal.brier_score}")
        else:
            print("Tracker has no outcomes yet — coherence will be estimated.")
    except Exception:
        pass


def _cmd_health(json_mode: bool = False) -> None:
    from dean_os.system_health import check_all, print_report
    import json
    result = check_all()
    if json_mode:
        class SetEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, set):
                    return list(sorted(o))
                return super().default(o)
        print(json.dumps(result, indent=2, cls=SetEncoder, default=str))
    else:
        print_report(result)


def _cmd_stats(json_mode: bool = False) -> None:
    from dean_os.agent_stats import AgentStatsStore, print_stats
    import json
    store = AgentStatsStore()
    stats = store.get_stats()
    if json_mode:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print_stats(stats)


def _cmd_inventory(json_mode: bool = False) -> None:
    from dean_os.data_inventory import get_table_info, print_table_info
    import json
    info = get_table_info()
    if json_mode:
        print(json.dumps({k: {sk: sv for sk, sv in v.items() if sk != "columns"} for k, v in info.items()}, indent=2, default=str))
    else:
        print_table_info(info)


def _cmd_search_columns(args: list[str]) -> None:
    from dean_os.data_inventory import search_columns
    query = args[1] if len(args) > 1 else ""
    if not query:
        print("Usage: python dean_domain_scaffold.py search <column_name>")
        return
    results = search_columns(query)
    if not results:
        print(f"No columns found matching '{query}'")
        return
    print(f"Columns matching '{query}':")
    for r in results:
        print(f"  {r['table']:35} {r['column']:30} {r['type']}")


def _cmd_dq() -> None:
    from dean_os.data_inventory import data_quality_report, print_dq_report
    print_dq_report(data_quality_report())


def _cmd_diag() -> None:
    import yaml, json
    from pathlib import Path

    project_root = Path(__file__).parent
    reg_path = project_root / "dean_os" / "config" / "agent_registry.yaml"
    agents = yaml.safe_load(reg_path.read_text(encoding="utf-8")).get("agents", {})
    enabled = [n for n, c in agents.items() if c.get("enabled")]
    disabled = [n for n, c in agents.items() if not c.get("enabled")]

    profiles = list((project_root / "config" / "domain_profiles").glob("*.yaml"))

    print("=" * 60)
    print("  DEAN-OS Diagnostic Summary")
    print("=" * 60)
    print(f"\n  Registry:     {len(agents)} agents ({len(enabled)} enabled, {len(disabled)} disabled)")
    print(f"  Profiles:     {len(profiles)} domain profiles")
    print(f"\n  Enabled agents ({len(enabled)}):")
    for n in sorted(enabled):
        cfg = agents[n]
        print(f"    {n:30} {cfg.get('branch','?'):12} veto={cfg.get('veto_level','?'):6} {cfg.get('domain_id','') or ''}")
    print()
    try:
        from dean_os.data_inventory import get_table_info
        info = get_table_info()
        total_rows = sum(v["rows"] for v in info.values())
        print(f"  DuckDB:       {len(info)} tables, {total_rows:,} total rows")
    except Exception:
        print("  DuckDB:       <not available>")
    print(f"\n  Project root: {project_root}")


def _cmd_outcomes() -> None:
    from dean_os.outcome_tracker import OutcomeTracker
    tracker = OutcomeTracker()
    stats = tracker.stats()
    print(f"Events tracked: {stats['events']}")
    print(f"Outcomes:       {stats['outcomes']}")
    print(f"Due for check:  {stats['due']}")
    print(f"Intervals:      {stats['intervals']}")
    print()
    events = tracker.list_events(10)
    if not events:
        print("No events yet. Register events via news_event_analyzer or manually.")
        return
    print(f"{'Type':>18} {'Preds':>5} {'Outs':>5} {'Headline':60}")
    print("-" * 88)
    for ev in events:
        print(f"  {ev['event_type']:>16} {ev['predictions']:>5} {ev['outcomes']:>5} {ev['headline']:60}")
    trades = tracker.check_paper_trades()
    if trades:
        print(f"\nPaper trade outcomes ({len(trades)}):")
        for t in trades:
            print(f"  {t['label']:>4}  {t['headline'][:70]}")
    cal = tracker.calibrate()
    if cal.total_outcomes > 0:
        print(f"\nCalibration: Brier={cal.brier_score:.4f}, Accuracy={cal.accuracy_rate:.2%}")


def _cmd_list_agents() -> None:
    import yaml
    path = Path(__file__).parent / "dean_os" / "config" / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    agents = data.get("agents", {})
    print(f"{'Agent':35} {'Enabled':>8} {'Branch':14} {'Veto':10} {'Domain':30}")
    print("-" * 97)
    for name, cfg in sorted(agents.items()):
        enabled = str(cfg.get("enabled", False))
        branch = cfg.get("branch", "?")
        veto = cfg.get("veto_level", "?")
        domain = cfg.get("domain_id", "") or ""
        print(f"  {name:33} {enabled:>8} {branch:14} {veto:10} {domain:30}")
    total = len(agents)
    en = sum(1 for c in agents.values() if c.get("enabled"))
    print(f"\n  Total: {total} agents, {en} enabled, {total - en} disabled")


def _cmd_registry_show(agent_name: str) -> None:
    import yaml
    path = Path(__file__).parent / "dean_os" / "config" / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    agents = data.get("agents", {})
    cfg = agents.get(agent_name)
    if not cfg:
        print(f"Agent '{agent_name}' not found in registry")
        return
    print(f"Agent: {agent_name}")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"  enabled: {cfg.get('enabled', False)}")


def _cmd_profile_show(domain_id: str) -> None:
    import yaml
    path = Path(__file__).parent / "config" / "domain_profiles" / f"{domain_id}.yaml"
    if not path.exists():
        print(f"Domain profile '{domain_id}' not found at config/domain_profiles/")
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"Domain: {domain_id}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    - {item}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                if isinstance(sv, list):
                    print(f"    {sk}:")
                    for si in sv:
                        print(f"      - {si}")
                else:
                    print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")


def _cmd_validate_config() -> None:
    import yaml
    import importlib

    config_dir = Path(__file__).parent / "dean_os" / "config"
    errors: list[str] = []

    # 1. Validate agent_registry.yaml
    reg_path = config_dir / "agent_registry.yaml"
    if not reg_path.exists():
        errors.append(f"Missing: {reg_path}")
    else:
        data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
        agents = data.get("agents", {})
        if not agents:
            errors.append("agent_registry.yaml: no agents found")
        for name, cfg in agents.items():
            class_path = cfg.get("class_path", "")
            if ":" in class_path:
                module, cls = class_path.rsplit(":", 1)
                try:
                    mod = importlib.import_module(module)
                    if not hasattr(mod, cls):
                        errors.append(f"{name}: class '{cls}' not found in {module}")
                except ImportError as e:
                    errors.append(f"{name}: cannot import {module} — {e}")
            else:
                errors.append(f"{name}: invalid class_path '{class_path}'")
            domain_id = cfg.get("domain_id")
            if domain_id:
                project_root = Path(__file__).parent
                dp = project_root / "config" / "domain_profiles" / f"{domain_id}.yaml"
                if not dp.exists():
                    errors.append(f"{name}: domain profile '{domain_id}.yaml' not found at config/domain_profiles/")

    # 2. Validate domain profiles
    project_root = Path(__file__).parent
    profile_dir = project_root / "config" / "domain_profiles"
    if not profile_dir.exists():
        errors.append("config/domain_profiles/ directory not found")
    for f in sorted(profile_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                errors.append(f"{f.name}: not a valid YAML mapping")
        except Exception as e:
            errors.append(f"{f.name}: parse error — {e}")

    # 3. Validate other YAML configs
    for f in sorted(config_dir.glob("*.yaml")):
        if f.name == "agent_registry.yaml":
            continue
        try:
            yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{f.name}: parse error — {e}")

    profile_count = len(list(profile_dir.glob("*.yaml"))) if profile_dir.exists() else 0
    if errors:
        print(f"  Found {len(errors)} issue(s):\n")
        for e in errors:
            print(f"  ! {e}")
    else:
        print("  All config files valid.")
        print(f"  Agents: {len(agents)}")
        print(f"  Domain profiles: {profile_count}")


def main() -> None:
    args = sys.argv[1:]
    json_mode = "--json" in args
    if json_mode:
        args = [a for a in args if a != "--json"]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python dean_domain_scaffold.py <command>")
        print("       create <domain_id>   New domain profile")
        print("       list                 Domain profiles")
        print("       calibration          Outcome tracker stats")
        print("       check                Domain readiness")
        print("       coherence            Overlap map")
        print("       health               System health")
        print("       stats                Agent run stats")
        print("       inventory            DuckDB table inventory")
        print("       search <col>         Search columns in DuckDB")
        print("       list-agents          Agent registry table")
        print("       registry show <name>  Show agent config")
        print("       profiles show <id>    Show domain profile")
        print("       validate-config      Check YAML + class_path validity")
        print("       outcomes             Outcome tracker events + calibration")
        print("       diag                 One-page system diagnostic")
        print("       dq                   DuckDB data quality report")
        print("       --json               JSON output (stats, health, inventory)")
        print("\nExamples:")
        print("  python dean_domain_scaffold.py create agriculture")
        print("  python dean_domain_scaffold.py inventory")
        print("  python dean_domain_scaffold.py list-agents")
        return

    if args[0] == "calibration":
        from dean_os.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker()
        stats = tracker.stats()
        print(f"Events tracked: {stats['events']}")
        print(f"Predictions:    {stats['predictions']}")
        print(f"Outcomes:       {stats['outcomes']}")
        print(f"Due for check:  {stats['due']}")
        print(f"Intervals:      {stats['intervals']}")
        cal = tracker.calibrate()
        if cal.total_outcomes > 0:
            print(f"\nCalibration:")
            print(f"  Brier score:    {cal.brier_score}")
            print(f"  Accuracy rate:  {cal.accuracy_rate:.2%}")
            print(f"  By interval:")
            for interval, metrics in sorted(cal.by_interval.items()):
                print(f"    {interval:>4}d: count={metrics['count']}, accuracy={metrics['accuracy']:.2%}, avg_score={metrics['avg_score']:.3f}")
        else:
            print("\nNo outcomes yet — register events and run the orchestrator to build data.")
        print(f"\nRecent events:")
        for ev in tracker.list_events(5):
            print(f"  [{ev['event_type']:>18}] {ev['headline']}")
        return

    if args[0] == "check":
        _cmd_check()
        return

    if args[0] == "coherence":
        _cmd_coherence()
        return

    if args[0] == "health":
        _cmd_health(json_mode=json_mode)
        return

    if args[0] == "stats":
        _cmd_stats(json_mode=json_mode)
        return

    if args[0] == "inventory":
        _cmd_inventory(json_mode=json_mode)
        return

    if args[0] == "search":
        _cmd_search_columns(args)
        return

    if args[0] == "outcomes":
        _cmd_outcomes()
        return

    if args[0] == "diag":
        _cmd_diag()
        return

    if args[0] == "dq":
        _cmd_dq()
        return

    if args[0] == "list-agents":
        _cmd_list_agents()
        return

    if args[0] == "registry" and len(args) >= 3 and args[1] == "show":
        _cmd_registry_show(args[2])
        return

    if args[0] == "profiles" and len(args) >= 3 and args[1] == "show":
        _cmd_profile_show(args[2])
        return

    if args[0] == "validate-config":
        _cmd_validate_config()
        return

    if args[0] == "list":
        if len(args) >= 2 and args[1] == "--details":
            import yaml
            project_root = Path(__file__).parent
            for f in sorted((project_root / "config" / "domain_profiles").glob("*.yaml")):
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                name = data.get("display_name", f.stem)
                req = len(data.get("required_evidence_types", []))
                use = len(data.get("useful_evidence_types", []))
                kw = len(data.get("sector_keywords", []))
                tickers = len(data.get("ticker_universe_hint", []))
                print(f"  {f.stem:35} {name:40} req={req} useful={use} kw={kw} tickers={tickers}")
        else:
            from dean_os.analysts.profiles import list_domain_profiles
            for pid in list_domain_profiles():
                print(f"  {pid}")
        return

    if args[0] == "create" and len(args) >= 2:
        domain_id = args[1]
        display_name = None
        i = 2
        while i < len(args):
            if args[i] == "--name" and i + 1 < len(args):
                display_name = args[i + 1]
                i += 2
            else:
                i += 1
        display_name = display_name or domain_id.replace("_", " ").title()

        try:
            path = create_domain_profile(domain_id, display_name)
            print(f"Profile created: {path}")
            print("\nAdd this to agent_registry.yaml:\n")
            print(generate_registry_entry(domain_id))
        except FileExistsError as e:
            print(f"Error: {e}")
            return

        print(f"Done. Domain '{domain_id}' scaffold is ready.")
        return

    print(f"Unknown command: {args[0]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
