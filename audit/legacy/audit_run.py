#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              DEAN FULL AUDIT RUNNER  v1.0                                  ║
║  Запускає audit.py + audit_logic.py і генерує зведений звіт                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Запуск:
    python audit_run.py                          # аудит поточної папки
    python audit_run.py --root src               # конкретна папка
    python audit_run.py --root src --html        # HTML звіт (audit_report.html)
    python audit_run.py --root src --fix         # показати тільки CRITICAL+HIGH з фіксами
    python audit_run.py --root src --top 20      # топ-20 найкритичніших файлів

Вимоги: audit.py + audit_logic.py в тій самій папці що і audit_run.py
"""

import argparse
import collections
import importlib.util
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# ── завантаження модулів аудиту ───────────────────────────────────────────────

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _find_audit_modules(runner_dir: Path):
    a1 = runner_dir / "audit.py"
    a2 = runner_dir / "audit_logic.py"
    a3 = runner_dir / "audit_engagement.py"
    missing = [p for p in (a1, a2, a3) if not p.exists()]
    if missing:
        print(f"❌ Не знайдено: {', '.join(str(m) for m in missing)}")
        print("   Переконайся що audit.py, audit_logic.py і audit_engagement.py лежать поруч з audit_run.py")
        sys.exit(1)
    return _load_module("audit", a1), _load_module("audit_logic", a2), _load_module("audit_engagement", a3)


# ── агрегація результатів ─────────────────────────────────────────────────────

SEVERITY_ORDER  = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
SEVERITY_EMOJI  = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}
SEVERITY_COLOR  = {
    "CRITICAL": "#e74c3c", "HIGH": "#e67e22",
    "MEDIUM":   "#f1c40f", "LOW": "#3498db", "INFO": "#95a5a6",
}
SEVERITY_SCORE  = {"CRITICAL": 100, "HIGH": 20, "MEDIUM": 5, "LOW": 1, "INFO": 0}

CATEGORY_DESC = {
    # audit.py
    "ARC":   ("Архітектура",        "Циклічні залежності, God Objects, tight coupling"),
    "DUP":   ("Дублювання",         "Copy-paste код, однакова логіка"),
    "BUG":   ("Потенційні баги",    "Async помилки, exception handling, None-checks"),
    "CFG":   ("Конфігурація",       "Hardcoded значення, magic numbers, шляхи"),
    "SEC":   ("Безпека",            "SQL injection, credentials, path traversal"),
    "CMX":   ("Складність",         "Висока cyclomatic complexity, великі функції"),
    "TYP":   ("Типи",               "Відсутні анотації, некоректне порівняння"),
    "LOG":   ("Логування",          "Відсутнє або некоректне логування"),
    "IMP":   ("Імпорти",            "Невикористані, зірочкові, неправильний порядок"),
    "ML":    ("ML (DEAN)",          "ModelFactory, DiaryEngine, noise filter"),
    # audit_logic.py
    "LEAK":  ("Витік даних",        "Temporal/target/scaler leakage"),
    "PANDAS":("Pandas анти-патерни","Chained indexing, inplace, NaN"),
    "MATH":  ("Математика",         "Ділення на нуль, mean-of-means, float порівняння"),
    "LOGIC": ("Логіка",             "Unreachable code, тавтології, dead code"),
    "ASYNC": ("Async",              "gather без return_exceptions, sleep, race conditions"),
    "RES":   ("Resource leaks",     "Незакриті файли, DB connections, threads"),
    "FEAT":  ("Feature Engineering","rolling min_periods, RSI clip, std=0"),
    "STATE": ("Стан",               "Global variables, mutable class attrs, singleton"),
    "API":   ("API використання",   "sklearn, pandas, numpy анти-патерни"),
    "FLOW":  ("Потік даних",        "Unreachable branches, завжди True/False"),
    # audit_engagement.py
    "ENG":   ("User Engagement",    "User feedback loops, interactive components, config options"),
    "EXP":   ("Explainability",    "Model explanation methods, feature importance, decision logging"),
    "MON":   ("Monitoring",         "Alerting systems, performance metrics tracking, anomaly detection"),
    "TEST":  ("Test Coverage",      "Integration tests, E2E tests, performance tests"),
    "DOC":   ("Documentation",      "User docs, API docs, architecture docs"),
}




class AggregatedReport:
    def __init__(self):
        self.issues:      list[dict] = []
        self.file_count:  int = 0
        self.line_count:  int = 0
        self.generated_at = datetime.now()

    def add_from_result(self, result_obj, source_label: str) -> None:
        for iss in result_obj.issues:
            self.issues.append({
                "source":   source_label,
                "category": iss.category,
                "severity": iss.severity,
                "file":     iss.file,
                "line":     iss.line,
                "message":  iss.message,
                "code":     getattr(iss, "code", ""),
                "fix_hint": getattr(iss, "fix_hint", ""),
            })
        self.file_count  = max(self.file_count, getattr(result_obj, "file_count", 0))
        self.line_count  = max(self.line_count, getattr(result_obj, "line_count", 0))

    def sorted_issues(self, min_severity: str = "LOW", categories: list[str] = None):
        min_idx = SEVERITY_ORDER.index(min_severity) if min_severity in SEVERITY_ORDER else 3
        out = [
            i for i in self.issues
            if SEVERITY_ORDER.index(i["severity"]) <= min_idx
            and (not categories or i["category"] in categories)
        ]
        return sorted(out, key=lambda x: (
            SEVERITY_ORDER.index(x["severity"]),
            x["category"],
            x["file"],
            x["line"],
        ))

    def stats(self) -> dict[str, Any]:
        by_sev  = collections.Counter(i["severity"]  for i in self.issues)
        by_cat  = collections.Counter(i["category"]  for i in self.issues)
        by_file = collections.Counter(i["file"]       for i in self.issues)
        score   = sum(SEVERITY_SCORE.get(i["severity"], 0) for i in self.issues)
        return {
            "total":    len(self.issues),
            "files":    self.file_count,
            "lines":    self.line_count,
            "score":    score,
            "grade":    _health_grade(score),
            "by_severity": dict(by_sev),
            "by_category": dict(by_cat),
            "top_files": by_file.most_common(15),
        }


def _health_grade(score: int) -> str:
    if score == 0:         return "A+ ✅"
    if score < 50:         return "A  ✅"
    if score < 200:        return "B  🟡"
    if score < 500:        return "C  🟠"
    if score < 1000:       return "D  🔴"
    return                        "F  💀"


# ── текстовий вивід ───────────────────────────────────────────────────────────

def print_console_report(
    report:       AggregatedReport,
    min_severity: str = "LOW",
    categories:   list[str] = None,
    show_fix:     bool = True,
    top_files:    int = 0,
    max_issues:   int = 600,
) -> None:
    issues  = report.sorted_issues(min_severity, categories)
    stats   = report.stats()

    print(f"\n{'═' * 72}")
    print(f"  🔍 DEAN FULL AUDIT REPORT  —  {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═' * 72}")
    print(f"  Файлів: {stats['files']}   Рядків: {stats['lines']:,}   "
          f"Issues: {stats['total']}   Health: {stats['grade']}  (score={stats['score']})")

    # ── підсумок за severity ──────────────────────────────────────────────────
    print()
    for sev in SEVERITY_ORDER:
        cnt = stats["by_severity"].get(sev, 0)
        if cnt:
            bar = "█" * min(cnt, 40)
            print(f"  {SEVERITY_EMOJI[sev]} {sev:<12} {cnt:4}  {bar}")

    # ── підсумок за категорією ────────────────────────────────────────────────
    print(f"\n  {'Категорія':<8}  {'Issues':>6}  Опис")
    print(f"  {'─'*8}  {'─'*6}  {'─'*40}")
    for cat, cnt in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        name, desc = CATEGORY_DESC.get(cat, (cat, ""))
        print(f"  [{cat:<6}]  {cnt:6}  {name} — {desc[:50]}")

    # ── топ проблемних файлів ─────────────────────────────────────────────────
    if top_files and stats["top_files"]:
        print(f"\n  📁 Топ-{top_files} найпроблемніших файлів:")
        for fname, cnt in stats["top_files"][:top_files]:
            bar = "▪" * min(cnt, 30)
            print(f"    {cnt:4}  {fname}  {bar}")

    # ── детальний список ──────────────────────────────────────────────────────
    current_sev = None
    shown       = 0
    for iss in issues:
        if shown >= max_issues:
            print(f"\n  ... і ще {len(issues) - shown} issues (використай --html для повного)")
            break
        sev = iss["severity"]
        if sev != current_sev:
            current_sev = sev
            print(f"\n{'═' * 72}")
            print(f"  {SEVERITY_EMOJI[sev]} {sev}")
            print(f"{'═' * 72}")

        print(f"\n{SEVERITY_EMOJI[sev]} [{iss['category']}] {iss['file']}:{iss['line']}")
        print(f"   {iss['message']}")
        if iss.get("code"):
            print(f"   ▶ {iss['code'].strip()[:110]}")
        if show_fix and iss.get("fix_hint"):
            print(f"   ✏ {iss['fix_hint']}")
        shown += 1

    print(f"\n{'═' * 72}\n")


# ── HTML звіт ─────────────────────────────────────────────────────────────────

def generate_html_report(report: AggregatedReport, out_path: Path) -> None:
    stats  = report.stats()
    issues = report.sorted_issues("INFO")

    # Групуємо по severity
    by_sev: dict[str, list] = collections.defaultdict(list)
    for iss in issues:
        by_sev[iss["severity"]].append(iss)

    rows_html = ""
    for sev in SEVERITY_ORDER:
        for iss in by_sev.get(sev, []):
            color  = SEVERITY_COLOR[iss["severity"]]
            fix    = f'<br><span class="fix">✏ {iss["fix_hint"]}</span>' if iss["fix_hint"] else ""
            code   = f'<code>{iss["code"].strip()[:100]}</code><br>' if iss.get("code") else ""
            rows_html += f"""
            <tr>
              <td><span class="badge" style="background:{color}">{iss['severity']}</span></td>
              <td><span class="cat">{iss['category']}</span></td>
              <td class="filepath">{iss['file']}:{iss['line']}</td>
              <td>{iss['message']}<br>{code}{fix}</td>
            </tr>"""

    # Статистика по категоріях
    cat_rows = ""
    for cat, cnt in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        name, desc = CATEGORY_DESC.get(cat, (cat, ""))
        cat_rows += f"<tr><td>[{cat}]</td><td>{cnt}</td><td>{name}</td><td>{desc}</td></tr>"

    sev_bars = ""
    for sev in SEVERITY_ORDER:
        cnt   = stats["by_severity"].get(sev, 0)
        color = SEVERITY_COLOR[sev]
        pct   = round(cnt / max(stats["total"], 1) * 100)
        sev_bars += f"""
        <div class="sev-row">
          <span class="sev-label">{SEVERITY_EMOJI[sev]} {sev}</span>
          <div class="bar-wrap">
            <div class="bar" style="width:{pct}%;background:{color}"></div>
          </div>
          <span class="sev-cnt">{cnt}</span>
        </div>"""

    top_files_html = ""
    for fname, cnt in stats["top_files"][:15]:
        top_files_html += f"<tr><td>{cnt}</td><td>{fname}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DEAN Audit Report — {report.generated_at.strftime('%Y-%m-%d %H:%M')}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #7ee8a2; font-size: 1.6rem; margin-bottom: 4px; }}
  h2 {{ color: #a0cfff; font-size: 1.1rem; margin: 24px 0 10px; border-bottom: 1px solid #2a2d3e; padding-bottom: 6px; }}
  .meta {{ color: #888; font-size: .85rem; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{ background: #1a1d2e; border-radius: 10px; padding: 16px 22px; min-width: 130px; text-align: center; }}
  .card .val {{ font-size: 2rem; font-weight: 700; color: #7ee8a2; }}
  .card .lbl {{ font-size: .75rem; color: #888; margin-top: 2px; }}
  .grade {{ font-size: 1.8rem !important; }}
  .sev-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .sev-label {{ width: 110px; font-size: .85rem; }}
  .bar-wrap {{ flex: 1; background: #1e2130; border-radius: 4px; height: 14px; margin: 0 10px; }}
  .bar {{ height: 14px; border-radius: 4px; min-width: 2px; transition: width .4s; }}
  .sev-cnt {{ width: 40px; text-align: right; font-size: .85rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; margin-top: 10px; }}
  th {{ background: #1e2130; color: #a0cfff; padding: 8px 10px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e2130; vertical-align: top; }}
  tr:hover td {{ background: #1a1d2e; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: .72rem;
            font-weight: 600; color: #fff; white-space: nowrap; }}
  .cat {{ display: inline-block; background: #2a2d3e; padding: 2px 6px; border-radius: 4px;
          font-size: .72rem; color: #a0cfff; }}
  .filepath {{ font-family: monospace; font-size: .78rem; color: #f1c40f; white-space: nowrap; }}
  code {{ background: #1e2130; padding: 2px 6px; border-radius: 3px; font-size: .78rem;
          color: #e8b4b8; display: inline-block; max-width: 500px; overflow-x: auto; }}
  .fix {{ color: #7ee8a2; font-size: .78rem; }}
  .filter-bar {{ margin: 14px 0; display: flex; gap: 8px; flex-wrap: wrap; }}
  .filter-btn {{ background: #1e2130; border: 1px solid #2a2d3e; color: #ccc; padding: 5px 12px;
                 border-radius: 20px; cursor: pointer; font-size: .78rem; transition: all .2s; }}
  .filter-btn:hover, .filter-btn.active {{ background: #3a4060; color: #fff; border-color: #a0cfff; }}
  #search {{ background: #1e2130; border: 1px solid #2a2d3e; color: #e0e0e0; padding: 7px 14px;
             border-radius: 6px; font-size: .85rem; width: 280px; }}
  .hidden {{ display: none !important; }}
  @media (max-width: 768px) {{ .cards {{ gap: 8px; }} .card {{ min-width: 100px; }} }}
</style>
</head>
<body>
<h1>🔍 DEAN Project Audit Report</h1>
<p class="meta">Згенеровано: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
Файлів: {stats['files']} &nbsp;|&nbsp; Рядків коду: {stats['lines']:,}</p>

<div class="cards">
  <div class="card"><div class="val">{stats['total']}</div><div class="lbl">Всього issues</div></div>
  <div class="card"><div class="val grade">{stats['grade']}</div><div class="lbl">Health grade</div></div>
  <div class="card"><div class="val" style="color:#e74c3c">{stats['by_severity'].get('CRITICAL',0)}</div><div class="lbl">Critical</div></div>
  <div class="card"><div class="val" style="color:#e67e22">{stats['by_severity'].get('HIGH',0)}</div><div class="lbl">High</div></div>
  <div class="card"><div class="val" style="color:#f1c40f">{stats['by_severity'].get('MEDIUM',0)}</div><div class="lbl">Medium</div></div>
  <div class="card"><div class="val">{stats['score']}</div><div class="lbl">Risk score</div></div>
</div>

<h2>📊 Розподіл за severity</h2>
{sev_bars}

<h2>📁 Топ проблемних файлів</h2>
<table>
  <tr><th>Issues</th><th>Файл</th></tr>
  {top_files_html}
</table>

<h2>🗂 За категоріями</h2>
<table>
  <tr><th>Категорія</th><th>Issues</th><th>Назва</th><th>Опис</th></tr>
  {cat_rows}
</table>

<h2>🐛 Всі issues</h2>
<div class="filter-bar">
  <input id="search" type="text" placeholder="🔍 Пошук по файлу або повідомленню…" oninput="filterTable()">
  {''.join(f'<button class="filter-btn" onclick="toggleSev(this,\'{s}\')">{SEVERITY_EMOJI[s]} {s}</button>' for s in SEVERITY_ORDER)}
  {''.join(f'<button class="filter-btn" onclick="toggleCat(this,\'{c}\')">[{c}]</button>' for c in sorted(CATEGORY_DESC.keys()))}
  <button class="filter-btn" onclick="clearFilters()">✕ Clear</button>
</div>

<table id="issues-table">
  <thead><tr><th>Severity</th><th>Cat</th><th>Файл:рядок</th><th>Повідомлення</th></tr></thead>
  <tbody>
  {rows_html}
  </tbody>
</table>

<script>
let activeSev = new Set();
let activeCat = new Set();

function filterTable() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('#issues-table tbody tr').forEach(tr => {{
    const sev = tr.querySelector('.badge')?.textContent?.trim();
    const cat = tr.querySelector('.cat')?.textContent?.trim();
    const txt = tr.textContent.toLowerCase();
    const sevOk = activeSev.size === 0 || activeSev.has(sev);
    const catOk = activeCat.size === 0 || activeCat.has(cat);
    const txtOk = !q || txt.includes(q);
    tr.classList.toggle('hidden', !(sevOk && catOk && txtOk));
  }});
}}

function toggleSev(btn, sev) {{
  btn.classList.toggle('active');
  if (activeSev.has(sev)) activeSev.delete(sev); else activeSev.add(sev);
  filterTable();
}}
function toggleCat(btn, cat) {{
  btn.classList.toggle('active');
  if (activeCat.has(cat)) activeCat.delete(cat); else activeCat.add(cat);
  filterTable();
}}
function clearFilters() {{
  activeSev.clear(); activeCat.clear();
  document.querySelectorAll('.filter-btn.active').forEach(b => b.classList.remove('active'));
  document.getElementById('search').value = '';
  filterTable();
}}
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML звіт: {out_path}  ({len(issues)} issues)")


# ── швидкий fix-mode ──────────────────────────────────────────────────────────

def print_fix_mode(report: AggregatedReport) -> None:
    """Показує тільки CRITICAL і HIGH з конкретними фіксами."""
    issues = report.sorted_issues("HIGH")
    if not issues:
        print("✅ Жодних CRITICAL або HIGH issues не знайдено!")
        return

    print(f"\n{'═'*72}")
    print(f"  🛠  FIX MODE — {len(issues)} issues потребують уваги")
    print(f"{'═'*72}\n")

    # Групуємо по файлу
    by_file: dict[str, list] = collections.defaultdict(list)
    for iss in issues:
        by_file[iss["file"]].append(iss)

    for fname, file_issues in sorted(by_file.items(), key=lambda x: -len(x[1])):
        print(f"\n📄 {fname}  ({len(file_issues)} issues)")
        for iss in file_issues:
            emoji = SEVERITY_EMOJI[iss["severity"]]
            print(f"  {emoji} L{iss['line']:4}  [{iss['category']}] {iss['message']}")
            if iss["fix_hint"]:
                print(f"         → {iss['fix_hint']}")


# ── JSON експорт ──────────────────────────────────────────────────────────────

def save_json_report(report: AggregatedReport, out_path: Path) -> None:
    data = {
        "generated_at": report.generated_at.isoformat(),
        "stats":        report.stats(),
        "issues": [
            {k: v for k, v in iss.items()}
            for iss in report.sorted_issues("INFO")
        ],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ JSON звіт: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DEAN Full Audit Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Приклади:
          python audit_run.py --root src
          python audit_run.py --root src --html --json
          python audit_run.py --root src --fix
          python audit_run.py --root src --severity HIGH --category LEAK,ASYNC,ML
          python audit_run.py --root src --top 20 --max-issues 1000
        """),
    )
    parser.add_argument("--root",       default=".",    help="Корінь проекту")
    parser.add_argument("--html",       action="store_true", help="Генерувати HTML звіт")
    parser.add_argument("--json",       action="store_true", help="Генерувати JSON звіт")
    parser.add_argument("--fix",        action="store_true", help="Fix mode: тільки CRITICAL+HIGH")
    parser.add_argument("--severity",   default="LOW",  help="Мін. рівень: CRITICAL/HIGH/MEDIUM/LOW/INFO")
    parser.add_argument("--category",   default="",     help="Фільтр категорій: ARC,DUP,LEAK,...")
    parser.add_argument("--top",        default=10, type=int, help="Топ N проблемних файлів")
    parser.add_argument("--max-issues", default=600,    type=int, help="Ліміт issues у консолі")
    parser.add_argument("--out-dir",    default=".",    help="Папка для HTML/JSON звітів")
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    runner_dir = Path(__file__).parent
    out_dir    = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"❌ Не існує: {root}")
        sys.exit(1)

    # Завантаження модулів
    audit_mod, logic_mod, engagement_mod = _find_audit_modules(runner_dir)

    # Запуск всіх аудиторів
    print(f"\n{'-'*72}")
    print(f"  STEP 1/3: Structural Audit (audit.py)")
    print(f"{'-'*72}")
    struct_auditor = audit_mod.ProjectAuditor(root)
    struct_result  = struct_auditor.audit()

    print(f"\n{'-'*72}")
    print(f"  STEP 2/3: Logic & ML Audit (audit_logic.py)")
    print(f"{'-'*72}")
    logic_auditor = logic_mod.DeepLogicAuditor(root)
    logic_result  = logic_auditor.audit()

    print(f"\n{'-'*72}")
    print(f"  STEP 3/3: Engagement & Coverage Audit (audit_engagement.py)")
    print(f"{'-'*72}")
    engagement_auditor = engagement_mod.EngagementAuditor(root)
    engagement_result  = engagement_auditor.audit()

    # Об'єднання
    report = AggregatedReport()
    report.add_from_result(struct_result, "structural")
    report.add_from_result(logic_result,  "logic")
    report.add_from_result(engagement_result, "engagement")

    # Фільтр категорій
    cats = [c.strip() for c in args.category.upper().split(",") if c.strip()] if args.category else []

    # Вивід
    if args.fix:
        print_fix_mode(report)
    else:
        print_console_report(
            report,
            min_severity=args.severity.upper(),
            categories=cats or None,
            show_fix=True,
            top_files=args.top,
            max_issues=args.max_issues,
        )

    # Генерація файлів
    ts = report.generated_at.strftime("%Y%m%d_%H%M%S")
    if args.html:
        generate_html_report(report, out_dir / f"audit_report_{ts}.html")
    if args.json:
        save_json_report(report, out_dir / f"audit_report_{ts}.json")

    # Exit code
    stats    = report.stats()
    critical = stats["by_severity"].get("CRITICAL", 0)
    high     = stats["by_severity"].get("HIGH", 0)

    print(f"\n  Health: {stats['grade']}  |  "
          f"Score: {stats['score']}  |  "
          f"Critical: {critical}  High: {high}\n")

    sys.exit(2 if critical > 0 else (1 if high > 0 else 0))


if __name__ == "__main__":
    main()
