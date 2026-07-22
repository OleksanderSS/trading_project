from __future__ import annotations

from pathlib import Path
from typing import Any


def _connect(db_path: str | Path = "data/trading_data.duckdb"):
    import duckdb
    return duckdb.connect(str(db_path), read_only=True)


def get_table_info() -> dict[str, dict[str, Any]]:
    con = _connect()
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for (table_name,) in tables:
        row_count = con.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
        cols = con.execute(
            f"SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name='{table_name}' ORDER BY ordinal_position"
        ).fetchall()
        columns = [{"name": c[0], "type": c[1], "nullable": c[2] == "YES"} for c in cols]
        info: dict[str, Any] = {"rows": row_count, "columns": columns}

        # Try date range if a datetime column exists
        datetime_cols = [c for c in cols if "timestamp" in c[1].lower() or "datetime" in c[1].lower() or "date" in c[1].lower()]
        if datetime_cols:
            try:
                dc = datetime_cols[0][0]
                rng = con.execute(f'SELECT min("{dc}"), max("{dc}") FROM "{table_name}"').fetchone()
                if rng[0]:
                    info["date_range"] = {"min": str(rng[0]), "max": str(rng[1]), "column": dc}
            except Exception:
                pass
        result[table_name] = info
    con.close()
    return result


def print_table_info(info: dict[str, dict[str, Any]]) -> None:
    print(f"{'Table':35} {'Rows':>10} {'Columns':>8} {'Date range':40}")
    print("-" * 95)
    for tname, tinfo in sorted(info.items(), key=lambda x: -x[1]["rows"]):
        rows = f"{tinfo['rows']:,}"
        cols = len(tinfo["columns"])
        dr = ""
        if "date_range" in tinfo:
            dr = f"{tinfo['date_range']['min'][:10]} .. {tinfo['date_range']['max'][:10]}"
        print(f"  {tname:33} {rows:>10} {cols:>8}  {dr:40}")


def search_columns(query: str) -> list[dict[str, Any]]:
    con = _connect()
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    results = []
    for (table_name,) in tables:
        cols = con.execute(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}' AND LOWER(column_name) LIKE '%{query.lower()}%'"
        ).fetchall()
        for c in cols:
            results.append({"table": table_name, "column": c[0], "type": c[1]})
    con.close()
    return results


def data_quality_report() -> dict[str, dict[str, Any]]:
    con = _connect()
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for (table_name,) in tables:
        try:
            col_rows = con.execute(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}' ORDER BY ordinal_position"
            ).fetchall()
            col_names = [r[0] for r in col_rows]
            null_ratios: dict[str, float] = {}
            null_cols = 0
            total = con.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
            for cn in col_names[:20]:
                row = con.execute(f'SELECT count(*) FROM "{table_name}" WHERE "{cn}" IS NULL').fetchone()
                if total > 0:
                    ratio = row[0] / total
                    if ratio > 0:
                        null_ratios[cn] = round(ratio, 4)
                        if ratio > 0.5:
                            null_cols += 1

            dup_count = 0
            if total > 0 and col_names:
                try:
                    cols_expr = ", ".join(f'"{r[0]}"' for r in col_rows[:3])
                    dup_count = con.execute(
                        f'SELECT count(*) - count(DISTINCT {cols_expr}) FROM "{table_name}"'
                    ).fetchone()[0]
                except Exception:
                    dup_count = -1

            date_gaps = ""
            for r in col_rows:
                if any(k in r[1].lower() for k in ("timestamp", "datetime", "date")):
                    try:
                        dc = r[0]
                        rng = con.execute(f'SELECT min("{dc}"), max("{dc}") FROM "{table_name}"').fetchone()
                        if rng[0] and rng[1]:
                            from datetime import UTC, datetime
                            mx = rng[1]
                            days = (datetime.now(UTC) - mx).days if isinstance(mx, datetime) and mx.tzinfo else 0
                            date_gaps = f"{str(rng[0])[:10]} .. {str(rng[1])[:10]} (latest: {days}d ago)"
                            break
                    except Exception:
                        continue

            result[table_name] = {
                "rows": total,
                "columns": len(col_names),
                "null_ratios": null_ratios,
                "high_null_cols": null_cols,
                "duplicate_estimate": dup_count,
                "date_gaps": date_gaps,
            }
        except Exception as e:
            result[table_name] = {"error": str(e)}
    con.close()
    return result


def print_dq_report(report: dict[str, dict[str, Any]]) -> None:
    for tname, tinfo in sorted(report.items(), key=lambda x: -x[1].get("rows", 0)):
        if "error" in tinfo:
            print(f"  {tname:35} ERROR: {tinfo['error']}")
            continue
        rows = f"{tinfo['rows']:,}"
        cols = tinfo["columns"]
        nulls = len(tinfo["null_ratios"])
        high = tinfo["high_null_cols"]
        dups = tinfo["duplicate_estimate"]
        gaps = tinfo["date_gaps"]
        flags = []
        if high > 0:
            flags.append(f"{high} cols >50% null")
        if dups > 0:
            flags.append(f"{dups:,} dup rows")
        flag_str = " | ".join(flags) if flags else "clean"
        print(f"  {tname:35} {rows:>10} {cols:>3} cols  nulls={nulls}  {flag_str}")
        if nulls > 0 and nulls <= 5:
            for cn, r in tinfo["null_ratios"].items():
                print(f"    - {cn}: {r:.1%} null")
        if gaps:
            print(f"    dates: {gaps}")


__all__ = ["get_table_info", "print_table_info", "search_columns", "data_quality_report", "print_dq_report"]
