"""One row per variable, and what it was measured to be.

The owner's idea, agreed 2026-08-29: mark the data rather than pick from it.
A role is not a property of a feature, it is a RESULT -- and a result nobody
wrote down gets produced again. The same 450 columns were screened four times
in one day because there was nowhere to look up what they had already been
found to be.

So this renders `docs/FEATURE_ROLES.md` from the CSV that
`leading_feature_report.py` leaves behind. Nothing here is maintained by
hand: rerun the measurement and the catalogue follows.

**A zero is an entry, not an absence.** "Measured, nothing there" and "never
measured" are different states, and only the first stops someone spending a
day rediscovering it. Every column that failed is listed with the reason it
failed.

    python scripts/diagnostics/feature_catalogue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SOURCE = Path("diagnostic_reports/feature_roles_1d.csv")
OUTPUT = Path("docs/FEATURE_ROLES.md")

#: What each verdict means for whoever reads the catalogue next.
MEANING = {
    "survives, worth testing":
        "не спростовано жодною перевіркою. Наступний крок — дохідність на "
        "одиницю ризику й беззбитковість проти витрат",
    "market-wide: use as interaction":
        "одне значення на дату для всіх імен. Ранжувати НЕ МОЖЕ за "
        "конструкцією; входить лише як взаємодія з чутливістю імені",
    "inside the noise for this many tests":
        "не проходить поправку Бенджаміні-Хохберга на кількість перевірок",
    "sign flipped out of sample":
        "напрямок не втримався поза вибіркою — монета",
    "labels the name, not the moment":
        "ранжує імена сталою величиною; прибери середнє тікера — і нічого не "
        "лишиться. Одна ставка, а не передбачення",
    "faded: held once, gone in the latest quarter":
        "ефект БУВ і згас. Пулована статистика його ще показує, останній "
        "квартал — ні",
    "reversed in the latest quarter":
        "останній квартал іде проти власного знаку",
    "too thin to judge":
        "покриття замале, щоб щось стверджувати",
    "not measurable":
        "кореляція не рахується: сталість або брак рядків",
}


def main() -> int:
    if not SOURCE.exists():
        print(f"no {SOURCE}; run leading_feature_report.py first.")
        return 1

    frame = pd.read_csv(SOURCE)
    stamp = frame["measured_on"].iloc[0]
    target = frame["target"].iloc[0]
    sealed = frame["sealed_from"].iloc[0]
    screened = int(frame["tests_screened"].iloc[0])
    counts = frame["verdict"].value_counts()

    lines = [
        "# Каталог ролей: що виміряно про кожну величину",
        "",
        "**Файл будується скриптом, не рукою.** Джерело — "
        "`diagnostic_reports/feature_roles_1d.csv`, який лишає по собі "
        "`leading_feature_report.py`. Щоб оновити — перезапустити вимір.",
        "",
        f"- виміряно: **{stamp}**",
        f"- ціль: `{target}`",
        f"- денний кадр, запечатано з **{sealed}**",
        f"- гіпотез перевірено: **{screened}**",
        f"- величин у каталозі: **{len(frame)}**",
        "",
        "## Скільки чого",
        "",
        "| роль | скільки | що це означає |",
        "|---|---|---|",
    ]
    for verdict, n in counts.items():
        lines.append(f"| {verdict} | {n} | {MEANING.get(verdict, '')} |")

    survivors = frame[frame["verdict"] == "survives, worth testing"]
    lines += ["", "## Що вижило", ""]
    if survivors.empty:
        lines += [
            "**Нічого.** Жодна з величин не проходить весь ланцюжок перевірок "
            "на цьому всесвіті.",
            "",
            "Це вимір, а не поразка: 110 великих американських імен "
            "відповіли, що тут нема чого ловити. Запис існує саме для того, "
            "щоб цього не перевіряли вдруге.",
        ]
    else:
        lines += ["| величина | ic/date | t | останній квартал | дат | покриття |",
                  "|---|---|---|---|---|---|"]
        for _, row in survivors.iterrows():
            lines.append(
                f"| `{row['feature']}` | {row['ic_daily']:+.4f} | "
                f"{row['t_daily']:+.2f} | {row['t_recent']:+.2f} | "
                f"{int(row['n_dates']):,} | {row['coverage']:.0%} |"
            )

    lines += ["", "## Найсильніші за кожною роллю", "",
              "Найбільший |ic_out| у групі — щоб було видно, чого саме "
              "коштувала кожна відмова.", ""]
    for verdict in counts.index:
        group = frame[frame["verdict"] == verdict].copy()
        group = group.reindex(group["ic_out"].abs()
                              .sort_values(ascending=False).index)
        lines += [f"**{verdict}** ({len(group)})", ""]
        for _, row in group.head(5).iterrows():
            daily = ("—" if pd.isna(row["ic_daily"])
                     else f"{row['ic_daily']:+.4f}")
            t = "—" if pd.isna(row["t_daily"]) else f"{row['t_daily']:+.2f}"
            lines.append(f"- `{row['feature']}` — ic_out {row['ic_out']:+.4f}, "
                         f"ic/date {daily}, t {t}, покриття {row['coverage']:.0%}")
        lines.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} from {len(frame)} measured features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
