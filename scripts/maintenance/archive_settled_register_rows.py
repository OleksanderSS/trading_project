"""Move settled entries out of the register, and keep what they taught.

REGISTER.md reached 559 KiB in 277 rows. Ninety percent of those bytes are
entries already closed, withdrawn or ruled unverifiable. The file stopped
being readable in one pass some weeks ago, and that is not a cosmetic problem:
it is WHY states rot. Nobody reaches row #142, so nobody notices that #142
says `закрито` while its fix was never made. Four such rows were found on
2026-09-04, one of them a gate rung.

WHAT THIS IS NOT. It is not deletion, and not "archive and forget". The owner
named the requirement exactly: a settled item must stay REMEMBERED -- what was
asked and what came out -- so the same ground is not walked twice.

So the register keeps, for every settled entry, one line:

    | N | state | type | the question, verbatim | the result |

and the full body -- two thousand characters of measurement, correction and
argument -- moves to `docs/AUDIT_HISTORY.md`. The numbers never change, so
every `#N` reference in code, commits and the other documents keeps pointing
at the same entry.

The result column is NEVER composed here. It is the sentence after the row's
own `**РЕЗУЛЬТАТ:**`, or failing that the row's own last sentence, quoted. A
summary written by this script would be text nobody measured, in the file that
exists to stop exactly that. Rule G of `stale_state_scan.py` already
guarantees a closed row's last sentence is a result rather than a plan.

IDEMPOTENT, and that took a defect to get right. The first version read only
the register and rebuilt the archive from what it found. Run a second time it
would have taken the one-line INDEX rows as the entries and rewritten the
archive with them -- destroying 253 full bodies silently, while the
verification step confirmed each row matched itself. Settled rows are now read
back from the archive, and a number present in both files is an error rather
than a guess.

    python scripts/maintenance/archive_settled_register_rows.py --dry-run
    python scripts/maintenance/archive_settled_register_rows.py
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "REGISTER.md"
ARCHIVE = ROOT / "docs" / "AUDIT_HISTORY.md"

SETTLED = {"закрито", "знято", "неперевірюване"}
ID_ROW = re.compile(r"^\|\s*(\d+)\s*\|")

#: Where the index begins in the register, and where the bodies begin in the
#: archive. Everything after each is regenerated on every run.
INDEX_HEADING = "## Покажчик вирішеного"
ARCHIVE_HEADING = "## Повні записи реєстру"

#: The paragraph the rebuild writes under the archive's heading.
#:
#: Kept as a constant because it is read by TWO things: the rebuild writes it,
#: and the stray-text guard must not mistake it for somebody's lost prose. The
#: first version of that guard had the lines only in the writer, so it fired on
#: the archiver's own output -- a check reporting the thing it had just written.
ARCHIVE_PREAMBLE = [
    "Перенесено з `REGISTER.md`, коли той перестав читатися за один прохід",
    "(559 КіБ, 90% байтів — вирішене), бо саме нечитаність і є причиною, з",
    "якої стани гнили: до рядка №142 ніхто не доходив.",
    "",
    "Нічого не видалено й не переписано — рядки ті самі, включно з",
    "поправками до моїх власних перших описів. Коротка лінія",
    "«питання → результат» лишилась у реєстрі.",
]

#: Long enough to recognise an entry without opening the archive; short enough
#: that 253 of them stay a list rather than a second register.
QUESTION_CHARS = 200
RESULT_CHARS = 260

NO_RESULT = "порожній запис — повний текст в архіві"


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.rstrip().rstrip("|").split("|")[1:]]


def _body(line: str) -> str:
    """The entry's prose, whichever cell it ended up in.

    The register's table is ragged -- 41 rows carry a different number of
    columns than the nominal eight, because the column was introduced late and
    pipes appear inside code spans. Slicing a fixed offset returned an empty
    string on 24 of them, and an empty result column would have read as "this
    entry produced nothing".
    """
    tail = "|".join(line.rstrip().rstrip("|").split("|")[6:]).strip()
    if tail:
        return tail
    cells = _cells(line)
    for cell in reversed(cells):
        if len(cell) > 40:
            return cell
    return ""


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip(" ,;.") + "…"


def _result_of(body: str) -> str:
    for marker in ("**РЕЗУЛЬТАТ", "**Результат"):
        at = body.find(marker)
        if at != -1:
            text = body[at:].lstrip("*")
            text = re.sub(r"^(РЕЗУЛЬТАТ|Результат)[^:]*:", "", text)
            # The marker is bold, so the closing `**` sits AFTER the colon.
            return _clip(text.lstrip("* "), RESULT_CHARS)

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body)
                 if part.strip()]
    if not sentences:
        return NO_RESULT
    tail = sentences[-1]
    if len(tail) < 60 and len(sentences) > 1:
        tail = sentences[-2] + " " + tail
    return _clip(tail, RESULT_CHARS)


def _split(text: str, heading: str) -> tuple[str, list[str]]:
    """Everything before `heading`, and the table rows after it."""
    at = text.find("\n" + heading)
    if at == -1:
        return text, []
    rows = [line.rstrip() for line in text[at:].splitlines()
            if ID_ROW.match(line)]
    return text[:at], rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    register_text = io.open(REGISTER, encoding="utf-8").read()
    archive_text = io.open(ARCHIVE, encoding="utf-8").read()

    archive_head, archived_rows = _split(archive_text, ARCHIVE_HEADING)

    # Prose after the heading is DESTROYED by the rebuild below, because the
    # archive is regenerated from its table rows. Found 2026-09-06 by moving a
    # journal section here by hand: the archiver reported "written" and the
    # section was gone -- the same shape as #278, where an index-section row
    # was eaten, and the same remedy. Anything after the heading that is not a
    # row stops the run instead of vanishing.
    at = archive_text.find("\n" + ARCHIVE_HEADING)
    if at != -1:
        stray = [line for line in archive_text[at:].splitlines()
                 if line.strip()
                 and not ID_ROW.match(line)
                 and not line.startswith(ARCHIVE_HEADING)
                 and not line.startswith("|")
                 and not line.startswith("#")
                 and line.strip() not in {p.strip() for p in ARCHIVE_PREAMBLE}]
        if stray:
            raise SystemExit(
                "the archive holds text after its heading that is not a table "
                "row, and rebuilding would delete it. Move it ABOVE the "
                "heading (the head is preserved verbatim):\n"
                + "\n".join(f"  {line[:120]}" for line in stray[:8])
            )
    settled: dict[int, str] = {int(ID_ROW.match(row).group(1)): row
                               for row in archived_rows}
    already = len(settled)

    register_head, index_rows = _split(register_text, INDEX_HEADING)

    # A row written into the index section by hand is DESTROYED by the rebuild
    # below, because the section is regenerated from the archive. The section's
    # own preamble says editing it makes no sense -- but "no sense" and
    # "silently deleted" are different promises, and on 2026-09-05 the second
    # one ate a freshly written #278: the archiver reported "0 newly archived"
    # and wrote a register without it. Nothing said a row had gone.
    #
    # So the numbers in the index are compared against the archive before
    # anything is overwritten, and an unknown one stops the run with its own
    # text in hand.
    strangers = [row for row in index_rows
                 if int(ID_ROW.match(row).group(1)) not in settled]
    if strangers:
        raise SystemExit(
            "the index section holds rows that are not in the archive, and "
            "rebuilding it would delete them. Move them into the live table "
            "above the index heading and run again:\n"
            + "\n".join(f"  {row[:160]}…" for row in strangers)
        )
    live: list[str] = []
    for line in register_head.splitlines():
        match = ID_ROW.match(line)
        if not match:
            live.append(line)
            continue
        number = int(match.group(1))
        cells = _cells(line)
        state = cells[1] if len(cells) > 1 else "?"
        if state not in SETTLED:
            live.append(line)
            continue
        if number in settled:
            raise SystemExit(
                f"#{number} is live in the register AND already archived; "
                "refusing to guess which one is the entry"
            )
        settled[number] = line.rstrip()

    # The header says its counts are "рахується скриптом із самої таблиці, не
    # друкується руками". That sentence was FALSE until 2026-09-05: no script
    # wrote it. `test_register_counter_matches_the_table.py` failed when the
    # numbers diverged -- so the mechanism was a person remembering, with a
    # test to catch them forgetting, described as automation.
    #
    # A document that claims a mechanism it does not have is the same family
    # as a config declaring a module that is not there. Writing the counts
    # here makes the sentence true, and this script already parses every row.
    all_rows = list(settled.values()) + [l for l in live if ID_ROW.match(l)]
    tally: dict[str, int] = {}
    for row in all_rows:
        cells = _cells(row)
        # cells[0] is the number: `_cells` already drops the leading empty
        # field, so the state is at 1, not at the 2 the test's own `split("|")`
        # uses. Reading it at 2 wrote a header of five zeros before this line
        # was corrected -- and the run reported "written" while doing it.
        state = cells[1] if len(cells) > 1 else "?"
        tally[state] = tally.get(state, 0) + 1
    # A count nobody checked is what the header used to be. Reading the state
    # from the wrong cell produced five zeros and a cheerful "written" on the
    # first run of this block, so the numbers have to survive their own
    # arithmetic before they are allowed into the file.
    known = {"закрито", "відкрито", "знято", "неперевірюване", "?"}
    unknown_states = {state for state in tally if state not in known}
    if unknown_states or sum(tally.values()) != len(all_rows):
        raise SystemExit(
            f"refusing to write a header nobody can reproduce: counted "
            f"{sum(tally.values())} states across {len(all_rows)} rows"
            + (f", and {sorted(unknown_states)[:5]} are not states at all"
               if unknown_states else "")
            + ". The state cell moved, or the table shape changed."
        )

    counted = (
        f"| **стан записів** | закрито **{tally.get('закрито', 0)}** · "
        f"відкрито **{tally.get('відкрито', 0)}** · "
        f"знято **{tally.get('знято', 0)}** · "
        f"неперевірюване **{tally.get('неперевірюване', 0)}** · "
        f"**стан не записаний: {tally.get('?', 0)}** з {len(all_rows)}. "
        f"Рахується цим скриптом із самої таблиці, не друкується руками — "
        f"`tests/contracts/test_register_counter_matches_the_table.py` падає, "
        f"коли розходиться |"
    )
    live = [counted if line.startswith("| **стан записів** |") else line
            for line in live]

    live_count = sum(1 for line in live if ID_ROW.match(line))
    print(f"{len(settled) - already} newly archived, {already} already there, "
          f"{live_count} rows stay live")

    index_lines, without = [], 0
    for number, row in sorted(settled.items(), reverse=True):
        cells = _cells(row)
        state = cells[1] if len(cells) > 1 else "?"
        kind = cells[2] if len(cells) > 2 else ""
        question = _clip(cells[4] if len(cells) > 4 else "", QUESTION_CHARS)
        result = _result_of(_body(row))
        without += result == NO_RESULT
        index_lines.append(
            f"| {number} | {state} | {kind} | {question} | {result} |")
    print(f"  {len(settled) - without} carry a quotable result, {without} do not")

    new_register = "\n".join(live).rstrip() + "\n" + "\n".join([
        "",
        INDEX_HEADING,
        "",
        "Одна лінія на вирішений запис: **питання і результат лишаються тут**,",
        "повний текст — у [`AUDIT_HISTORY.md`](AUDIT_HISTORY.md). Номери не",
        "змінюються, тож усі посилання `#N` далі ведуть туди ж.",
        "",
        "Стовпець «що вийшло» — це **цитата з самого запису** (речення після",
        "`**РЕЗУЛЬТАТ:**`, інакше останнє речення), а не переказ: переказ був би",
        "текстом, якого ніхто не міряв, у файлі, заведеному саме проти цього.",
        "",
        "Перебудовується скриптом",
        "`scripts/maintenance/archive_settled_register_rows.py` — правити руками",
        "цю секцію немає сенсу, правиться запис в архіві.",
        "",
        "| # | стан | тип | що питали | що вийшло |",
        "|---|---|---|---|---|",
        *index_lines,
        "",
    ])

    new_archive = archive_head.rstrip() + "\n\n" + "\n".join([
        ARCHIVE_HEADING,
        "",
        *ARCHIVE_PREAMBLE,
        "",
        "| # | стан | тип | звідки | що знайдено | подробиці |",
        "|---|---|---|---|---|---|",
        *[row for _, row in sorted(settled.items(), reverse=True)],
        "",
    ])

    for number, row in settled.items():
        if row not in new_archive:
            raise SystemExit(f"#{number} did not survive the move; nothing written")
        if f"| {number} | " not in new_register:
            raise SystemExit(f"#{number} lost its index line; nothing written")

    def kib(value: str) -> str:
        return f"{len(value.encode('utf-8')) / 1024:,.0f} KiB"

    print(f"  register {kib(register_text)} -> {kib(new_register)}")
    print(f"  archive  {kib(archive_text)} -> {kib(new_archive)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    io.open(REGISTER, "w", encoding="utf-8", newline="\n").write(new_register)
    io.open(ARCHIVE, "w", encoding="utf-8", newline="\n").write(new_archive)
    print("\nwritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
