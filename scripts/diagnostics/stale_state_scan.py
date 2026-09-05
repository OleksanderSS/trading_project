"""Does the recorded state still match the evidence?

The three tables only grow, and a row is written once and read rarely. So the
failure mode is not a wrong entry -- it is a RIGHT entry whose state was never
updated after the thing was measured, fixed, or refuted. Read from the top,
such a row is indistinguishable from outstanding work, and the count of "what
is left" quietly stops meaning anything.

There is already a checker for the ARITHMETIC: `test_register_counter_matches
_the_table.py` fails when the printed counters disagree with the rows. Nothing
checked the rows against the EVIDENCE. That is this.

Each rule below exists because a real row was found in that shape on
2026-09-04:

    A  a ROADMAP box is unticked while its own text says the work was done
       ("`state_FRED_*` рахувати один раз на дату (#145) -- виправлено й ...")

    B  a ROADMAP box is unticked and cites a register entry that is closed
       (three of the four items in the gate section cite #188, #191, #192,
       all of them `закрито`)

    C  a ROADMAP box holds no task at all -- a rule, a finding, or a line
       admitting it is a duplicate ("Для поіменної колонки поточна поведінка
       правильна.") Counted as outstanding work, it inflates what is left.

    D  a REGISTER row is `?` while its own text carries a verdict
       ("**ЗМІРЯНО ТОГО Ж ДНЯ, І КАНДИДАТ ФАКТИЧНО ЗАКРИТИЙ.**" -- #175)

    E  a REGISTER row is `?` or `відкрито` while a LATER row that cites it is
       closed

    F  a REGISTER row is `?` while the fix it describes is pinned by a
       contract test that exists on disk

    G  a REGISTER row is CLOSED while its own closing sentences describe the
       fix in the future tense and never say it was made. This is the
       expensive direction: #188 ("щабель 5 обирає найКОРЕЛЬОВАНІШУ колонку")
       ends with "**Виправлення:** щабель має брати 5-10 найкорельованіших
       кандидатів..." and `base_trainer._score_single_feature_baseline` still
       does `corr.idxmax()`. A gate rung read as verified for four days
       because one word in one cell said so.

    H  a CLOSED row declares a part of itself unfixed, ANYWHERE in the row.
       Rule G reads only the last 420 characters, because that is where a
       verdict sits -- and it therefore cannot see an admission made in the
       middle. #264 closes with a proper result and says, two thirds of the
       way through, "**Друга знахідка, НЕ виправлена:** `apply_seal` ... має
       нуль викликачів". That is a live defect inside a closed entry, and it
       survived every pass rule G made. Found on 2026-09-05 by reading the
       row a person had chosen from a list, which is the method rule H
       automates. That remainder was then fixed the same day, and the row
       now records the result -- so rule H no longer fires on #264, which
       is what a rule doing its job looks like.

Nothing here is proof. Like `row_order_dependency_scan.py`, it does not show
that the tables are clean -- it shows the places worth a human minute, and
refuses to guess on their behalf.

    python scripts/diagnostics/stale_state_scan.py
    python scripts/diagnostics/stale_state_scan.py --rule D
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

# The report is Ukrainian and the Windows console is cp1252 by default, so
# printing it raised UnicodeEncodeError anywhere PYTHONIOENCODING was not set
# by hand -- including from pytest, which is where it has to work. A tool that
# runs only when invoked a particular way is not wired in.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "REGISTER.md"
ARCHIVE = ROOT / "docs" / "AUDIT_HISTORY.md"
ROADMAP = ROOT / "docs" / "ROADMAP.md"
CONTRACTS = ROOT / "tests" / "contracts"

#: Tests that guard the RECORDS rather than the code, and therefore quote row
#: numbers as data.
#:
#: Rule F reads "a contract test names this open row" as evidence the row may
#: have been fixed -- somebody wrote a ratchet for it. That inference is void
#: for a test whose subject IS the register: on 2026-09-05 rule F reported #48
#: because a comment in the counter test explains why #48 stopped being `?`,
#: and #202 because the state scanner's own test lists the rules. Neither is a
#: ratchet on anything the rows describe.
#:
#: Named individually rather than pattern-matched, so a new bookkeeping test
#: has to be added here deliberately and cannot quietly blind the rule.
BOOKKEEPING_TESTS = frozenset({
    "test_register_counter_matches_the_table.py",
    "test_the_recorded_state_matches_the_evidence.py",
    "test_the_archive_keeps_what_the_register_learned.py",
    "test_an_unchanged_observation_is_not_a_new_row.py",
})

#: Only these are entry states. A digit-leading table row that carries
#: anything else in its second cell belongs to some other table in the same
#: file, and must not be read as an entry.
STATES = {"закрито", "відкрито", "знято", "неперевірюване", "?"}

CLOSED_STATES = ("закрито", "знято")
UNSETTLED_STATES = ("?", "відкрито")

#: Phrases that assert the work is finished. Deliberately narrow: "виміряно"
#: alone appears in rows that then say the measurement changed nothing, so it
#: is not on this list.
DONE_PHRASES = (
    "виправлено й",
    "виправлено та",
    "вже виправлено",
    "ФАКТИЧНО ЗАКРИТИЙ",
    "фактично закритий",
    "ВИКОНАНО",
    "зроблено й",
    "закрито цим",
    # "більше не мовчить", "більше не падає" -- a behaviour that was fixed.
    # NOT "більше немає", which is a plain statement about the world and
    # tripped rule D on #149 ("того кадру в батчі більше немає"). A phrase
    # that fires on ordinary prose is how a check earns the reputation that
    # gets it switched off.
    "більше не ",
)

#: A line that is a rule or a finding rather than a task. An imperative in
#: this project starts with a verb; these openings never do.
NOT_A_TASK = (
    "[дублікат",
    "Отже",
    "Спільне",
    "Це ПРАВИЛО",
    "це ПРАВИЛО",
    "Для **поіменної**",
    "Ці рядки",
)

RULE_NAMES = {
    "A": "ROADMAP: unticked, but the line says it was done",
    "B": "ROADMAP: unticked, but the register entry it cites is closed",
    "C": "ROADMAP: not a task -- a rule, a finding, or a self-declared duplicate",
    # The label said "state '?'" while the code has always taken every
    # UNSETTLED state. Corrected 2026-09-05: a rule whose name describes a
    # narrower check than it performs sends the reader looking for a row that
    # is not there -- the same defect as a docstring that promises a failure
    # and delivers a skip (#278).
    "D": "REGISTER: unsettled while the row declares a part of itself done",
    "E": "REGISTER: unsettled while a later closed row cites it",
    "F": "REGISTER: unsettled while a contract test names it",
    "G": "REGISTER: closed, but the closing text describes a fix in the future",
    "H": "REGISTER: closed, but the row declares a part of itself unfixed",
}

#: Written where a row admits, mid-text, that something in it was not done.
#: Distinct from FIX_PROPOSED: those phrases describe what a fix WOULD be,
#: these announce that a named part of the entry remains open. A closed row
#: may legitimately contain one -- #264 does -- but it must then be visible,
#: which is the whole difference between a recorded remainder and a lost one.
#: Only phrases that announce a REMAINDER. "нуль викликачів" and "мертвий
#: код" were on this list for ten minutes and came off it: they are DIAGNOSIS
#: vocabulary, and #170 uses "нуль викликачів" to describe the finding it then
#: fixed in the same row. A rule that cannot tell "we found zero callers and
#: wired it" from "it still has zero callers" reports the fix as the fault --
#: the same lesson as "більше не " against "більше немає", one rule earlier.
DECLARED_UNFIXED = (
    "НЕ виправлена", "НЕ виправлено", "не виправлена", "не виправлено",
    "лишається відкрит", "лишилось відкрит", "лишається за власником",
)

#: How much of the sentence around the phrase to quote back.
#:
#: Naming the phrase alone was not enough to act on: five rows came back as
#: `declares "не виправлено"` and each still cost opening a body of two to
#: three thousand characters to learn WHAT was not done. Two of the five
#: turned out to be already fixed -- #110's absolute velocity gate now prefers
#: `context_velocity_rank`, #260's two skips were repaired when
#: `test_financial_math_correctness` was rewritten -- and neither could be
#: told from the list. A reading list that does not say what to read is a
#: second list to work through.
REMAINDER_QUOTE = 240


def _remainder(row: str, phrase: str) -> str:
    """The sentence the phrase sits in, so the list can be acted on directly."""
    start = row.index(phrase)
    # Back up to the start of the clause, not the row: the phrase usually
    # follows a bold lead-in like "**Друга знахідка, не виправлена:**".
    #
    # Bold markers alternate, so the nearest one going backwards is as often a
    # CLOSING marker as an opening one -- #260 quoted back as "** — знову
    # форма..." and cut off the words that said what was not done. Counting
    # them decides which: an even index in the list opens, an odd one closes.
    markers, at = [], row.find("**")
    while at != -1 and at < start:
        markers.append(at)
        at = row.find("**", at + 2)
    if markers and len(markers) % 2 == 0:   # the last one closed a span
        markers.pop()
    opening = markers[-1] if markers else -1
    if opening == -1 or start - opening > 200:
        # No lead-in within reach. Cut at a clause boundary rather than a
        # character count: `start - 60` landed mid-word on #260 ("ву форма
        # «конфіг оголошує…»") and hid the very words that named the defect.
        window = row[max(0, start - 200):start]
        cut = max(window.rfind(". "), window.rfind("— "), window.rfind("; "))
        opening = (max(0, start - 200) + cut + 2) if cut != -1 else \
            max(0, start - 200)
    quoted = " ".join(row[opening:start + REMAINDER_QUOTE].split())
    ellipsis = "…" if start + REMAINDER_QUOTE < len(row) else ""
    return f"declares a remainder: {quoted}{ellipsis}"


#: Written where a fix is being PROPOSED. Read only in the tail of the row,
#: because a row that proposes a fix and then reports making it ends with the
#: report.
FIX_PROPOSED = (
    "Виправлення:", "Виправлення —", "Виправлення -",
    "Перевірка до виправлення", "Наступний крок",
    "Правка:", "Правка —", "мусить", "Треба ", "треба ",
    "має брати", "має бути", "має перелічувати", "має мати",
    "не зроблен", "не реалізован", "Що з цим робити",
)

#: Written where a fix is being REPORTED. Any of these in the tail clears the
#: row: the entry says the work happened, and this scanner does not second
#: guess a report -- rules E and F do that from the other side.
FIX_REPORTED = (
    # Matched on the STEM, not on a phrase: the first version listed
    # "виправлено й" and "тестами", so a row ending "виправлено в межах однієї
    # правки" and one ending "6 тестів" both read as unfinished. A rule with
    # false positives gets switched off, which is #181 and the `|| true` in
    # ci.yml.
    "виправлено", "ВИПРАВЛЕНО", "ЗАКРИТО", "закрито й", "Зроблено",
    "контрактн", "тест", "закрито цим", "перевірено",
    # The canonical form, added 04.09 at the owner's instruction: a settled
    # row ENDS with what came out, stated as an outcome and not as a plan.
    # Everything above is the older prose this rule had to tolerate; new rows
    # use the marker, and then the check is one word long.
    "**РЕЗУЛЬТАТ", "**Результат",
    # A closed row MAY end by proposing a fix -- if it says where that fix is
    # tracked. That is the whole difference between #191, which handed its
    # remainder to #199, and #188, which named a successor for nobody and sat
    # four days looking finished.
    "продовження записане як",
)

#: How much of the end of a row counts as "the closing text". A register row
#: runs to two thousand characters and the verdict is always last.
TAIL = 420


def _read(path: Path) -> str:
    return io.open(path, encoding="utf-8").read()


def register_rows() -> dict[int, tuple[str, str]]:
    """Every entry, wherever its body now lives.

    Settled entries were moved to AUDIT_HISTORY.md on 04.09 because the
    register had stopped being readable in one pass -- which is what let four
    wrong `закрито` marks sit unnoticed. The register keeps a one-line index
    of them. Rules D, E and G read BODIES, so a scanner that read only the
    register would go quiet on 253 entries the moment they were archived, and
    a check that silently stops checking is the defect this whole file exists
    to catch (#202).

    The archive row wins where both files carry the same number: the index
    line is a summary, the archived row is the entry.
    """
    rows: dict[int, tuple[str, str]] = {}
    pattern = re.compile(r"\|\s*(\d+)\s*\|\s*([^|]*?)\s*\|")
    for path in (REGISTER, ARCHIVE):
        if not path.exists():
            continue
        for line in _read(path).splitlines():
            match = pattern.match(line)
            if not match:
                continue
            number = int(match.group(1))
            state = match.group(2).strip()
            if state not in STATES:
                continue
            if path is ARCHIVE or number not in rows:
                rows[number] = (state, line)
    return rows


def roadmap_items() -> list[tuple[int, str]]:
    return [
        (number, line.strip())
        for number, line in enumerate(_read(ROADMAP).splitlines(), start=1)
        if re.match(r"^\s*[-*]?\s*\[ \]", line)
    ]


def _cited(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"#(\d{1,3})\b", text)}


#: Written inside the citation's own parentheses when the writer has checked
#: WHY a settled entry sits beside an open task. Deliberately demanding: it
#: must appear in the same bracket as the `#N`, so a stray "закрито" elsewhere
#: in a long line does not silence the rule. An unmarked discrepancy is still
#: reported; a marked one has been read by a person, and that is the whole
#: difference this repository keeps rediscovering.
EXAMINED = ("закрит", "закрыт")

#: Written by a row that has finished a PART of itself and stays open on
#: purpose. Rule D exists to catch a row that says "done" while sitting open,
#: and #169 is the case where that is correct rather than stale: its free half
#: is complete and measured (Р46, Р47), while the paid half is a standing
#: owner decision, so closing it would hide a live constraint on what the
#: project may claim.
#:
#: The phrase is demanding on purpose -- a row must say, in these words, that
#: it stays open and why. Silence is what rule D reports; an explicit,
#: written-out reason is a person having decided.
STAYS_OPEN = (
    "ЛИШАЄТЬСЯ ВІДКРИТИМ",
    "ЛИШАЄТЬСЯ ВІДКРИТОЮ",
    "СВІДОМО ВІДКРИТИЙ",
)


def _examined(text: str) -> bool:
    for bracket in re.findall(r"\(([^()]*#\d{1,3}[^()]*)\)", text):
        if any(word in bracket for word in EXAMINED):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", default=None, choices=sorted(RULE_NAMES))
    args = parser.parse_args()

    rows = register_rows()
    items = roadmap_items()
    contract_text = "\n".join(
        _read(path) for path in sorted(CONTRACTS.glob("test_*.py"))
        if path.name not in BOOKKEEPING_TESTS
    )

    findings: dict[str, list[str]] = {key: [] for key in RULE_NAMES}

    for line_number, text in items:
        if any(phrase in text for phrase in DONE_PHRASES):
            findings["A"].append(f"ROADMAP:{line_number}  {text[:110]}")
        closed = sorted(n for n in _cited(text)
                        if n in rows and rows[n][0] in CLOSED_STATES)
        if closed and _examined(text):
            # The writer has stated, on the line, why the cited entry is
            # settled while the task is not. Checked 2026-09-05 on all four
            # hits this rule then had, one entry at a time: #123 closed as a
            # DIAGNOSIS with "зміна НЕ зроблена", #122 closed by fixing the
            # two causes rather than by building checkpoints, #138 closed by a
            # guard instead of by appointing one writer. All four items were
            # honestly open, and the rule would have gone on naming them for
            # ever -- which is how a check that always fires gets switched
            # off, the lesson of `|| true` in ci.yml (#181).
            #
            # So the discrepancy has to be MARKED, not merely true: the same
            # invariant as "a default must say it was a default".
            closed = []
        if closed:
            findings["B"].append(
                f"ROADMAP:{line_number}  cites {closed} (closed)  {text[:80]}"
            )
        stripped = re.sub(r"^\s*[-*]?\s*\[ \]\s*\**", "", text)
        if any(stripped.startswith(opening.lstrip("*")) or opening in text
               for opening in NOT_A_TASK):
            findings["C"].append(f"ROADMAP:{line_number}  {text[:110]}")

    for number, (state, line) in sorted(rows.items()):
        if state not in UNSETTLED_STATES:
            continue
        if any(phrase in line for phrase in DONE_PHRASES) \
                and not any(phrase in line for phrase in STAYS_OPEN):
            findings["D"].append(f"REGISTER #{number} [{state}]  "
                                 f"{_first_phrase(line)}")
        citing = [
            other for other, (other_state, other_line) in rows.items()
            if other > number and other_state in CLOSED_STATES
            and number in _cited(other_line)
        ]
        if citing:
            findings["E"].append(
                f"REGISTER #{number} [{state}]  cited by closed "
                f"{sorted(citing)}"
            )
        if f"#{number}" in contract_text:
            findings["F"].append(
                f"REGISTER #{number} [{state}]  named by a contract test"
            )

    for number, (state, line) in sorted(rows.items()):
        if state not in CLOSED_STATES:
            continue
        for phrase in DECLARED_UNFIXED:
            if phrase in line:
                findings["H"].append(
                    f"REGISTER #{number} [{state}]  {_remainder(line, phrase)}"
                )
                break

    for number, (state, line) in sorted(rows.items()):
        if state != "закрито":
            continue
        tail = line.rstrip()[-TAIL:]
        if (any(phrase in tail for phrase in FIX_PROPOSED)
                and not any(phrase in tail for phrase in FIX_REPORTED)):
            findings["G"].append(
                f"REGISTER #{number} [закрито]  ...{tail[-150:]}"
            )

    total = 0
    for key in sorted(RULE_NAMES):
        if args.rule and key != args.rule:
            continue
        hits = findings[key]
        total += len(hits)
        print(f"=== {key}. {RULE_NAMES[key]} -- {len(hits)}")
        for hit in hits:
            print(f"  {hit}")
        print()

    print(f"{total} rows worth a human minute. None of this is proof: the "
          f"scanner\nshows shapes, and the state is still a judgement.")
    return 0


def _first_phrase(line: str) -> str:
    parts = [part.strip() for part in line.split("|")]
    claim = parts[5] if len(parts) > 5 else line
    return claim[:100]


if __name__ == "__main__":
    raise SystemExit(main())
