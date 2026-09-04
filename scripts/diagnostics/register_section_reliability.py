"""Can the register's own sections be trusted to say an old entry's state?

141 of 255 entries predate the `стан` column and carry `?`. The register's own
test forbids guessing them -- "guessing the state of 62% of the register would
be exactly the failure the register exists to record" -- but reading is not
guessing, and the file already sorts entries into ЗАКРИТО / ВІДКРИТО / ЗНЯТО /
НЕПЕРЕВІРЮВАНЕ sections that someone filed them into deliberately.

Whether that filing is reliable is itself a measurable question: among the 114
entries that DO carry a state, how often does the state agree with the section
the row sits in? If agreement is high, the section is evidence. If it is low,
the sections have drifted and the signal is worthless -- and it is known that
some have, because #205 reads `закрито` while sitting under ВІДКРИТО.

Measure first, then decide whether to use it.
"""
import re
from pathlib import Path

REGISTER = Path("D:/trading_project/docs/REGISTER.md")
SECTIONS = {
    "## ЗАКРИТО": "закрито",
    "## ВІДКРИТО": "відкрито",
    "## ЗНЯТО": "знято",
    "## НЕПЕРЕВІРЮВАНЕ": "неперевірюване",
}

section = None
rows = []
for line in REGISTER.read_text(encoding="utf-8").splitlines():
    for prefix, name in SECTIONS.items():
        if line.startswith(prefix):
            section = name
    match = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|", line)
    if match and section:
        rows.append((int(match.group(1)), match.group(2).strip(), section, line))

stated = [r for r in rows if r[1] != "?"]
unknown = [r for r in rows if r[1] == "?"]
agree = sum(1 for r in stated if r[1] == r[2])

print(f"rows in sections            {len(rows)}")
print(f"with a state written down   {len(stated)}")
print(f"state agrees with section   {agree}  ({agree/max(len(stated),1):.1%})")
print(f"still unknown               {len(unknown)}\n")

print("where the stated ones disagree with their section:")
for entry_id, state, sect, _ in stated:
    if state != sect:
        print(f"  #{entry_id:<4} says {state:<15} but sits under {sect}")

print("\nthe unknown ones, by the section they sit in:")
by_section: dict[str, int] = {}
for _, _, sect, _ in unknown:
    by_section[sect] = by_section.get(sect, 0) + 1
for sect, count in sorted(by_section.items(), key=lambda kv: -kv[1]):
    print(f"  {sect:<16} {count}")
