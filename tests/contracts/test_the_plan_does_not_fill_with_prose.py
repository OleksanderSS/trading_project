"""ROADMAP is a plan. Prose in it may not grow.

The owner's complaint that started this whole audit, in his words: "таблиці
збільшуються і збільшуються". The register answered it by archiving settled
rows and keeping a one-line index. The plan had no such answer, and on
2026-09-06 the person adding prose to it was me: three new sections, about
13 KB, none carrying a single task.

Measured that day, after moving one of them out:

    ROADMAP                       96 KB
    open task items               46 (11.7 KB on their own lines)
    completed task items          41 (6.9 KB)
    sections with NO task at all  9  (17.4 KB)

So archiving completed ITEMS is not the lever here -- they are 7% of the file.
The lever is prose: nine task-free sections hold more bytes than every open
task in the plan.

WHY A CEILING AND NOT A BAN. Several task-free sections earn their place: §0
is the thesis the rest follows from, §3 is the index, §9 is the dependency
graph, §29 holds the standing preconditions that must be read BEFORE starting
any item. A rule requiring every section to carry tasks would fire on all four,
and a rule that fires on correct structure gets switched off.

WHAT THIS PINS. The total may fall and may not rise. New reasoning goes where
the split says it goes: a measurement to CLAIMS, an observation about method to
WORKING_METHOD, a doubt to CRITIQUE. What stays in the plan is what changes
what to do next.
"""
from __future__ import annotations

import re
from pathlib import Path

ROADMAP = Path(__file__).resolve().parents[2] / "docs" / "ROADMAP.md"

SECTION = re.compile(r"^## ")
TASK = re.compile(r"^\s*[-*]\s*\[")

#: Bytes held by sections that carry no task at all. Measured 2026-09-06 at
#: 17,392 after §30 was moved to WORKING_METHOD (my first note here said
#: 16,632 and this very test caught the arithmetic). Lower it when prose moves
#: out; never raise it. Raising it is precisely the growth the owner named.
PROSE_CEILING = 76_100


def _task_free_sections() -> list[tuple[int, str]]:
    """Prose per section: every line that is not a task line, everywhere.

    THE FIRST VERSION SKIPPED A WHOLE SECTION IF IT HELD ONE TASK, which is a
    rule a writer defeats by pressing a key. Measured 2026-09-07 after I had
    done exactly that twice in one day without noticing: the guard was counting
    13,323 bytes and NOT counting 62,667 more, sitting in sections that
    happened to contain a checkbox. It watched 18% of what it claimed to watch,
    and its ceiling had been ratcheted down twice on that 18%, which made it
    look like it was working.

    Now a line is prose unless it is a task line. A task wrapping across lines
    has its continuation counted as prose, which overstates slightly and in the
    safe direction: the ceiling can then only be paid down by removing text.
    """
    lines = ROADMAP.read_text(encoding="utf-8").split("\n")
    starts = [i for i, line in enumerate(lines) if SECTION.match(line)]
    if not starts:
        return [(sum(len(line) for line in lines if not TASK.match(line)),
                 "(no sections)")]
    out: list[tuple[int, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        block = lines[start:end]
        prose = sum(len(line) for line in block if not TASK.match(line))
        if prose:
            out.append((prose, lines[start][:70]))
    return out


def test_prose_in_the_plan_does_not_grow():
    sections = _task_free_sections()
    total = sum(size for size, _ in sections)
    assert total <= PROSE_CEILING, (
        f"task-free prose in ROADMAP rose to {total:,} bytes across "
        f"{len(sections)} sections (ceiling {PROSE_CEILING:,}).\n"
        + "\n".join(f"  {size:>7,}  {title}"
                    for size, title in sorted(sections, reverse=True))
        + "\n\nA measurement belongs in CLAIMS, an observation about method in "
        "WORKING_METHOD, a doubt in CRITIQUE. The plan keeps what changes what "
        "to do next."
    )


#: Prose bytes per byte of task text. A RATCHET, not a goal: the plan is at
#: 4.07 today and that number may fall and never rise.
#:
#: It replaced an assertion that task text simply EXCEEDS prose -- which read
#: as a fact and was false. It passed only because the counting skipped any
#: section holding a task, and on that 18% view the claim happened to hold. So
#: the guard was asserting something untrue about the document while looking
#: like it was enforcing it, which is worse than not checking: it answered the
#: question nobody re-asked.
PROSE_TO_TASK_CEILING = 4.10


def test_the_plan_does_not_become_more_essay_than_it_already_is():
    """The shape that makes it a plan rather than a journal.

    Not a size limit -- a proportion, and a truthful one. The plan holds four
    times more reasoning than work. That is the measured state, not the target;
    the target is for this number to fall, and the ratchet is what makes falling
    the only direction available.
    """
    lines = ROADMAP.read_text(encoding="utf-8").split("\n")
    task_text = sum(len(line) for line in lines if TASK.match(line))
    prose = sum(size for size, _ in _task_free_sections())
    ratio = prose / max(task_text, 1)
    assert ratio <= PROSE_TO_TASK_CEILING, (
        f"prose is {prose:,} bytes against {task_text:,} of task text -- a "
        f"ratio of {ratio:.2f} against a ceiling of {PROSE_TO_TASK_CEILING}.\n"
        "Reasoning belongs in CLAIMS (a measurement), WORKING_METHOD (a method) "
        "or CRITIQUE (a doubt).\nWhat stays in the plan is what changes what to "
        "do next."
    )


def test_the_ceiling_is_not_slack():
    """A ratchet with room to spare stops the NEXT addition and not this one.

    17,392 -> 16,878 on the same day: the map in §3 was rewritten truthfully and
    PAID for by moving §19a, a dated journal entry, into AUDIT_HISTORY. The
    ceiling followed down to 16,900 -- twenty-two bytes of headroom,
    which is a line of text, not a budget. A rule introduced with a budget
    never binds (#203).
    """
    total = sum(size for size, _ in _task_free_sections())
    assert PROSE_CEILING - total < 500, (
        f"the ceiling sits {PROSE_CEILING - total:,} bytes above the actual "
        f"{total:,}; lower it to what the file holds now"
    )
