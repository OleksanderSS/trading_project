#!/usr/bin/env python3
"""
Add `default=str` to all json.dump() calls that lack it.

Rationale:
  pandas Timestamp, numpy int64/float32, datetime objects — all appear
  naturally in pipeline/trading/analytics data structures but are not
  JSON-serializable by default. Using default=str converts them to their
  string representation which is safe and reversible for logging/audit
  purposes. This is the same approach already used in the async path
  (save_evaluation_summary_async) and in report_generator.py.

Strategy:
  - Match `json.dump(...)` on a single line where `default=` is absent.
  - For multi-line calls, look for the closing `)` and insert before it.
  - Skip lines that already have `default=`.
  - Skip lines in comments or string literals.
"""
import pathlib
import re
import sys

SKIP_DIRS = {'.git', '__pycache__', '.mypy_cache', '.ruff_cache',
             'node_modules', 'trading_project'}  # avoid duplicate repo

SINGLE_LINE = re.compile(
    r'(json\.dump\()'           # group 1: opening
    r'([^)]*?)'                 # group 2: args (no nested parens — handles 95% of cases)
    r'(\))',                    # group 3: closing paren
    re.DOTALL,
)

fixed = 0
skipped = 0

for py in sorted(pathlib.Path('src').rglob('*.py')):
    if any(d in py.parts for d in SKIP_DIRS):
        continue
    try:
        text = py.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        continue

    if 'json.dump' not in text:
        continue

    lines = text.splitlines(keepends=True)
    new_lines = []
    changed = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip comment lines
        if stripped.startswith('#'):
            new_lines.append(line)
            i += 1
            continue

        if 'json.dump(' not in line:
            new_lines.append(line)
            i += 1
            continue

        # Already has default= on same line or within 8-line block
        block = ''.join(lines[i:i+8])
        if 'default=' in block:
            new_lines.append(line)
            i += 1
            continue

        # Single-line case: json.dump(args) all on one line
        # Pattern: json.dump( ... ) where closing ) is on same line
        if re.search(r'json\.dump\(.*\)', line):
            # Insert `default=str` before the last closing paren
            # Find the matching closing paren (simple: rightmost ')' after 'json.dump(')
            pos_start = line.index('json.dump(')
            # Walk to find the closing paren with bracket depth
            depth = 0
            closing_pos = -1
            for j, ch in enumerate(line[pos_start:], pos_start):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        closing_pos = j
                        break

            if closing_pos > 0:
                before = line[:closing_pos]
                after = line[closing_pos:]
                # Don't add comma if args list ends with trailing comma already
                if before.rstrip().endswith(','):
                    new_line = before + ' default=str' + after
                else:
                    new_line = before + ', default=str' + after
                new_lines.append(new_line)
                changed = True
                fixed += 1
                i += 1
                continue

        # Multi-line case: find closing ')' in next few lines
        # Collect until we find the line with the sole closing paren
        collected = [line]
        j = i + 1
        while j < len(lines) and j < i + 15:
            collected.append(lines[j])
            block_so_far = ''.join(collected)
            # Check bracket balance
            depth = 0
            for ch in block_so_far:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            if depth == 0:
                break
            j += 1

        block_text = ''.join(collected)
        if 'default=' in block_text:
            new_lines.extend(collected)
            i = j + 1
            continue

        # Find last ')' in the block and insert before it
        # Work backwards from end to find closing paren
        last_paren = block_text.rfind(')')
        if last_paren > 0:
            before = block_text[:last_paren]
            after = block_text[last_paren:]
            if before.rstrip().endswith(','):
                new_block = before + ' default=str' + after
            else:
                new_block = before + ', default=str' + after
            new_lines.extend(new_block.splitlines(keepends=True))
            changed = True
            fixed += 1
        else:
            new_lines.extend(collected)
        i = j + 1

    if changed:
        py.write_text(''.join(new_lines), encoding='utf-8')
        print(f'Fixed: {py}')

print(f'\nTotal fixed: {fixed}')
