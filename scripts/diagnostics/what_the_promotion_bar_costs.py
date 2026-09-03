"""What would the additive power curve look like at family_size = 1?

Computed from the runs already made rather than re-run. Each refusal records
its margin over the strongest sigma-scored opponent and the sigma of that
margin, so the only thing the bar changes is the multiplier the margin must
clear: 2.90 for 27 attempts, 1.645 for one.

The trap this has to avoid: the single-feature rung carries NO sigma and is
independent of the bar, so a run refused by it stays refused however low the
bar goes. Counting a sigma-clearing run as "would promote" without checking
that rung would overstate what a smaller family buys.
"""
import re
from pathlib import Path

LOG = Path("C:/Users/Alexa/AppData/Local/Temp/claude/D--trading-project/"
           "c07ac5f5-6220-4d27-a8fe-466217bafe25/scratchpad/additive_curve.log")
text = LOG.read_text(encoding="utf-8", errors="replace")

# One row per run, in order.
rows = re.findall(r"^\s+(0\.\d{4})\s+(\d\.\d{3})\s+(\d{8})\s+(\d)\s+.*?(True|False)\s*$",
                  text, re.M)
# One block per refusal, in the same order.
refusals = re.findall(r"Champion NOT promoted for [^:]+: (.+)", text)

print(f"{len(rows)} runs, {len(refusals)} refusals\n")
BAR_27, BAR_1 = 2.90, 1.645

by_share_27, by_share_1, totals = {}, {}, {}
refusal_index = 0
for share, best, seed, kept, promoted in rows:
    share = float(share)
    totals[share] = totals.get(share, 0) + 1
    if promoted == "True":
        by_share_27[share] = by_share_27.get(share, 0) + 1
        by_share_1[share] = by_share_1.get(share, 0) + 1
        continue
    reason = refusals[refusal_index] if refusal_index < len(refusals) else ""
    refusal_index += 1
    single_feature = "SINGLE FEATURE" in reason
    m = re.search(r"margin ([+-][\d.]+), sigma ([\d.]+)", reason)
    if not m:
        continue
    z = float(m.group(1)) / float(m.group(2))
    would = (z >= BAR_1) and not single_feature
    if would:
        by_share_1[share] = by_share_1.get(share, 0) + 1
    print(f"  share {share:.4f} seed {seed}: margin/sigma {z:+.2f}, "
          f"single-feature rung {'BINDS' if single_feature else 'clear'} "
          f"-> at 1.645 sigma: {'PROMOTES' if would else 'still refused'}")

print(f"\n{'share':>8}{'Sharpe':>9}{'at 2.90 sigma':>16}{'at 1.645 sigma':>17}")
SHARPE = {0.3395: 8.34, 0.0173: 1.68, 0.0057: 0.96, 0.0025: 0.64}
for share in sorted(totals, reverse=True):
    n = totals[share]
    print(f"{share:>8.4f}{SHARPE.get(share, float('nan')):>9.2f}"
          f"{f'{by_share_27.get(share, 0)} of {n}':>16}"
          f"{f'{by_share_1.get(share, 0)} of {n}':>17}")
