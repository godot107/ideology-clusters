"""The issues post's figure: how far apart the parties voted, by policy area (needs scripts/issues.py).

    python scripts/plot_issues.py   # -> reports/04_issues.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0"
ACCENT = "#2a78d6"

pa = pd.read_csv(ROOT / "results" / "issues_policy_areas_119H.csv")
pa = pa[pa.votes >= 5].sort_values("median_gap")
fig, ax = plt.subplots(figsize=(10, 7.6), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for i, (_, r) in enumerate(pa.iterrows()):
    c = ACCENT if r.policy_area == "Energy" else MUTED
    w = max(r.median_gap, 0.006)
    ax.add_patch(FancyBboxPatch((0, i - 0.32), w, 0.64, boxstyle="round,pad=0,rounding_size=0.02",
                                mutation_aspect=0.04, fc=c, ec="none"))
    ax.text(w + 0.012, i, f"{r.median_gap:.2f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(pa)))
ax.set_yticklabels([f"{a}  ({int(v)})" for a, v in zip(pa.policy_area, pa.votes)], fontsize=9.5, color=INK)
ax.set_xlim(0, 1.12)
ax.set_ylim(-0.7, len(pa) - 0.3)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0\nvoted alike", "0.25", "0.5", "0.75", "1\nstrict party line"], fontsize=9, color=INK2)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.text(0.40, 2.2, "Energy (blue): 83% of its votes were party-line.\nThe exception: a bill pushing states to make big data\ncenters pay for their own grid upgrades passed 417–3.",
        fontsize=10, color=INK, va="center", linespacing=1.4)
fig.text(0.02, 0.975, "How far apart the two parties voted, by issue", fontsize=14, weight="bold", color=INK, va="top")
fig.text(0.02, 0.94, "U.S. House, 2025–26. Typical gap between the share of Democrats and the share of Republicans voting yes\n"
         "(median over the votes in each area; number of votes in brackets).", fontsize=9.5, color=INK2, va="top")
fig.text(0.02, 0.012, "Data: Voteview roll calls; policy areas from Congress.gov (Library of Congress). "
         "Final-passage, fast-track and amendment votes on bills; areas with 5+ votes. willieman.com",
         fontsize=8, color=INK2)
fig.subplots_adjust(left=0.33, right=0.97, top=0.875, bottom=0.1)
fig.savefig(ROOT / "reports" / "04_issues.png", facecolor=SURFACE)
print("saved")
