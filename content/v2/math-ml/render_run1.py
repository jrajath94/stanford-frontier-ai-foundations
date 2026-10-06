"""Render provenance record for math-ml RUN 1, figure u01 f09.

RUN 1 (2026-10-06) shipped visuals/u01/f09_honest_scatter.png without a
preserved render script. The exact snippet from the lesson's C09 section
(lessons/u01/lesson-01-mathematical-language.md) is reproduced below,
with the course palette (#F7F4EE) applied per visual_system_generic.md.

Run: python3 render_run1.py  ->  writes /tmp/f09_honest_scatter_rerun.png
The original visuals/u01/f09_honest_scatter.png is RETAINED as the
audited artifact (opened and read 2026-10-06, six dots, axes at 0,
labels carry units). This script is a provenance record, not a
byte-exact reproduction: RUN 1's exact style calls were not preserved.
See visual_audit.md logged decision F-08.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#F7F4EE"
INK = "#1B2838"
MUTED = "#5C6B7A"
LINE = "#D9D3C7"
PANEL = "#FFFDF8"

x = [800, 950, 1100, 1300, 1500, 1800]
y = [320, 385, 445, 520, 610, 740]

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(PANEL)
for s in ax.spines.values():
    s.set_color(LINE)
ax.tick_params(colors=MUTED, labelsize=10)
ax.scatter(x, y, s=64, color="#1F7A72", edgecolors=INK, linewidths=0.8)
ax.set_xlabel("size (sq ft)", color=INK)
ax.set_ylabel("price (thousand dollars)", color=INK)
ax.set_title("House size vs price (toy data, 6 points)", color=INK)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 800)
fig.savefig("/tmp/f09_honest_scatter_rerun.png", dpi=150)
print("wrote /tmp/f09_honest_scatter_rerun.png")
