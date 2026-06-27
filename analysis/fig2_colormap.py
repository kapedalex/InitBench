"""
Figure 2: Comparative Analysis — Average Agent Initiative by Model
Вертикальные бары, цвет по значению (YlGn), эмодзи 😇 (original) / 😈 (abliterated).

Данные считаются из ../model_family_experiments/results_table.tsv + старая
gpt-oss/heretic пара (см. data_loader.py).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from data_loader import load_dataset

# ── данные ─────────────────────────────────────────────────────────────────────
models, scores, _fracs, is_abl = load_dataset()

# ── цветовая шкала ─────────────────────────────────────────────────────────────
cmap   = plt.get_cmap("YlGn")
vmin, vmax = 2.5, 5.0
norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
colors = [cmap(norm(s)) for s in scores]

# ── фигура ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(models))
bars = ax.bar(x, scores, color=colors, width=0.62, zorder=3, edgecolor="none")

# ── разделители между парами семейств ──────────────────────────────────────────
for i in range(2, len(models), 2):
    ax.axvline(i - 0.5, color="#e0e0e0", linestyle="-", linewidth=1, zorder=0)

# ── аннотации значений ────────────────────────────────────────────────────────
for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, color="#333", fontweight="bold")

# ── эмодзи (😈 abliterated / 😇 original) ───────────────────────────────────────
for bar, abl in zip(bars, is_abl):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.30,
            "😈" if abl else "😇",
            ha="center", va="bottom", fontsize=15)

# ── colorbar ──────────────────────────────────────────────────────────────────
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.7)
cbar.set_label("score", fontsize=11)
cbar.ax.tick_params(labelsize=10)

# ── оси и заголовок ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5, rotation=30, ha="right")
ax.set_ylabel("Average Initiative Score (1–5)", fontsize=12)
ax.set_ylim(0, 6.0)
ax.set_title("Comparative Analysis: Average Agent Initiative — Original (😇) vs Abliterated (😈)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Model", fontsize=12)

ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig("fig2_colormap.png", dpi=180, bbox_inches="tight")
print("Saved: fig2_colormap.png")
