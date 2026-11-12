"""Plot for sep_steering.py results: attack success vs trusted-instruction success per condition (targeted vs random)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path(sys.argv[1])
d = json.loads((p / "results.json").read_text())
R = {r["name"]: r for r in d["results"]}
plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e1e0d9", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left"})
conds = [("data_ablate_mid", "remove, data tokens, L8-22"), ("data_reflect_mid", "flip, data tokens, L8-22"),
         ("data_ablate_mid_all_layers", "remove, data tokens, all layers"),
         ("all_ablate_mid", "remove, ALL tokens, L8-22"), ("all_ablate_last", "remove 'noticed' dir, ALL tokens")]
fig, ax = plt.subplots(figsize=(7.2, 5.6))
b = R["baseline"]
ax.scatter(b["attack_success"], b["utility_exec"], s=90, color="#0b0b0b", zorder=3)
ax.annotate("no intervention", (b["attack_success"], b["utility_exec"]), xytext=(-8, 8), textcoords="offset points",
            ha="right", color="#0b0b0b", fontsize=9)
for name, lab in conds:
    t, r = R[name], R[name + " [random dir]"]
    ax.annotate("", (t["attack_success"], t["utility_exec"]), (r["attack_success"], r["utility_exec"]),
                arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=1))
    ax.scatter(r["attack_success"], r["utility_exec"], s=55, facecolor="#fcfcfb", edgecolor="#eb6834", lw=1.8, zorder=3)
    ax.scatter(t["attack_success"], t["utility_exec"], s=60, color="#2a78d6", edgecolor="#fcfcfb", lw=1.5, zorder=4)
    ax.annotate(lab, (t["attack_success"], t["utility_exec"]), xytext=(7, -3), textcoords="offset points",
                color="#52514e", fontsize=8.5)
ax.scatter([], [], s=60, color="#2a78d6", label="injection direction")
ax.scatter([], [], s=55, facecolor="#fcfcfb", edgecolor="#eb6834", lw=1.8, label="random direction (same layers/tokens)")
ax.set_xlim(-0.03, 1.0)
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel("attack success (injected instruction executed)  ← better")
ax.set_ylabel("trusted instruction executed  ↑ better")
ax.set_title(f"Mellum: steering away the 'instruction in data' direction\n"
             f"(held-out probes; attack n={d['n_attack']}, trusted n={d['n_utility']})", color="#0b0b0b")
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(p / "steering.png", dpi=150)
