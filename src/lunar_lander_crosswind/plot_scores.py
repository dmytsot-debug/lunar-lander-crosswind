from pathlib import Path

import matplotlib.pyplot as plt

rows = Path("results/scores.csv").read_text().strip().splitlines()[1:]
weather = ["calm", "mild", "moderate", "severe", "unseen"]
conditions = ["calm", "wind", "shaped"]
totals = {c: {w: [] for w in weather} for c in conditions}
for line in rows:
    condition, _seed, name, score = line.split(",")
    totals[condition][name].append(float(score))

plt.figure(figsize=(7, 4))
for condition in conditions:
    means = [sum(totals[condition][w]) / len(totals[condition][w]) for w in weather]
    plt.plot(weather, means, marker="o", label=condition)
plt.axhline(200, color="gray", linestyle="--", linewidth=1)
plt.ylabel("Average game score")
plt.xlabel("Weather")
plt.legend()
plt.tight_layout()
out = Path("media/milestones/05-weather-scores.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out)
print(f"Saved {out.resolve()}")