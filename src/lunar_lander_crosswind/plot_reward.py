from pathlib import Path

import matplotlib.pyplot as plt

from lunar_lander_crosswind.envs import make_lander
from lunar_lander_crosswind.rewards import ShapedReward


def main() -> None:
    env = ShapedReward(make_lander())
    observation, _ = env.reset(seed=0)
    raw_scores, bonuses, training_scores = [], [], []
    done = False
    while not done:
        observation, total, terminated, truncated, info = env.step(0)
        raw_scores.append(info["raw_reward"])
        bonuses.append(info["shaping_bonus"])
        training_scores.append(total)
        done = terminated or truncated
    env.close()

    out = Path("media/milestones")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(raw_scores, label="Game score")
    plt.plot(bonuses, label="Bonus")
    plt.plot(training_scores, label="Training score")
    plt.xlabel("Moment in the flight")
    plt.ylabel("Points this moment")
    plt.legend()
    plt.tight_layout()
    path = out / "02-reward-components.png"
    plt.savefig(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
    