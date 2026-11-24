from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander_crosswind.envs import RandomWind, make_lander
from lunar_lander_crosswind.rewards import ShapedReward

STEPS = 1_000_000
SEEDS = (0, 1, 2)


def make_env(condition: str):
    if condition == "calm":
        return make_lander(enable_wind=False)
    lander = RandomWind(make_lander(enable_wind=True))
    if condition == "shaped":
        return ShapedReward(lander)
    return lander


def main() -> None:
    for condition in ("calm", "wind", "shaped"):
        for seed in SEEDS:
            out = Path("runs") / "experiment" / condition / f"seed-{seed}"
            out.mkdir(parents=True, exist_ok=True)
            env = make_env(condition)
            pilot = PPO("MlpPolicy", env, seed=seed, verbose=1)
            pilot.learn(total_timesteps=STEPS)
            pilot.save(out / "pilot")
            env.close()
            print(f"Saved {condition} seed {seed} in {out.resolve()}")


if __name__ == "__main__":
    main()
    