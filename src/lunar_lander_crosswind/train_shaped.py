from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander_crosswind.envs import RandomWind, make_lander
from lunar_lander_crosswind.rewards import ShapedReward

OUT = Path("runs/pilot-shaped")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = ShapedReward(RandomWind(make_lander(enable_wind=True)))
    pilot = PPO("MlpPolicy", env, seed=0, verbose=1)
    pilot.learn(total_timesteps=20_000)
    pilot.save(OUT / "pilot")
    env.close()
    print(f"Saved brain in {OUT.resolve()}")


if __name__ == "__main__":
    main()  