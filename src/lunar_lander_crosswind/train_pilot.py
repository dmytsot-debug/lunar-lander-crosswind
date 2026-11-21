from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander_crosswind.envs import make_lander

OUT = Path("runs/pilot-calm")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_lander(enable_wind=False)
    # PPO is the training method. It tries actions, keeps the ones that scored well.
    pilot = PPO("MlpPolicy", env, seed=0, verbose=1)
    # 20_000 moments is a few minutes, enough to prove training works, not enough to land well.
    pilot.learn(total_timesteps=20_000)
    pilot.save(OUT / "pilot")
    env.close()
    print(f"Saved brain in {OUT.resolve()}")


if __name__ == "__main__":
    main()