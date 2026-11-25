from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from lunar_lander_crosswind.envs import make_lander

OUT = Path("media/milestones")
FLIGHTS = (
    (Path("runs/experiment/calm/seed-1/pilot"), "05-calm-seed1-severe"),
    (Path("runs/experiment/shaped/seed-2/pilot"), "05-shaped-seed2-severe"),
)


def record_flight(brain: Path, name_prefix: str) -> float:
    env = make_lander(
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="rgb_array",
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(OUT),
        episode_trigger=lambda episode_number: episode_number == 0,
        name_prefix=name_prefix,
    )
    pilot = PPO.load(brain)
    observation, _ = env.reset(seed=7)
    score = 0.0
    finished = False
    while not finished:
        action, _ = pilot.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        score += float(reward)
        finished = terminated or truncated
    env.close()
    return score


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for brain, name_prefix in FLIGHTS:
        score = record_flight(brain, name_prefix)
        print(f"{name_prefix} game score: {score}")


if __name__ == "__main__":
    main()
