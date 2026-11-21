from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from lunar_lander_crosswind.envs import make_lander

VIDEO = Path("media/milestones")


def main() -> None:
    VIDEO.mkdir(parents=True, exist_ok=True)
    env = make_lander(enable_wind=False, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(VIDEO),
        episode_trigger=lambda n: n == 0,
        name_prefix="03-pilot-calm",
    )
    pilot = PPO.load("runs/pilot-calm/pilot")
    observation, _ = env.reset(seed=0)
    total = 0.0
    done = False
    while not done:
        action, _ = pilot.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        total += reward
        done = terminated or truncated
    env.close()
    print(f"Game score: {total:.1f}")


if __name__ == "__main__":
    main()
    