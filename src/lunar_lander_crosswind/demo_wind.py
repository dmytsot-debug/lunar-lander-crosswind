from pathlib import Path

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import heuristic

from lunar_lander_crosswind.envs import make_lander

VIDEO_FOLDER = Path("media/milestones")
SEED = 7


def fly_once(name: str, enable_wind: bool) -> float:
    env = make_lander(enable_wind=enable_wind, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(VIDEO_FOLDER),
        episode_trigger=lambda episode_number: episode_number == 0,
        name_prefix=name,
    )
    observation, _ = env.reset(seed=SEED)
    total_reward = 0.0
    done = False
    while not done:
        action = heuristic(env.unwrapped, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    return total_reward


def main() -> None:
    VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)
    calm_score = fly_once("01-calm", enable_wind=False)
    windy_score = fly_once("01-crosswind", enable_wind=True)
    print(f"Calm air score: {calm_score:.1f}")
    print(f"Strong crosswind score: {windy_score:.1f}")


if __name__ == "__main__":
    main()