from pathlib import Path
import gymnasium as gym

OBSERVATION_LABELS = [
    "Horizontal position",
    "Vertical position",
    "Horizontal speed",
    "Vertical speed",
    "Tilt angle",
    "Spinning speed",
    "Left leg touching ground",
    "Right leg touching ground",
]


def main() -> None:
    video_folder = Path("media/milestones")
    video_folder.mkdir(parents=True, exist_ok=True)

    environment = gym.make("LunarLander-v3", render_mode="rgb_array")
    environment = gym.wrappers.RecordVideo(
        environment,
        video_folder=str(video_folder),
        episode_trigger=lambda episode_number: episode_number == 0,
        name_prefix="00-random-crash",
    )

    observation, _ = environment.reset(seed=42)
    environment.action_space.seed(42)

    print("Initial Lunar Lander observation:")
    for label, value in zip(OBSERVATION_LABELS, observation):
        print(f"- {label}: {value:.3f}")

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, _ = environment.step(action)
        total_reward += reward

    environment.close()
    print(f"\nEpisode finished with total reward: {total_reward:.2f}")
    print(f"Video saved in: {video_folder.resolve()}")