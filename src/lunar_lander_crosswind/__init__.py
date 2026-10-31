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
    environment = gym.make("LunarLander-v3")
    observation, _ = environment.reset(seed=42)

    print("Initial Lunar Lander observation:")
    for label, value in zip(OBSERVATION_LABELS, observation):
        print(f"- {label}: {value:.3f}")

    environment.close()