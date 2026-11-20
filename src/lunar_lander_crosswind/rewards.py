import gymnasium as gym

def messiness(observation) -> float:
    sideways, _, sideways_speed, _, tilt, spin, _, _ = observation
    return abs(sideways) + abs(sideways_speed) + abs(tilt) + abs(spin)


def shaping_bonus(before, after, finished: bool) -> float:
    if finished:
        return 0.0
    change = messiness(before) - messiness(after)
    return max(-1.0, min(1.0, change))


class ShapedReward(gym.Wrapper):
    """Adds the progress bonus on top of the game's own score."""

    def __init__(self, env: gym.Env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        self.previous = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous = observation
        return observation, info

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        finished = terminated or truncated
        bonus = 0.0
        if self.enabled:
            bonus = shaping_bonus(self.previous, observation, finished)
        info["raw_reward"] = raw_reward
        info["shaping_bonus"] = bonus
        self.previous = observation
        return observation, raw_reward + bonus, terminated, truncated, info    