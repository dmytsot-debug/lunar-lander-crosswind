import gymnasium as gym


def make_lander(
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
    render_mode: str | None = None,
) -> gym.Env:
    return gym.make(
        "LunarLander-v3",
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        render_mode=render_mode,
    )


class RandomWind(gym.Wrapper):
    """Picks new wind and turbulence strengths at the start of every episode."""

    def __init__(
        self,
        env: gym.Env,
        calm_chance: float = 0.2,
        max_wind_power: float = 15.0,
        max_turbulence_power: float = 1.5,
    ):
        super().__init__(env)
        self.calm_chance = calm_chance
        self.max_wind_power = max_wind_power
        self.max_turbulence_power = max_turbulence_power

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        # Wind strength is only read while flying, so it is safe to set it right after reset.
        dice = self.env.unwrapped.np_random
        if dice.random() < self.calm_chance:
            severity = 0.0
        else:
            severity = float(dice.uniform(0.0, 1.0))

        lander = self.env.unwrapped
        lander.wind_power = severity * self.max_wind_power
        lander.turbulence_power = severity * self.max_turbulence_power

        info["wind_severity"] = round(severity, 3)
        info["wind_power"] = round(lander.wind_power, 2)
        info["turbulence_power"] = round(lander.turbulence_power, 2)
        return observation, info