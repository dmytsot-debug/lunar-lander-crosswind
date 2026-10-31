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