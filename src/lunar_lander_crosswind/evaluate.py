from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander_crosswind.envs import make_lander

BRAINS = Path("runs/experiment")
CONDITIONS = ("calm", "wind", "shaped")
SEEDS = (0, 1, 2)
FLIGHTS = 100
FIRST_SEED = 1_000
WEATHER = {
    "calm": (False, 0.0, 0.0),
    "mild": (True, 5.0, 0.5),
    "moderate": (True, 10.0, 1.0),
    "severe": (True, 15.0, 1.5),
    "unseen": (True, 20.0, 2.0),
}


def one_flight(pilot, env) -> float:
    observation, _ = env.reset()
    score = 0.0
    finished = False
    while not finished:
        action, _ = pilot.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        score += float(reward)
        finished = terminated or truncated
    return score


def main() -> None:
    print("condition,seed,weather,mean_score")
    for condition in CONDITIONS:
        for seed in SEEDS:
            pilot = PPO.load(BRAINS / condition / f"seed-{seed}" / "pilot")
            for name, (wind, power, turbulence) in WEATHER.items():
                scores = []
                for flight in range(FLIGHTS):
                    env = make_lander(
                        enable_wind=wind,
                        wind_power=power,
                        turbulence_power=turbulence,
                    )
                    env.reset(seed=FIRST_SEED + flight)
                    scores.append(one_flight(pilot, env))
                    env.close()
                mean = sum(scores) / len(scores)
                print(f"{condition},{seed},{name},{mean:.1f}")


if __name__ == "__main__":
    main()