from lunar_lander_crosswind.envs import RandomWind, make_lander


def make_windy():
    return RandomWind(make_lander(enable_wind=True))


def test_same_seed_gives_same_wind():
    env = make_windy()
    first = env.reset(seed=123)[1]
    second = env.reset(seed=123)[1]
    env.close()
    assert first == second


def test_wind_stays_within_limits():
    env = make_windy()
    for seed in range(200):
        info = env.reset(seed=seed)[1]
        assert 0.0 <= info["wind_power"] <= 15.0
        assert 0.0 <= info["turbulence_power"] <= 1.5
    env.close()


def test_about_one_in_five_episodes_are_calm():
    env = make_windy()
    calm = sum(env.reset(seed=s)[1]["wind_severity"] == 0.0 for s in range(1000))
    env.close()
    assert 150 <= calm <= 250