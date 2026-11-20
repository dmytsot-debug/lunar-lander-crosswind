from lunar_lander_crosswind.rewards import shaping_bonus
from lunar_lander_crosswind.envs import make_lander
from lunar_lander_crosswind.rewards import ShapedReward

TIDY_BEFORE = [0.4, 0, 0.3, 0, 0.2, 0.1, 0, 0]
TIDY_AFTER = [0.2, 0, 0.1, 0, 0.1, 0.0, 0, 0]
MESSY_AFTER = [0.8, 0, 0.6, 0, 0.5, 0.4, 0, 0]


def test_tidier_flight_earns_a_bonus():
    assert round(shaping_bonus(TIDY_BEFORE, TIDY_AFTER, False), 3) == 0.6

def test_huge_worsening_is_capped_at_minus_one():
    assert shaping_bonus(TIDY_BEFORE, MESSY_AFTER, False) == -1.0


def test_finished_flight_gets_no_bonus():
    assert shaping_bonus(TIDY_BEFORE, TIDY_AFTER, True) == 0.0


def test_training_score_is_game_score_plus_bonus():
    env = ShapedReward(make_lander())
    env.reset(seed=0)
    _, total, _, _, info = env.step(0)
    env.close()
    assert round(info["raw_reward"] + info["shaping_bonus"], 5) == round(total, 5)


def test_turning_shaping_off_uses_only_the_game_score():
    env = ShapedReward(make_lander(), enabled=False)
    env.reset(seed=0)
    _, total, _, _, info = env.step(0)
    env.close()
    assert info["shaping_bonus"] == 0.0
    assert total == info["raw_reward"]