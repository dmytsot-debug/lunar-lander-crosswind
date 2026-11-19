from lunar_lander_crosswind.rewards import shaping_bonus

TIDY_BEFORE = [0.4, 0, 0.3, 0, 0.2, 0.1, 0, 0]
TIDY_AFTER = [0.2, 0, 0.1, 0, 0.1, 0.0, 0, 0]
MESSY_AFTER = [0.8, 0, 0.6, 0, 0.5, 0.4, 0, 0]


def test_tidier_flight_earns_a_bonus():
    assert round(shaping_bonus(TIDY_BEFORE, TIDY_AFTER, False), 3) == 0.6

def test_huge_worsening_is_capped_at_minus_one():
    assert shaping_bonus(TIDY_BEFORE, MESSY_AFTER, False) == -1.0


def test_finished_flight_gets_no_bonus():
    assert shaping_bonus(TIDY_BEFORE, TIDY_AFTER, True) == 0.0