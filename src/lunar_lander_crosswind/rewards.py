def messiness(observation) -> float:
    sideways, _, sideways_speed, _, tilt, spin, _, _ = observation
    return abs(sideways) + abs(sideways_speed) + abs(tilt) + abs(spin)


def shaping_bonus(before, after, finished: bool) -> float:
    if finished:
        return 0.0
    change = messiness(before) - messiness(after)
    return max(-1.0, min(1.0, change))