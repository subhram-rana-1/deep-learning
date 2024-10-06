import random


def randomness(lo: float, hi: float) -> float:
    """get random value  which is in b/w lo and high """
    if lo > hi:
        raise Exception(f"can't generate random float in range [{lo}, {hi}] ")

    return lo + random.gauss(0, hi-lo)
