import random
import time


def wait(min_seconds: float = 3, max_seconds: float = 8) -> float:
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)
    return delay
