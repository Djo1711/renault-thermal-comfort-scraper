from unittest.mock import patch

from src.utils.rate_limiter import wait


def test_wait_sleeps_for_random_delay_within_bounds():
    with patch("src.utils.rate_limiter.random.uniform", return_value=5.0) as mock_uniform, \
         patch("src.utils.rate_limiter.time.sleep") as mock_sleep:
        delay = wait(min_seconds=3, max_seconds=8)

    mock_uniform.assert_called_once_with(3, 8)
    mock_sleep.assert_called_once_with(5.0)
    assert delay == 5.0
