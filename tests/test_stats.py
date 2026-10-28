"""Check the statistics against values computed independently."""

import pytest

from mandatebench.stats import holm, mcnemar_exact, wilson


def test_wilson_matches_reference_values():
    # Reference: Wilson interval for 10/50 at 95% is (0.1124, 0.3304).
    lo, hi = wilson(10, 50)
    assert lo == pytest.approx(0.1124, abs=1e-4) and hi == pytest.approx(0.3304, abs=1e-4)


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    assert wilson(0, 24)[0] == 0.0 and wilson(0, 24)[1] < 0.15
    assert wilson(24, 24)[1] == 1.0 and wilson(24, 24)[0] > 0.85


def test_mcnemar_exact_reference_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)          # 2 * 0.5**6
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)   # 2 * (1 + 10) / 2**10
    assert mcnemar_exact(5, 5) == 1.0


def test_holm_step_down():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)        # 3 * 0.01
    assert adj["c"] == pytest.approx(0.06)        # 2 * 0.03
    assert adj["b"] == pytest.approx(0.06)        # max(0.06, 1 * 0.04): monotone
