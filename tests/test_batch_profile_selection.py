import pytest

from tools.profile_batch_size import select_batch_size


def _result(batch_size, reserved, status="success"):
    return {
        "batch_size": batch_size,
        "rank": 64,
        "peak_allocated_mib": reserved - 256,
        "peak_reserved_mib": reserved,
        "status": status,
    }


def test_selects_largest_candidate_in_preferred_memory_band():
    results = [_result(8, 12000), _result(12, 16000), _result(16, 19800)]
    assert select_batch_size(results, 15360, 20480) == 16


def test_falls_back_to_largest_safe_candidate_below_target_band():
    results = [_result(4, 9000), _result(8, 11000), _result(12, 14000)]
    assert select_batch_size(results, 15360, 20480) == 12


def test_never_selects_candidate_above_hard_memory_limit():
    results = [_result(8, 16000), _result(12, 20500)]
    assert select_batch_size(results, 15360, 20480) == 8


def test_failed_attempts_are_ignored():
    results = [_result(8, 16000), _result(12, 19000, status="oom")]
    assert select_batch_size(results, 15360, 20480) == 8


def test_no_safe_success_raises():
    with pytest.raises(RuntimeError):
        select_batch_size([_result(20, 21000)], 15360, 20480)
