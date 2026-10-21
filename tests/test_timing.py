import pytest
from argus.timing import TickTiming


def test_stage_timing_includes_descheduling_and_tick_interval():
    times = iter([1.0, 1.003, 1.303, 1.310])
    tick = TickTiming(previous_start=0.5, clock=lambda: next(times))
    tick.mark('capture_wait')
    tick.mark('depth')
    r = tick.finish(depth_ok=True)
    assert r['interval_ms'] == 500
    assert r['capture_wait_ms'] == pytest.approx(3)
    assert r['depth_ms'] == pytest.approx(300)
    assert r['tick_ms'] == pytest.approx(310)
    assert r['depth_ok'] is True


def test_missing_pair_is_not_a_successful_depth_tick():
    tick = TickTiming(clock=lambda: 5)
    result = tick.finish()
    assert result['interval_ms'] is None
    assert result['depth_ok'] is False
    assert result['tick_ms'] == 0


def test_terminal_camera_outage_is_counted_without_recovery():
    from argus.timing import heartbeat_gaps
    gaps = heartbeat_gaps([1.1, 1.2], 1, 5, previous=.99)
    assert gaps == [dict(start=1.2, end=5, gap_s=3.8, open=True)]


def test_outage_spanning_phase_boundary_keeps_real_start():
    from argus.timing import heartbeat_gaps
    gaps = heartbeat_gaps([2.5, 2.6], 2, 2.7, previous=1)
    assert len(gaps) == 1
    assert gaps[0]['gap_s'] == 1.5
    assert gaps[0]['open'] is False


def test_no_heartbeat_is_an_outage_and_250ms_is_not_over_limit():
    from argus.timing import heartbeat_gaps
    assert heartbeat_gaps([], 1, 2)[0]['open'] is True
    assert heartbeat_gaps([1.25], 1, 1.5) == []
