from argus.stereo_pairing import select_pair


def f(t, seq):
    return (object(), t, seq)


def test_uses_previous_arrival_when_latest_pair_exceeds_gate():
    left = [f(1.000, 1), f(1.017, 2), f(1.034, 3)]
    right = [f(1.002, 1), f(1.019, 2)]
    l, r = select_pair(left, right, (0, 0), 1.035, 12)
    assert (l[2], r[2]) == (2, 2)
    assert abs(l[1]-r[1])*1000 < 12


def test_stalled_camera_never_reuses_frame_as_new_depth():
    assert select_pair([f(1.02, 2)], [f(1.01, 1)], (1, 1), 1.03, 12) is None


def test_stale_unconsumed_frames_are_unknown():
    assert select_pair([f(1, 1)], [f(1, 1)], (0, 0), 1.2, 12) is None


def test_no_compatible_pair_does_not_weaken_gate():
    assert select_pair([f(1, 1)], [f(1.02, 1)], (0, 0), 1.03, 12) is None


def test_recovery_advances_both_sequences():
    l, r = select_pair([f(2, 30)], [f(2.003, 20)], (10, 10), 2.01, 12)
    assert (l[2], r[2]) == (30, 20)


def test_future_arrival_and_empty_history_rejected():
    assert select_pair([f(2, 1)], [f(2, 1)], (0, 0), 1, 12) is None
    assert select_pair([], [], (0, 0), 1, 12) is None
