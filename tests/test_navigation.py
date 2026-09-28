"""Corridor guidance over synthetic calibrated depth."""
import numpy as np

from argus.config import NavigationConfig
from argus.navigation import Navigator

H, W = 240, 320
FX, CX = 300.0, W / 2


def nav():
    return Navigator(NavigationConfig(bins=16, clear_distance_m=2.0, min_corridor_deg=8.0,
                                      straight_deg=6.0, min_pixels_per_bin=10), FX, CX)


def depth(fill=6.0):
    return np.full((H, W), fill, dtype=np.float32)


def test_open_space_is_clear_ahead():
    g = nav().evaluate(depth(6.0))
    assert g.status == "clear"
    assert abs(g.heading_deg) < 3


def test_wall_across_the_path_is_blocked():
    g = nav().evaluate(depth(1.0))
    assert g.status == "blocked"
    assert g.heading_deg is None
    assert "Stop" in g.message


def test_obstacle_on_the_right_veers_left():
    d = depth(6.0)
    d[:, W // 2:] = 1.0
    g = nav().evaluate(d)
    assert g.status == "veer_left"
    assert g.heading_deg < 0


def test_obstacle_on_the_left_veers_right():
    d = depth(6.0)
    d[:, : W // 2] = 1.0
    g = nav().evaluate(d)
    assert g.status == "veer_right"
    assert g.heading_deg > 0


def test_unknown_depth_is_never_free():
    g = nav().evaluate(depth(np.inf))
    assert g.status == "blocked"


def test_guidance_is_spoken_on_change_only():
    n = nav()
    clear = n.evaluate(depth(6.0))
    assert n.spoken(clear, now=0.0) == "Path clear ahead."
    assert n.spoken(clear, now=1.0) is None
    blocked = n.evaluate(depth(1.0))
    assert n.spoken(blocked, now=2.0) == "Path blocked. Stop."
    assert n.spoken(blocked, now=3.0) is None          # not due yet
    assert n.spoken(blocked, now=9.0) == "Path blocked. Stop."   # repeats while unsafe
