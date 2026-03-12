import numpy as np
from ramp_ml.eval import match_events_with_tolerance


def test_match_events():
    pred = np.array([10, 50, 90])
    gt = np.array([12, 49, 200])
    tp, fp, fn = match_events_with_tolerance(pred, gt, tol=2)
    assert tp == 2
    assert fp == 1
    assert fn == 1