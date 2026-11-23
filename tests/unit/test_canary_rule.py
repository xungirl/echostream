from echostream.rollout.controller import canary_reasons

CTL = {"requests": 1000, "errors": 0, "degraded": 10, "p99_ms": 20.0}


def test_healthy_canary_is_promoted():
    can = {"requests": 100, "errors": 0, "degraded": 1, "p99_ms": 25.0}
    assert canary_reasons(CTL, can, (-0.02, 0.01)) == []


def test_each_rule_triggers_rollback():
    assert "error rate" in canary_reasons(CTL, {**CTL, "requests": 100, "errors": 2, "degraded": 0}, (-0.01, 0.01))[0]
    assert "p99" in canary_reasons(CTL, {"requests": 100, "errors": 0, "degraded": 0, "p99_ms": 80.0}, (-0.01, 0.01))[0]
    assert "degraded" in canary_reasons(CTL, {"requests": 100, "errors": 0, "degraded": 20, "p99_ms": 20.0}, (-0.01, 0.01))[0]
    assert "online quality" in canary_reasons(CTL, {"requests": 100, "errors": 0, "degraded": 0, "p99_ms": 20.0}, (-0.05, -0.01))[0]


def test_p99_threshold_scales_with_control():
    slow_ctl = {**CTL, "p99_ms": 40.0}
    assert canary_reasons(slow_ctl, {"requests": 100, "errors": 0, "degraded": 0, "p99_ms": 70.0}, (0, 0.1)) == []
