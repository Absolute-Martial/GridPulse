from gridpulse.engine import GridPulseEngine


def test_dispatch_objective_metrics_present_and_bounded():
    engine = GridPulseEngine()
    snapshot = engine.advance_tick()
    dispatch = snapshot.dispatch

    assert dispatch.transmission_loss_kw >= 0.0
    assert dispatch.overload_penalty_kw >= 0.0
    assert dispatch.renewable_bonus_kw >= 0.0
    assert isinstance(dispatch.objective_score, float)


def test_fault_mode_keeps_dispatch_algorithm_active():
    engine = GridPulseEngine()
    engine.inject_fault("S2")
    snapshot = engine.advance_tick()

    assert snapshot.dispatch.degraded is True
    assert snapshot.dispatch.total_supplied_kw > 0
    assert snapshot.dispatch.routes
