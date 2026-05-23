from gridpulse.engine import GridPulseEngine


def test_engine_advance_tick_produces_operator_snapshot():
    engine = GridPulseEngine()

    snapshot = engine.advance_tick()

    assert snapshot.tick == 0
    assert snapshot.scenario.name == "normal"
    assert len(snapshot.zone_states) == 8
    assert snapshot.prediction.label in {"Critical", "Stressed", "Normal", "Surplus"}
    assert snapshot.dispatch.total_demand_kw > 0
    assert snapshot.dispatch.total_supplied_kw > 0
    assert len(snapshot.suggestions) >= 1


def test_engine_fault_injection_surfaces_reroute_guidance():
    engine = GridPulseEngine()

    engine.inject_fault("S2")
    snapshot = engine.advance_tick()

    assert "S2" in snapshot.faulted_nodes
    assert snapshot.dispatch.degraded is True
    assert any(item.category in {"reroute", "inspect"} for item in snapshot.suggestions)
