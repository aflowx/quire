from quire.calibration import TypeTemperature


def test_temperature_never_changes_the_answer_and_moves_confidence_the_right_way():
    p = {"a": 0.6, "b": 0.3, "c": 0.1}
    sharp = TypeTemperature({"choice": 0.5}).apply(p, "choice")
    soft = TypeTemperature({"choice": 2.0}).apply(p, "choice")
    assert max(sharp, key=sharp.get) == max(soft, key=soft.get) == "a"
    assert sharp["a"] > 0.6 > soft["a"]
    assert abs(sum(sharp.values()) - 1) < 1e-9 and abs(sum(soft.values()) - 1) < 1e-9


def test_each_type_uses_its_own_temperature():
    tt = TypeTemperature({"choice": 1.0, "noul": 0.5, "score": 1.0})
    p = {"true": 0.7, "false": 0.3}
    assert tt.apply(p, "choice") == p
    assert tt.apply(p, "noul")["true"] > 0.7
