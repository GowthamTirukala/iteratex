from iteratex.evaluation.evaluator import Evaluator


def test_should_promote_when_no_prod():
    ev = Evaluator(primary_metric="accuracy")
    assert ev.should_promote({"accuracy": 0.5})
