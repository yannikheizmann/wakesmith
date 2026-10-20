from wakesmith.config import TrainingArgs
from wakesmith.evaluation import objective, operating_point


def metrics(rates: dict[str, float], recall: float, adversarial: float = 0.0) -> dict:
    return {
        "recall": recall,
        "adversarial_accept": adversarial,
        "false_activations_per_hour": rates,
    }


ARGS = TrainingArgs()


def test_default_threshold_is_kept_when_feasible():
    grid = {"0.3": metrics({"a": 0.0}, 0.9), "0.5": metrics({"a": 0.0}, 0.8)}
    assert operating_point(grid, ARGS) == (0.5, 0.8)


def test_higher_threshold_when_default_activates_too_often():
    grid = {"0.5": metrics({"a": 1.0}, 0.8), "0.7": metrics({"a": 0.1}, 0.7)}
    assert operating_point(grid, ARGS) == (0.7, 0.7)


def test_adversarial_limit_applies():
    grid = {"0.5": metrics({"a": 0.0}, 0.8, adversarial=0.05), "0.9": metrics({"a": 0.0}, 0.5)}
    assert operating_point(grid, ARGS) == (0.9, 0.5)


def test_infeasible_checkpoints_rank_by_false_activations():
    worse = {"0.5": metrics({"a": 3.0}, 0.9)}
    better = {"0.5": metrics({"a": 1.0}, 0.9)}
    assert operating_point(worse, ARGS) == (None, 0.0)
    assert objective(better, ARGS) > objective(worse, ARGS)
