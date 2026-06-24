import pytest
from catenets.datasets import load


@pytest.mark.parametrize("train_ratio", [0.5, 0.8])
@pytest.mark.parametrize("treatment_type", ["rand", "logistic"])
@pytest.mark.parametrize("treat_prop", [0.1, 0.9])
def test_dataset_sanity_twins(
    train_ratio: float, treatment_type: str, treat_prop: float
) -> None:
    x_train, W_train, Y_train, Y_train_full, x_test, Y_test = load(
        "twins",
        train_ratio=train_ratio,
        treatment_type=treatment_type,
        treat_prop=treat_prop,
    )

    total = x_train.shape[0] + x_test.shape[0]

    assert int(total * train_ratio) == x_train.shape[0]
    assert x_train.shape[1] == x_test.shape[1]
    assert x_train.shape[0] == Y_train.shape[0]
    assert x_train.shape[0] == Y_train_full.shape[0]
    assert x_train.shape[0] == W_train.shape[0]
    assert x_test.shape[0] == Y_test.shape[0]


def test_dataset_sanity_ihdp() -> None:
    x_train, W_train, Y_train, Y_train_full, x_test, Y_test = load("ihdp")

    assert x_train.shape[1] == x_test.shape[1]
    assert x_train.shape[0] == Y_train.shape[0]
    assert x_train.shape[0] == Y_train_full.shape[0]
    assert x_train.shape[0] == W_train.shape[0]
    assert x_test.shape[0] == Y_test.shape[0]


@pytest.mark.slow
@pytest.mark.parametrize("preprocessed", [False, True])
def test_dataset_sanity_acic2016(preprocessed: bool) -> None:
    x_train, W_train, Y_train, Y_train_full, x_test, Y_test = load(
        "acic2016", preprocessed=preprocessed
    )

    assert x_train.shape[1] == x_test.shape[1]
    assert x_train.shape[0] == Y_train.shape[0]
    assert x_train.shape[0] == Y_train_full.shape[0]
    assert x_train.shape[0] == W_train.shape[0]
    assert x_test.shape[0] == Y_test.shape[0]
