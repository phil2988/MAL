from euclidianDistances import L1

def MAE(y_pred, y_true):
    assert y_pred.ndim == 1, f"Expected a vector, y_pred is a {y_pred.ndim}d array"
    assert y_true.ndim == 1, f"Expected a vector, y_true is a {y_true.ndim}d array"
    assert len(y_true) == len(y_pred), f"Vectors are not same length. y_true is {len(y_true)} and y_pred is {len(y_pred)}"

    print(f"y_pred is {y_pred}")
    print(f"y_true is {y_true}")

    l1 = L1(y_pred - y_true)
    print(f"L1 is {l1}")

    mean = l1 / len(y_pred)
    print(f"Mean is {mean}")

    return mean