from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

class GroupKFold:
    def __init__(self, n_splits: int = ...) -> None: ...
    def split(
        self, X: NDArray[np.float64], y: NDArray[np.float64], groups: NDArray[np.int64]
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

def train_test_split(
    array: NDArray[np.int64], *, test_size: float, stratify: NDArray[Any] | None = ..., random_state: int | None = ...
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
