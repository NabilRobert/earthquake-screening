from typing import Self

import numpy as np
from numpy.typing import NDArray

class HistGradientBoostingRegressor:
    def __init__(
        self,
        *,
        learning_rate: float = ...,
        max_iter: int = ...,
        early_stopping: bool | str = ...,
        random_state: int | None = ...,
    ) -> None: ...
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self: ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
