from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

class Pipeline:
    steps: list[tuple[str, Any]]
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self: ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...

def make_pipeline(*steps: object) -> Pipeline: ...
