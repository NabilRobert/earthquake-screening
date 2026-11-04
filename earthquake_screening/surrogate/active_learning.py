"""Pool-based active learning with a linear SVM (Sainct et al. 2020, sections 4.1 and 4.3).

The classifier learns failure / no failure from four record features, (L, PGA, PGV, w0),
each Box-Cox transformed and standardised. It starts from two labelled records, one
mild and one strong, and then asks for the label of the record it is least sure of
(the one closest to its separating hyperplane), one record at a time.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.intp]

# The paper describes a maximal-margin separator and gives no regularisation constant.
# Fixed before any real run: C = 1000 (near hard margin) took 150 ms a fit, too slow for
# 20 x 1,000 fits; C = 10 gave the same hyperplane within 0.1% in 6 ms (docs/spikes.md).
SVM_REGULARIZATION = 10.0
# The strong starting record comes from above this quantile of both PGA and L (section 4.3).
STRONG_START_QUANTILE = 0.9


@dataclass(frozen=True)
class Snapshot:
    """The classifier's state after a given number of labelled records."""

    labelled_indices: IndexArray
    scores: FloatArray  # signed distance to the hyperplane for every record in the pool


def standardised_features(raw_features: FloatArray) -> FloatArray:
    """Box-Cox per column (exponent by maximum likelihood, Eq. 21), then zero mean and unit variance."""
    return np.asarray(PowerTransformer(method="box-cox", standardize=True).fit_transform(raw_features))


def _draw_until(candidates: IndexArray, wanted_failure: bool, has_failed: BoolArray, rng: np.random.Generator) -> int:
    """A random candidate with the wanted label; the paper redraws until it gets one."""
    matching = candidates[has_failed[candidates] == wanted_failure]
    if matching.size == 0:
        raise ValueError(f"no starting candidate has failure={wanted_failure}: the pool is too small or one-sided")
    return int(rng.choice(matching))


def starting_pair(
    pga: FloatArray, linear_peak: FloatArray, has_failed: BoolArray, rng: np.random.Generator
) -> IndexArray:
    """One mild record (PGA and L below their medians) that didn't fail, one strong one that did."""
    mild = np.flatnonzero((pga < np.median(pga)) & (linear_peak < np.median(linear_peak)))
    strong = np.flatnonzero(
        (pga > np.quantile(pga, STRONG_START_QUANTILE))
        & (linear_peak > np.quantile(linear_peak, STRONG_START_QUANTILE))
    )
    return np.array([_draw_until(mild, False, has_failed, rng), _draw_until(strong, True, has_failed, rng)])


def run_active_learning(
    features: FloatArray, has_failed: BoolArray, starting_indices: IndexArray, budgets: tuple[int, ...]
) -> dict[int, Snapshot]:
    """Label records one at a time, closest to the hyperplane first; snapshot at each budget."""
    is_labelled = np.zeros(len(features), dtype=bool)
    is_labelled[starting_indices] = True
    labelled_order = list(starting_indices)
    snapshots: dict[int, Snapshot] = {}
    while len(labelled_order) <= max(budgets):
        classifier = SVC(kernel="linear", C=SVM_REGULARIZATION).fit(
            features[labelled_order], has_failed[labelled_order]
        )
        scores = np.asarray(classifier.decision_function(features), dtype=np.float64)
        if len(labelled_order) in budgets:
            snapshots[len(labelled_order)] = Snapshot(np.array(labelled_order), scores)
        if len(labelled_order) == max(budgets):
            break
        unlabelled = np.flatnonzero(~is_labelled)
        next_index = int(unlabelled[np.argmin(np.abs(scores[unlabelled]))])
        is_labelled[next_index] = True
        labelled_order.append(next_index)
    return snapshots
