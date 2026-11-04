"""Fragility curves from the classifier, and their error against the full simulation (section 5).

- The SVM's score becomes a probability of failure through a logistic fit on the
  labelled records only (Eq. 35).
- A fragility curve against an intensity measure (PGA or L) is the mean probability
  within groups of records with similar values, grouped by k-means (Eq. 38).
- The reference curve is the actual failure rate in the same groups, from all records.
- The error is the record-weighted root mean square gap between them, Delta L2 (Eq. 37).
"""

import math

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.intp]


def failure_probabilities(scores: FloatArray, labelled_indices: IndexArray, has_failed: BoolArray) -> FloatArray:
    """p(X) = 1 / (1 + exp(-a*score + b)), with a and b fitted on the labelled records only (Eq. 35)."""
    # C = infinity means no penalty: the paper fits a plain logistic regression, and
    # scikit-learn's default adds one.
    logistic = LogisticRegression(C=math.inf).fit(scores[labelled_indices].reshape(-1, 1), has_failed[labelled_indices])
    return np.asarray(logistic.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float64)


def group_by_value(values: FloatArray, group_count: int, seed: int) -> IndexArray:
    """k-means groups of records with similar values of one measure (section 5.2)."""
    kmeans = KMeans(n_clusters=group_count, n_init=10, random_state=seed)
    return np.asarray(kmeans.fit_predict(values.reshape(-1, 1)), dtype=np.intp)


def fragility_error(estimated_probabilities: FloatArray, has_failed: BoolArray, groups: IndexArray) -> float:
    """Delta L2 of Eq. 37 as a fraction: sqrt(sum_k n_k (p_ref_k - p_est_k)^2 / N)."""
    group_sizes = np.bincount(groups)
    is_present = group_sizes > 0
    reference = np.bincount(groups, weights=has_failed.astype(np.float64))[is_present] / group_sizes[is_present]
    estimated = np.bincount(groups, weights=estimated_probabilities)[is_present] / group_sizes[is_present]
    squared_gap = group_sizes[is_present] * (reference - estimated) ** 2
    return float(np.sqrt(np.sum(squared_gap) / groups.size))
