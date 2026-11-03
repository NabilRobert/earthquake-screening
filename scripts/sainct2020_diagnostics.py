"""Post-hoc checks on why the Sainct et al. (2020) reproduction missed. They don't change the verdict.

1. Noise floor: the Delta L2 that even a perfect estimator shows, from the binomial noise
   in each k-means group's reference failure rate, for 10, 20 and 50 groups.
2. Passive ceiling: a linear SVM (and a plain logistic regression) given 5,000 random
   labels, five times the paper's largest budget, with no active learning. If even
   these miss the paper's numbers, the gap is in the data, not the learning loop.

Run: uv run python scripts/sainct2020_diagnostics.py  (needs data/generated/pool.npz)
"""

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from earthquake_screening.reproduction.sainct2020 import four_feature_matrix
from earthquake_screening.simulation.pool import Pool
from earthquake_screening.surrogate.active_learning import SVM_REGULARIZATION, standardised_features
from earthquake_screening.surrogate.fragility import failure_probabilities, fragility_error, group_by_value

GROUP_COUNTS = (10, 20, 50)
PASSIVE_LABELS = 5000
SEED = 2020


def main() -> None:
    pool = Pool.load("data/generated/pool.npz")
    rows = np.flatnonzero(pool.labelled_rows)
    has_failed = pool.has_failed[rows]
    raw = four_feature_matrix(pool, rows)
    features = standardised_features(raw)
    measures = {"pga": raw[:, 1], "linear_peak": raw[:, 0]}
    groups = {
        (name, count): group_by_value(values, count, SEED)
        for name, values in measures.items()
        for count in GROUP_COUNTS
    }
    print("Noise floor of a perfect estimator, Delta L2 %:")
    for (name, count), grouping in groups.items():
        sizes = np.bincount(grouping)
        rates = np.bincount(grouping, weights=has_failed.astype(np.float64)) / sizes
        print(f"  {name:12s} K={count:3d}: {100 * math.sqrt(np.sum(rates * (1 - rates)) / rows.size):.2f}")

    labelled = np.random.default_rng(SEED).choice(rows.size, PASSIVE_LABELS, replace=False)
    scores = SVC(kernel="linear", C=SVM_REGULARIZATION).fit(features[labelled], has_failed[labelled])
    estimates = {
        "linear SVM + logistic": failure_probabilities(scores.decision_function(features), labelled, has_failed),
        "logistic on the 4 features": LogisticRegression(C=math.inf)
        .fit(features[labelled], has_failed[labelled])
        .predict_proba(features)[:, 1],
    }
    print(f"Passive, {PASSIVE_LABELS} random labels, Delta L2 %:")
    for label, probabilities in estimates.items():
        errors = ", ".join(
            f"{name} K={count}: {100 * fragility_error(probabilities, has_failed, grouping):.2f}"
            for (name, count), grouping in groups.items()
        )
        print(f"  {label}: {errors}")


if __name__ == "__main__":
    main()
