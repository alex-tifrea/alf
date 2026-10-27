"""Posterior-variance batch-selection checks."""
import numpy as np
from alf_icmd.acquisition import activation_covariance, greedy_pv

def test_pv_matches_explicit_conditioning():
    rng = np.random.default_rng(12)
    train, pool = rng.normal(size=(7, 10)), rng.normal(size=(30, 10))
    selected = greedy_pv(activation_covariance(train, pool), 8, 1e-3)
    expected = []
    for _ in range(8):
        conditioning = np.vstack([train, pool[expected]])
        scores = np.diag(activation_covariance(conditioning, pool)).copy()
        scores[expected] = -np.inf
        expected.append(int(np.argmax(scores)))
    np.testing.assert_array_equal(selected, expected)
