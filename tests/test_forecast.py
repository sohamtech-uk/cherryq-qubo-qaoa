import numpy as np

from cherryq.forecast import (
    FEATURE_NAMES,
    generate_synthetic_payment_history,
    predict_payment_probability,
    train_payment_classifier,
)


def test_forecast_pipeline_uses_temporal_holdout_and_beats_prevalence_brier():
    dataset = generate_synthetic_payment_history(n_samples=400, seed=42)
    report = train_payment_classifier(dataset, seed=42)

    assert report.train_size == 320
    assert report.test_size == 80
    assert 0.5 < report.roc_auc <= 1.0
    assert report.brier_score < report.baseline_brier_score


def test_predict_payment_probability_shape_and_bounds():
    dataset = generate_synthetic_payment_history(n_samples=400, seed=42)
    report = train_payment_classifier(dataset, seed=42)

    probabilities = predict_payment_probability(report.model, dataset.features[-3:])

    assert probabilities.shape == (3,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert dataset.features.shape[1] == len(FEATURE_NAMES)
