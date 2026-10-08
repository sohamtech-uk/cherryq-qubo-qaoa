from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


FEATURE_NAMES = (
    "invoice_amount_gbp",
    "payment_terms_days",
    "days_until_due",
    "customer_avg_days_late_last_5",
    "customer_on_time_rate_90d",
    "customer_invoice_count",
)


@dataclass(frozen=True)
class ForecastDataset:
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


@dataclass(frozen=True)
class ForecastReport:
    model: GradientBoostingClassifier
    train_size: int
    test_size: int
    roc_auc: float
    brier_score: float
    baseline_brier_score: float
    train_positive_rate: float
    test_positive_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "roc_auc": self.roc_auc,
            "brier_score": self.brier_score,
            "baseline_brier_score": self.baseline_brier_score,
            "train_positive_rate": self.train_positive_rate,
            "test_positive_rate": self.test_positive_rate,
        }


def generate_synthetic_payment_history(
    n_samples: int = 1200,
    seed: int = 42,
) -> ForecastDataset:
    """Create deterministic synthetic invoice-payment history for the prototype.

    The rows are ordered in time. The generated label means "paid before the
    next payment run". This dataset is only for engineering the workflow; it is
    not evidence that the forecasting model works on real SME data.
    """

    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")

    rng = np.random.default_rng(seed)
    time_index = np.linspace(0.0, 1.0, n_samples)

    amount = np.exp(rng.normal(np.log(1600.0), 0.8, n_samples)).clip(100, 25000)
    payment_terms = rng.choice(
        [7, 14, 30, 45, 60],
        size=n_samples,
        p=[0.05, 0.15, 0.5, 0.15, 0.15],
    )
    days_until_due = rng.integers(-30, 31, size=n_samples)
    avg_days_late = rng.normal(6.0 + 5.0 * time_index, 9.0, n_samples).clip(-10, 60)
    on_time_rate = rng.beta(5.0, 2.2, size=n_samples)
    invoice_count = rng.integers(2, 80, size=n_samples)

    # A simple synthetic data-generating process. Later rows are slightly harder
    # to predict/pay, creating a mild temporal drift for the held-out period.
    logit = (
        0.2
        + 2.4 * (on_time_rate - 0.5)
        - 0.07 * avg_days_late
        + 0.035 * days_until_due
        - 0.000045 * amount
        + 0.006 * np.minimum(invoice_count, 40)
        - 0.5 * time_index
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    labels = rng.binomial(1, probability).astype(int)

    features = np.column_stack(
        [
            amount,
            payment_terms,
            days_until_due,
            avg_days_late,
            on_time_rate,
            invoice_count,
        ]
    ).astype(float)

    return ForecastDataset(features=features, labels=labels)


def train_payment_classifier(
    dataset: ForecastDataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> ForecastReport:
    """Train on earlier rows and evaluate on later rows to avoid future leakage."""

    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")

    split = int(len(dataset.labels) * train_fraction)
    x_train = dataset.features[:split]
    y_train = dataset.labels[:split]
    x_test = dataset.features[split:]
    y_test = dataset.labels[split:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Both train and test sets must contain both classes")

    model = GradientBoostingClassifier(
        random_state=seed,
        n_estimators=120,
        learning_rate=0.04,
        max_depth=2,
        subsample=0.9,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    train_prevalence = float(np.mean(y_train))
    baseline_probabilities = np.full_like(probabilities, train_prevalence, dtype=float)

    return ForecastReport(
        model=model,
        train_size=len(y_train),
        test_size=len(y_test),
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        brier_score=float(brier_score_loss(y_test, probabilities)),
        baseline_brier_score=float(brier_score_loss(y_test, baseline_probabilities)),
        train_positive_rate=train_prevalence,
        test_positive_rate=float(np.mean(y_test)),
    )


def predict_payment_probability(
    model: GradientBoostingClassifier,
    feature_rows: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    rows = np.asarray(feature_rows, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features in order {FEATURE_NAMES}, got {rows.shape[1]}"
        )
    return model.predict_proba(rows)[:, 1]
