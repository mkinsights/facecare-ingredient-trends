from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


# Konfiguracja

root = Path(__file__).resolve().parents[1]
benchmark_dir = root / "processed" / "sentiment" / "benchmark"
predictions_csv = benchmark_dir / "supervised_sentiment_test_predictions.csv"
comparison_csv = benchmark_dir / "supervised_sentiment_pairwise_comparison.csv"

sentiment_labels = ["negative", "neutral", "positive"]
bootstrap_iterations = 1_000
random_state = 42


# Porownanie sparowane z bootstrapem klastrowym po produktach


def macro_f1(actual: pd.Series, predicted: pd.Series) -> float:
    return f1_score(
        actual,
        predicted,
        labels=sentiment_labels,
        average="macro",
        zero_division=0,
    )


def cluster_bootstrap_differences(
    data: pd.DataFrame,
    prediction_a: str,
    prediction_b: str,
) -> tuple[float, float, float, float]:
    """Szacuje przedzialy roznic, losujac cale produkty zamiast wierszy."""
    rng = np.random.default_rng(random_state)
    product_ids, product_index = np.unique(data["id"].to_numpy(), return_inverse=True)
    actual = data["manual_sentiment"].to_numpy()
    predicted_a = data[prediction_a].to_numpy()
    predicted_b = data[prediction_b].to_numpy()
    accuracy_differences: list[float] = []
    macro_f1_differences: list[float] = []

    for _ in range(bootstrap_iterations):
        sampled_product_positions = rng.integers(
            0,
            len(product_ids),
            size=len(product_ids),
        )
        product_weights = np.bincount(
            sampled_product_positions,
            minlength=len(product_ids),
        )
        row_weights = product_weights[product_index]
        accuracy_differences.append(
            accuracy_score(actual, predicted_a, sample_weight=row_weights)
            - accuracy_score(actual, predicted_b, sample_weight=row_weights)
        )
        macro_f1_differences.append(
            f1_score(
                actual,
                predicted_a,
                labels=sentiment_labels,
                average="macro",
                zero_division=0,
                sample_weight=row_weights,
            )
            - f1_score(
                actual,
                predicted_b,
                labels=sentiment_labels,
                average="macro",
                zero_division=0,
                sample_weight=row_weights,
            )
        )

    accuracy_interval = np.quantile(accuracy_differences, [0.025, 0.975])
    macro_f1_interval = np.quantile(macro_f1_differences, [0.025, 0.975])
    return (
        float(accuracy_interval[0]),
        float(accuracy_interval[1]),
        float(macro_f1_interval[0]),
        float(macro_f1_interval[1]),
    )


def compare_models(
    data: pd.DataFrame,
    comparison: str,
    prediction_a: str,
    prediction_b: str,
) -> dict[str, object]:
    """Porownuje dwa modele na identycznych obserwacjach."""
    data = data.dropna(
        subset=["manual_sentiment", prediction_a, prediction_b]
    ).copy()
    actual = data["manual_sentiment"]
    correct_a = data[prediction_a].eq(actual)
    correct_b = data[prediction_b].eq(actual)
    a_only_correct = int((correct_a & ~correct_b).sum())
    b_only_correct = int((~correct_a & correct_b).sum())
    discordant = a_only_correct + b_only_correct
    mcnemar_p = (
        binomtest(a_only_correct, discordant, p=0.5).pvalue
        if discordant
        else 1.0
    )
    (
        accuracy_ci_lower,
        accuracy_ci_upper,
        macro_f1_ci_lower,
        macro_f1_ci_upper,
    ) = cluster_bootstrap_differences(data, prediction_a, prediction_b)

    accuracy_a = accuracy_score(actual, data[prediction_a])
    accuracy_b = accuracy_score(actual, data[prediction_b])
    macro_f1_a = macro_f1(actual, data[prediction_a])
    macro_f1_b = macro_f1(actual, data[prediction_b])
    return {
        "comparison": comparison,
        "rows": len(data),
        "products": data["id"].nunique(),
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "accuracy_difference_a_minus_b": accuracy_a - accuracy_b,
        "accuracy_difference_ci_95_lower": accuracy_ci_lower,
        "accuracy_difference_ci_95_upper": accuracy_ci_upper,
        "macro_f1_a": macro_f1_a,
        "macro_f1_b": macro_f1_b,
        "macro_f1_difference_a_minus_b": macro_f1_a - macro_f1_b,
        "macro_f1_difference_ci_95_lower": macro_f1_ci_lower,
        "macro_f1_difference_ci_95_upper": macro_f1_ci_upper,
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "mcnemar_exact_p": mcnemar_p,
    }


def main() -> None:
    predictions = pd.read_csv(predictions_csv)
    covered = predictions.loc[
        predictions["aspect_match"].eq(True)
        & predictions["model_sentiment_for_manual_aspect"].notna()
    ].copy()

    rows = [
        compare_models(
            predictions,
            "supervised_text_vs_review_rating_baseline_all_test",
            "supervised_sentiment",
            "rating_baseline_sentiment",
        ),
        compare_models(
            covered,
            "supervised_text_vs_current_hybrid_covered_subset",
            "supervised_sentiment",
            "model_sentiment_for_manual_aspect",
        ),
        compare_models(
            covered,
            "current_hybrid_vs_review_rating_baseline_covered_subset",
            "model_sentiment_for_manual_aspect",
            "rating_baseline_sentiment",
        ),
    ]
    result = pd.DataFrame(rows).round(2)
    result.to_csv(
        comparison_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    print(result.to_string(index=False))
    print(f"Saved paired comparisons to {comparison_csv}")


if __name__ == "__main__":
    main()
