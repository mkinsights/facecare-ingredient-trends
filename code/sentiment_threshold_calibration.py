from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import evaluate_sentiment_post_tuning_validation as validation


# Konfiguracja kalibracji

root = Path(__file__).resolve().parents[1]
sentiment_dir = root / "processed" / "sentiment"
annotation_dir = sentiment_dir / "validation"
output_dir = sentiment_dir / "calibration"
mentions_csv = sentiment_dir / "model" / "review_aspect_sentiment.csv"

thresholds_csv = output_dir / "sentiment_class_thresholds.csv"
comparison_csv = output_dir / "sentiment_threshold_validation_comparison.csv"
methodology_csv = output_dir / "sentiment_threshold_calibration_methodology.csv"

annotation_sets = {
    "development": (
        annotation_dir / "manual_annotation_sample_200.csv",
        annotation_dir / "manual_annotation_key_200.csv",
    ),
    "initial_validation": (
        annotation_dir / "initial" / "manual_validation_sample_150.csv",
        annotation_dir / "initial" / "manual_validation_key_150.csv",
    ),
    "post_tuning_validation": (
        annotation_dir
        / "tuned"
        / "manual_post_tuning_validation_sample_150.csv",
        annotation_dir
        / "tuned"
        / "manual_post_tuning_validation_key_150.csv",
    ),
    "stratified_validation": (
        annotation_dir
        / "final"
        / "manual_stratified_validation_sample_240.csv",
        annotation_dir
        / "final"
        / "manual_stratified_validation_key_240.csv",
    ),
}

sentiment_labels = ["negative", "neutral", "positive"]
pre_calibration_negative_threshold = -0.25
pre_calibration_positive_threshold = 0.25


# Przygotowanie niezależnych zbiorów anotacji


def load_scores() -> pd.Series:
    """Tworzy mapę wyniku ciągłego dla fragmentu i aspektu."""
    mentions = pd.read_csv(mentions_csv)
    mentions["_fragment_key"] = (
        mentions["review_id"].astype(str)
        + "_"
        + mentions["fragment_index"].astype(str)
    )
    return (
        mentions.drop_duplicates(["_fragment_key", "aspect"])
        .set_index(["_fragment_key", "aspect"])["sentiment_score"]
    )


def load_annotation_set(
    sample_path: Path,
    key_path: Path,
    score_map: pd.Series,
) -> pd.DataFrame:
    """Łączy ręczne etykiety z ciągłym wynikiem bieżącego modelu."""
    validation.base_evaluation.sample_csv = sample_path
    validation.base_evaluation.key_csv = key_path
    data, _ = validation.base_evaluation.load_annotations()
    data = validation.base_evaluation.add_evaluation_columns(data)
    data = validation.add_multilabel_columns(data)
    data = data.loc[data["aspect_match"]].copy()
    data["sentiment_score"] = [
        score_map.get((fragment, aspect), np.nan)
        for fragment, aspect in zip(
            data["_fragment_key"],
            data["manual_aspect"],
        )
    ]
    return data.dropna(subset=["sentiment_score"])


def classify_scores(
    scores: pd.Series,
    negative_threshold: float,
    positive_threshold: float,
) -> np.ndarray:
    """Przekształca wynik ciągły na trzy uporządkowane klasy."""
    return np.select(
        [
            scores.to_numpy() < negative_threshold,
            scores.to_numpy() < positive_threshold,
        ],
        ["negative", "neutral"],
        default="positive",
    )


# Dobór progów wyłącznie na próbie rozwojowej


def optimize_thresholds(development: pd.DataFrame) -> tuple[float, float]:
    """Maksymalizuje macro F1 bez zaglądania do zbiorów walidacyjnych."""
    best: tuple[float, float, float, float] | None = None
    for negative_threshold in np.arange(-0.60, 0.11, 0.05):
        for positive_threshold in np.arange(
            negative_threshold + 0.05,
            0.61,
            0.05,
        ):
            prediction = classify_scores(
                development["sentiment_score"],
                negative_threshold,
                positive_threshold,
            )
            macro_f1 = f1_score(
                development["manual_sentiment"],
                prediction,
                labels=sentiment_labels,
                average="macro",
                zero_division=0,
            )
            accuracy = accuracy_score(
                development["manual_sentiment"],
                prediction,
            )
            candidate = (
                macro_f1,
                accuracy,
                float(negative_threshold),
                float(positive_threshold),
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise RuntimeError("Could not optimize sentiment thresholds")
    return round(best[2], 2), round(best[3], 2)


def evaluate_thresholds(
    datasets: dict[str, pd.DataFrame],
    threshold_name: str,
    negative_threshold: float,
    positive_threshold: float,
) -> list[dict[str, object]]:
    """Liczy metryki tych samych progów na każdym zbiorze."""
    rows: list[dict[str, object]] = []
    for dataset_name, data in datasets.items():
        prediction = classify_scores(
            data["sentiment_score"],
            negative_threshold,
            positive_threshold,
        )
        rows.append(
            {
                "threshold_version": threshold_name,
                "dataset": dataset_name,
                "dataset_role": (
                    "optimization"
                    if dataset_name == "development"
                    else "held_out_validation"
                ),
                "evaluable_mentions": len(data),
                "accuracy": accuracy_score(
                    data["manual_sentiment"],
                    prediction,
                ),
                "macro_f1": f1_score(
                    data["manual_sentiment"],
                    prediction,
                    labels=sentiment_labels,
                    average="macro",
                    zero_division=0,
                ),
                "predicted_negative": int((prediction == "negative").sum()),
                "predicted_neutral": int((prediction == "neutral").sum()),
                "predicted_positive": int((prediction == "positive").sum()),
            }
        )
    return rows


# Zapis audytu


def save_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_map = load_scores()
    datasets = {
        name: load_annotation_set(sample, key, score_map)
        for name, (sample, key) in annotation_sets.items()
    }
    calibrated_negative, calibrated_positive = optimize_thresholds(
        datasets["development"]
    )

    threshold_table = pd.DataFrame(
        [
            {
                "threshold_version": "pre_calibration",
                "negative_upper_bound": pre_calibration_negative_threshold,
                "neutral_upper_bound": pre_calibration_positive_threshold,
                "derived_from": "original rating interpretation",
            },
            {
                "threshold_version": "implemented_development_calibrated",
                "negative_upper_bound": calibrated_negative,
                "neutral_upper_bound": calibrated_positive,
                "derived_from": "manual development sample macro F1",
            },
        ]
    )
    save_csv(threshold_table, thresholds_csv)

    rows = evaluate_thresholds(
        datasets,
        "pre_calibration",
        pre_calibration_negative_threshold,
        pre_calibration_positive_threshold,
    )
    rows.extend(
        evaluate_thresholds(
            datasets,
            "implemented_development_calibrated",
            calibrated_negative,
            calibrated_positive,
        )
    )
    comparison = pd.DataFrame(rows)
    save_csv(comparison, comparison_csv)

    methodology = pd.DataFrame(
        [
            ("optimization_metric", "three-class macro F1"),
            ("optimization_dataset", "manual development sample only"),
            ("threshold_grid_step", 0.05),
            (
                "held_out_sets",
                "initial, post-tuning, and stratified validation",
            ),
            (
                "selection_bias_note",
                "stratified validation oversamples difficult model predictions",
            ),
            (
                "population_prevalence",
                "accuracy is diagnostic and not prevalence-weighted",
            ),
        ],
        columns=["parameter", "value"],
    )
    save_csv(methodology, methodology_csv)

    print(
        "Calibrated sentiment score thresholds: "
        f"negative < {calibrated_negative:.2f}, "
        f"neutral < {calibrated_positive:.2f}"
    )
    print(f"Saved calibration audit to {output_dir}")


if __name__ == "__main__":
    main()
