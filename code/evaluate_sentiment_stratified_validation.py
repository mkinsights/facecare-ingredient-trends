from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import classification_report

import evaluate_sentiment_post_tuning_validation as validation
from validation_metrics import (
    model_quality_by_category,
    model_quality_by_category_aspect,
    model_quality_by_category_sentiment,
)


# Konfiguracja rozszerzonej walidacji

root = Path(__file__).resolve().parents[1]
validation_dir = (
    root
    / "processed"
    / "sentiment"
    / "validation"
    / "final"
)

sample_csv = validation_dir / "manual_stratified_validation_sample_240.csv"
key_csv = validation_dir / "manual_stratified_validation_key_240.csv"

evaluation_rows_csv = validation_dir / "stratified_validation_evaluation_rows.csv"
evaluation_summary_csv = (
    validation_dir / "stratified_validation_evaluation_summary.csv"
)
aspect_performance_csv = (
    validation_dir / "stratified_validation_aspect_performance.csv"
)
sentiment_confusion_csv = (
    validation_dir / "stratified_validation_sentiment_confusion.csv"
)
sentiment_class_performance_csv = (
    validation_dir / "stratified_validation_sentiment_class_performance.csv"
)
confidence_intervals_csv = (
    validation_dir / "stratified_validation_confidence_intervals.csv"
)
error_rows_csv = validation_dir / "stratified_validation_error_rows.csv"
category_quality_csv = validation_dir / "stratified_model_quality_by_category1.csv"
category_aspect_quality_csv = (
    validation_dir / "stratified_model_quality_by_category1_aspect.csv"
)
category_sentiment_quality_csv = (
    validation_dir / "stratified_model_quality_by_category1_sentiment.csv"
)
methodology_csv = validation_dir / "stratified_validation_methodology.csv"
continuous_category_csv = (
    validation_dir / "stratified_continuous_score_quality_by_category1.csv"
)
continuous_aspect_csv = (
    validation_dir / "stratified_continuous_score_quality_by_aspect.csv"
)
aspect_reliability_csv = (
    validation_dir / "stratified_aspect_measurement_reliability.csv"
)


# Wczytanie ręcznych etykiet i predykcji modelu


def load_evaluated_data() -> tuple[pd.DataFrame, int]:
    """Łączy ślepą anotację, klucz losowania i predykcje OOF modelu."""
    validation.base_evaluation.sample_csv = sample_csv
    validation.base_evaluation.key_csv = key_csv
    data, duplicate_count = validation.base_evaluation.load_annotations()
    data = validation.base_evaluation.add_evaluation_columns(data)
    data = validation.add_multilabel_columns(data)
    data = add_continuous_sentiment_scores(data)
    return data, duplicate_count


def add_continuous_sentiment_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Dołącza wynik ciągły modelu dla ręcznie wskazanego aspektu."""
    mentions = pd.read_csv(validation.base_evaluation.mentions_csv)
    mentions["_fragment_key"] = (
        mentions["review_id"].astype(str)
        + "_"
        + mentions["fragment_index"].astype(str)
    )
    scores = (
        mentions.drop_duplicates(["_fragment_key", "aspect"])
        .set_index(["_fragment_key", "aspect"])["sentiment_score"]
    )
    data = data.copy()
    data["model_sentiment_score_for_manual_aspect"] = [
        scores.get((fragment, aspect), np.nan)
        for fragment, aspect in zip(
            data["_fragment_key"],
            data["manual_aspect"],
        )
    ]
    data["manual_sentiment_score"] = data["manual_sentiment"].map(
        {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    )
    return data


def save_csv(data: pd.DataFrame, path: Path, include_index: bool = False) -> None:
    """Zapisuje czytelny CSV ze stałym formatem liczb."""
    data.to_csv(
        path,
        index=include_index,
        encoding="utf-8-sig",
        float_format="%.2f",
    )


def continuous_score_quality(
    data: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    """Ocenia zgodność porządkową i błąd ciągłego sentiment_score."""
    matched = data.loc[data["aspect_match"]].dropna(
        subset=[
            "manual_sentiment_score",
            "model_sentiment_score_for_manual_aspect",
        ]
    )
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = (
        grouping_columns[0] if len(grouping_columns) == 1 else grouping_columns
    )
    for keys, group in matched.groupby(grouper, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(grouping_columns, key_values))
        error = (
            group["model_sentiment_score_for_manual_aspect"]
            - group["manual_sentiment_score"]
        )
        if (
            len(group) >= 3
            and group["manual_sentiment_score"].nunique() >= 2
            and group["model_sentiment_score_for_manual_aspect"].nunique() >= 2
        ):
            rho = float(
                spearmanr(
                    group["manual_sentiment_score"],
                    group["model_sentiment_score_for_manual_aspect"],
                ).statistic
            )
        else:
            rho = np.nan
        row.update(
            {
                "evaluable_mentions": len(group),
                "spearman_rho": rho,
                "mean_error_model_minus_manual": error.mean(),
                "mean_absolute_error": error.abs().mean(),
                "root_mean_squared_error": np.sqrt((error**2).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_aspect_measurement_reliability(
    data: pd.DataFrame,
    aspect_performance: pd.DataFrame,
    continuous_aspect: pd.DataFrame,
) -> pd.DataFrame:
    """Łączy detekcję aspektu z jakością jego ciągłego sentymentu."""
    matched = data.loc[data["aspect_match"]]
    sentiment_accuracy = (
        matched.groupby("manual_aspect", as_index=False)
        .agg(
            sentiment_evaluable_mentions=("annotation_id", "size"),
            sentiment_accuracy=("sentiment_match_same_aspect", "mean"),
        )
        .rename(columns={"manual_aspect": "aspect"})
    )
    reliability = (
        aspect_performance.merge(
            sentiment_accuracy,
            on="aspect",
            how="left",
            validate="one_to_one",
        )
        .merge(
            continuous_aspect[
                [
                    "aspect",
                    "spearman_rho",
                    "mean_error_model_minus_manual",
                    "mean_absolute_error",
                ]
            ],
            on="aspect",
            how="left",
            validate="one_to_one",
        )
    )
    adequate = (
        reliability["sentiment_evaluable_mentions"].ge(15)
        & reliability["recall"].ge(0.70)
        & reliability["spearman_rho"].ge(0.50)
    )
    caution = (
        reliability["sentiment_evaluable_mentions"].ge(10)
        & reliability["recall"].ge(0.60)
        & reliability["spearman_rho"].ge(0.30)
    )
    reliability["measurement_status"] = np.select(
        [adequate, caution],
        ["adequate_for_exploratory_comparison", "caution"],
        default="insufficient",
    )
    return reliability.sort_values(
        ["measurement_status", "gold_mentions"],
        ascending=[True, False],
    )


# Pełna ocena i zapis


def main() -> None:
    data, duplicate_count = load_evaluated_data()

    base_summary = validation.base_evaluation.build_summary(
        data, duplicate_count
    )
    summary = pd.concat(
        [base_summary, validation.build_multilabel_summary(data)],
        ignore_index=True,
    )
    save_csv(summary, evaluation_summary_csv)
    aspect_performance = validation.build_aspect_performance(data)
    save_csv(aspect_performance, aspect_performance_csv)
    save_csv(
        validation.build_confidence_intervals(data),
        confidence_intervals_csv,
    )

    sentiment_data = data.loc[data["aspect_match"]].copy()
    sentiment_confusion = pd.crosstab(
        sentiment_data["manual_sentiment"],
        sentiment_data["model_sentiment_for_manual_aspect"],
        dropna=False,
    ).reindex(
        index=validation.sentiment_labels,
        columns=validation.sentiment_labels,
        fill_value=0,
    )
    save_csv(sentiment_confusion, sentiment_confusion_csv, include_index=True)

    class_performance = pd.DataFrame(
        classification_report(
            sentiment_data["manual_sentiment"],
            sentiment_data["model_sentiment_for_manual_aspect"],
            labels=validation.sentiment_labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    save_csv(
        class_performance,
        sentiment_class_performance_csv,
        include_index=True,
    )

    save_csv(model_quality_by_category(data), category_quality_csv)
    save_csv(
        model_quality_by_category_aspect(data),
        category_aspect_quality_csv,
    )
    save_csv(
        model_quality_by_category_sentiment(data),
        category_sentiment_quality_csv,
    )
    continuous_category = continuous_score_quality(data, ["category1"])
    continuous_aspect = continuous_score_quality(data, ["manual_aspect"]).rename(
        columns={"manual_aspect": "aspect"}
    )
    save_csv(continuous_category, continuous_category_csv)
    save_csv(continuous_aspect, continuous_aspect_csv)
    save_csv(
        build_aspect_measurement_reliability(
            data,
            aspect_performance,
            continuous_aspect,
        ),
        aspect_reliability_csv,
    )

    data["error_type"] = data.apply(
        validation.build_error_description,
        axis=1,
    )
    output = validation.stringify_tuple_columns(data)
    output_columns = [
        "annotation_id",
        "sample_type",
        "sampling_stratum",
        "category1",
        "category2",
        "review_id",
        "id",
        "aspect_text",
        "manual_aspect",
        "gold_aspects",
        "model_aspects",
        "matched_gold_aspects",
        "missing_gold_aspects",
        "extra_model_aspects",
        "manual_sentiment",
        "model_sentiment_for_manual_aspect",
        "manual_sentiment_score",
        "model_sentiment_score_for_manual_aspect",
        "has_negation",
        "aspect_match",
        "sentiment_match_same_aspect",
        "error_type",
        "manual_notes",
    ]
    save_csv(output[output_columns], evaluation_rows_csv)
    save_csv(
        output.loc[output["has_validation_error"], output_columns],
        error_rows_csv,
    )

    methodology = pd.DataFrame(
        [
            (
                "sampling_design",
                "stratified by category1 and predicted sentiment",
            ),
            (
                "class_distribution",
                "negative and neutral predictions intentionally oversampled",
            ),
            (
                "interpretation",
                "quality validation, not population sentiment prevalence",
            ),
            (
                "independence",
                "fragments excluded from all earlier annotation samples",
            ),
            (
                "sentiment_evaluation",
                "conditional on correct detection of the manual primary aspect",
            ),
            (
                "manual_sentiment_score",
                "negative=-1, neutral=0, positive=1",
            ),
            (
                "continuous_score_validation",
                "Spearman rho and prediction error; not population calibration",
            ),
            (
                "multilabel_aspects",
                "additional exact aspect codes parsed from manual_notes",
            ),
            (
                "measurement_status",
                "post-validation quality gate for exploratory comparisons",
            ),
        ],
        columns=["parameter", "value"],
    )
    save_csv(methodology, methodology_csv)

    summary_lookup = summary.set_index("metric")["value"]
    print(f"Evaluated {len(data)} independent annotations")
    print(
        "Aspect detection F1: "
        f"{summary_lookup['aspect_detection_f1']:.2f}"
    )
    print(
        "Primary aspect recall: "
        f"{summary_lookup['primary_aspect_recall']:.2f}"
    )
    print(
        "Sentiment accuracy: "
        f"{summary_lookup['sentiment_accuracy_same_aspect']:.2f}"
    )
    print(
        "Sentiment macro F1: "
        f"{summary_lookup['sentiment_macro_f1_same_aspect']:.2f}"
    )
    print(f"Saved stratified validation results to {validation_dir}")


if __name__ == "__main__":
    main()
