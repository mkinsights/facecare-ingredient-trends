from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

import evaluate_sentiment_validation_sample as base_evaluation


# Konfiguracja koncowej walidacji po dostrojeniu

root = Path(__file__).resolve().parents[1]
validation_dir = (
    root / "processed" / "sentiment" / "validation" / "tuned"
)
sample_csv = validation_dir / "manual_post_tuning_validation_sample_150.csv"
key_csv = validation_dir / "manual_post_tuning_validation_key_150.csv"

evaluation_rows_csv = validation_dir / "post_tuning_validation_evaluation_rows.csv"
evaluation_summary_csv = (
    validation_dir / "post_tuning_validation_evaluation_summary.csv"
)
aspect_performance_csv = (
    validation_dir / "post_tuning_validation_aspect_performance.csv"
)
sentiment_confusion_csv = (
    validation_dir / "post_tuning_validation_sentiment_confusion.csv"
)
sentiment_class_performance_csv = (
    validation_dir / "post_tuning_validation_sentiment_class_performance.csv"
)
confidence_intervals_csv = (
    validation_dir / "post_tuning_validation_confidence_intervals.csv"
)
error_rows_csv = validation_dir / "post_tuning_validation_error_rows.csv"

no_aspect_label = "no_aspect"
sentiment_labels = ["negative", "neutral", "positive"]
aspect_labels = {
    "effectiveness",
    "hydration_nourishment",
    "skin_tolerance",
    "pore_clogging",
    "cleansing_makeup_removal",
    "texture_consistency",
    "absorption_finish",
    "scent",
    "application",
    "packaging",
    "price_value",
    "efficiency",
    "formula_composition",
}


# Przygotowanie glownego i dodatkowych aspektow recznych


def parse_gold_aspects(row: pd.Series) -> tuple[str, ...]:
    """Laczy aspekt glowny z poprawnymi etykietami zapisanymi w notatce."""
    if row["manual_aspect"] == no_aspect_label:
        return tuple()

    aspects = {row["manual_aspect"]}
    if pd.notna(row["manual_notes"]):
        for value in str(row["manual_notes"]).split(";"):
            candidate = value.strip()
            if candidate in aspect_labels:
                aspects.add(candidate)
    return tuple(sorted(aspects))


def configure_base_evaluation() -> None:
    """Kieruje wspolne funkcje ewaluacyjne do nowej, niezaleznej proby."""
    base_evaluation.sample_csv = sample_csv
    base_evaluation.key_csv = key_csv


def add_multilabel_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Dodaje roznice zbiorow pomiedzy aspektami recznymi i modelowymi."""
    data = data.copy()
    data["gold_aspects"] = data.apply(parse_gold_aspects, axis=1)
    data["matched_gold_aspects"] = data.apply(
        lambda row: tuple(sorted(set(row["gold_aspects"]) & set(row["model_aspects"]))),
        axis=1,
    )
    data["missing_gold_aspects"] = data.apply(
        lambda row: tuple(sorted(set(row["gold_aspects"]) - set(row["model_aspects"]))),
        axis=1,
    )
    data["extra_model_aspects"] = data.apply(
        lambda row: tuple(sorted(set(row["model_aspects"]) - set(row["gold_aspects"]))),
        axis=1,
    )
    data["aspect_set_match"] = data.apply(
        lambda row: set(row["gold_aspects"]) == set(row["model_aspects"]), axis=1
    )
    data["primary_sentiment_error"] = (
        data["aspect_match"] & data["sentiment_match_same_aspect"].eq(False)
    )
    data["has_validation_error"] = (
        data["missing_gold_aspects"].str.len().gt(0)
        | ((data["gold_aspects"].str.len() == 0) & data["model_has_aspect"])
        | data["primary_sentiment_error"]
    )
    return data


# Metryki wieloetykietowe i diagnostyka


def build_multilabel_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Liczy mikro-metryki dla wszystkich recznie wskazanych aspektow."""
    true_positive = int(data["matched_gold_aspects"].str.len().sum())
    false_negative = int(data["missing_gold_aspects"].str.len().sum())
    false_positive = int(data["extra_model_aspects"].str.len().sum())

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    primary_rows = data.loc[data["manual_has_aspect"]]

    rows = [
        ("gold_aspect_mentions", int(data["gold_aspects"].str.len().sum())),
        ("model_aspect_mentions", int(data["model_aspects"].str.len().sum())),
        ("multilabel_true_positive", true_positive),
        ("multilabel_false_positive", false_positive),
        ("multilabel_false_negative", false_negative),
        ("multilabel_micro_precision", precision),
        ("multilabel_micro_recall", recall),
        ("multilabel_micro_f1", f1),
        ("multilabel_exact_set_match_rate", data["aspect_set_match"].mean()),
        ("primary_aspect_recall", primary_rows["aspect_match"].mean()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).round(2)


def build_aspect_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Pokazuje recall osobno dla kazdego aspektu z pelnej adnotacji."""
    rows: list[dict[str, object]] = []
    for aspect in sorted(aspect_labels):
        gold_mask = data["gold_aspects"].apply(lambda values: aspect in values)
        model_mask = data["model_aspects"].apply(lambda values: aspect in values)
        gold_mentions = int(gold_mask.sum())
        detected_mentions = int((gold_mask & model_mask).sum())
        rows.append(
            {
                "aspect": aspect,
                "gold_mentions": gold_mentions,
                "detected_mentions": detected_mentions,
                "missed_mentions": gold_mentions - detected_mentions,
                "recall": detected_mentions / gold_mentions if gold_mentions else pd.NA,
            }
        )
    return pd.DataFrame(rows).sort_values(["recall", "gold_mentions"]).round(2)


def build_error_description(row: pd.Series) -> str:
    """Tworzy czytelna liste problemow obecnych w jednym fragmencie."""
    errors: list[str] = []
    if row["missing_gold_aspects"]:
        errors.append("missing_aspect")
    if not row["gold_aspects"] and row["model_aspects"]:
        errors.append("false_positive_aspect")
    if row["primary_sentiment_error"]:
        errors.append("wrong_sentiment")
    return "; ".join(errors)


def wilson_interval(successes: int, observations: int) -> tuple[float, float]:
    """Liczy 95% przedzial Wilsona dla proporcji w probie walidacyjnej."""
    if observations == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    proportion = successes / observations
    denominator = 1 + z**2 / observations
    centre = proportion + z**2 / (2 * observations)
    margin = z * np.sqrt(
        proportion * (1 - proportion) / observations
        + z**2 / (4 * observations**2)
    )
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def build_confidence_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Szacuje niepewnosc kluczowych proporcji w ocenionej probie."""
    manual_aspect_rows = data.loc[data["manual_has_aspect"]]
    model_detected_rows = data.loc[data["model_has_aspect"]]
    sentiment_rows = data.loc[data["aspect_match"]]

    metrics = [
        (
            "aspect_detection_precision",
            int((data["manual_has_aspect"] & data["model_has_aspect"]).sum()),
            len(model_detected_rows),
        ),
        (
            "aspect_detection_recall",
            int((data["manual_has_aspect"] & data["model_has_aspect"]).sum()),
            len(manual_aspect_rows),
        ),
        (
            "primary_aspect_recall",
            int(manual_aspect_rows["aspect_match"].sum()),
            len(manual_aspect_rows),
        ),
        (
            "sentiment_accuracy_same_aspect",
            int(sentiment_rows["sentiment_match_same_aspect"].eq(True).sum()),
            len(sentiment_rows),
        ),
    ]
    rows: list[dict[str, object]] = []
    for metric, successes, observations in metrics:
        lower, upper = wilson_interval(successes, observations)
        rows.append(
            {
                "metric": metric,
                "estimate": successes / observations,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "observations": observations,
            }
        )
    return pd.DataFrame(rows).round(2)


def stringify_tuple_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Zapisuje zbiory aspektow w czytelnej postaci rozdzielonej srednikiem."""
    data = data.copy()
    tuple_columns = [
        "gold_aspects",
        "model_aspects",
        "matched_gold_aspects",
        "missing_gold_aspects",
        "extra_model_aspects",
    ]
    for column in tuple_columns:
        data[column] = data[column].apply("; ".join)
    return data


# Zapis wynikow


def main() -> None:
    configure_base_evaluation()
    data, duplicate_count = base_evaluation.load_annotations()
    data = base_evaluation.add_evaluation_columns(data)
    data = add_multilabel_columns(data)

    base_summary = base_evaluation.build_summary(data, duplicate_count)
    summary = pd.concat(
        [base_summary, build_multilabel_summary(data)], ignore_index=True
    )
    summary.to_csv(
        evaluation_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )

    build_aspect_performance(data).to_csv(
        aspect_performance_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )

    sentiment_data = data.loc[data["aspect_match"]]
    sentiment_confusion = pd.crosstab(
        sentiment_data["manual_sentiment"],
        sentiment_data["model_sentiment_for_manual_aspect"],
        dropna=False,
    ).reindex(
        index=sentiment_labels,
        columns=sentiment_labels,
        fill_value=0,
    )
    sentiment_confusion.to_csv(sentiment_confusion_csv, encoding="utf-8-sig")
    class_performance = pd.DataFrame(
        classification_report(
            sentiment_data["manual_sentiment"],
            sentiment_data["model_sentiment_for_manual_aspect"],
            labels=sentiment_labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    class_performance.to_csv(
        sentiment_class_performance_csv,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    build_confidence_intervals(data).to_csv(
        confidence_intervals_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )

    data["error_type"] = data.apply(build_error_description, axis=1)
    output = stringify_tuple_columns(data)
    output_columns = [
        "annotation_id",
        "sample_type",
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
        "has_negation",
        "aspect_match",
        "sentiment_match_same_aspect",
        "error_type",
        "manual_notes",
    ]
    output[output_columns].to_csv(
        evaluation_rows_csv,
        index=False,
        encoding="utf-8-sig",
    )
    output.loc[output["has_validation_error"], output_columns].to_csv(
        error_rows_csv,
        index=False,
        encoding="utf-8-sig",
    )

    sentiment_accuracy = accuracy_score(
        sentiment_data["manual_sentiment"],
        sentiment_data["model_sentiment_for_manual_aspect"],
    )
    sentiment_macro_f1 = f1_score(
        sentiment_data["manual_sentiment"],
        sentiment_data["model_sentiment_for_manual_aspect"],
        labels=sentiment_labels,
        average="macro",
        zero_division=0,
    )
    print(f"Evaluated {len(data)} independent annotations")
    print(f"Primary-aspect sentiment accuracy: {sentiment_accuracy:.2f}")
    print(f"Primary-aspect sentiment macro F1: {sentiment_macro_f1:.2f}")
    print(f"Saved results to {evaluation_summary_csv}")


if __name__ == "__main__":
    main()
