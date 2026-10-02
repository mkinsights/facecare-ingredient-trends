from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# Pliki wejsciowe i wyjsciowe

root = Path(__file__).resolve().parents[1]
validation_dir = root / "processed" / "sentiment" / "validation" / "initial"

sample_csv = validation_dir / "manual_validation_sample_150.csv"
key_csv = validation_dir / "manual_validation_key_150.csv"
mentions_csv = root / "processed" / "sentiment" / "model" / "review_aspect_sentiment.csv"
rows_csv = validation_dir / "manual_validation_evaluation_rows.csv"
summary_csv = validation_dir / "manual_validation_evaluation_summary.csv"
aspect_confusion_csv = validation_dir / "manual_validation_aspect_confusion.csv"
sentiment_confusion_csv = validation_dir / "manual_validation_sentiment_confusion.csv"

no_aspect_label = "no_aspect"
sentiment_labels = ["negative", "neutral", "positive"]


# Przygotowanie oznaczen


def load_annotations() -> tuple[pd.DataFrame, int]:
    """Laczy slepe oznaczenia z kluczem i odrzuca tylko pelne duplikaty."""
    sample = pd.read_csv(sample_csv)
    key = pd.read_csv(key_csv)
    mentions = pd.read_csv(mentions_csv)

    duplicate_count = int(sample.duplicated().sum())
    sample = sample.drop_duplicates().copy()
    if sample["annotation_id"].duplicated().any():
        raise ValueError("Annotation IDs must be unique after removing exact duplicates")
    if sample[["manual_aspect", "manual_sentiment", "has_negation"]].isna().any().any():
        raise ValueError("Manual annotations are incomplete")

    data = sample.merge(
        key,
        on=["annotation_id", "sample_type", "review_id", "id", "aspect_text"],
        how="left",
        validate="one_to_one",
    )
    if data["candidate_aspect"].isna().sum() != (data["sample_type"] == "no_aspect_control").sum():
        raise ValueError("Some candidate rows do not have a matching model prediction")

    # Fragment moze zawierac wiecej niz jeden aspekt. Klucz przechowuje aspekt,
    # wedlug ktorego fragment wylosowano, dlatego do ewaluacji pobieramy pelny
    # zestaw predykcji OOF dla tego samego fragmentu.
    mentions["_fragment_key"] = (
        mentions["review_id"].astype(str)
        + "_"
        + mentions["fragment_index"].astype(str)
    )
    data["_fragment_key"] = (
        data["review_id"].astype(str)
        + "_"
        + data["fragment_index"].astype(str)
    )
    aspects_by_fragment = mentions.groupby("_fragment_key")["aspect"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    sentiment_by_fragment_aspect = (
        mentions.drop_duplicates(["_fragment_key", "aspect"])
        .set_index(["_fragment_key", "aspect"])["sentiment_label"]
    )
    data["model_aspects"] = data["_fragment_key"].map(aspects_by_fragment).apply(
        lambda value: value if isinstance(value, tuple) else tuple()
    )
    data["model_sentiment_for_manual_aspect"] = [
        sentiment_by_fragment_aspect.get((fragment, aspect), pd.NA)
        for fragment, aspect in zip(data["_fragment_key"], data["manual_aspect"])
    ]
    return data, duplicate_count


def add_evaluation_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Tworzy wskazniki wykrywania aspektu, zgodnosci aspektu i polaryzacji."""
    data = data.copy()
    data["manual_has_aspect"] = data["manual_aspect"].ne(no_aspect_label)
    data["model_has_aspect"] = data["model_aspects"].str.len().gt(0)
    data["aspect_match"] = (
        data["manual_has_aspect"]
        & data["model_has_aspect"]
        & data.apply(lambda row: row["manual_aspect"] in row["model_aspects"], axis=1)
    )
    data["sentiment_match_same_aspect"] = pd.NA
    data.loc[data["aspect_match"], "sentiment_match_same_aspect"] = data.loc[
        data["aspect_match"], "manual_sentiment"
    ].eq(data.loc[data["aspect_match"], "model_sentiment_for_manual_aspect"])
    return data


# Metryki i pliki wynikowe


def build_summary(data: pd.DataFrame, duplicate_count: int) -> pd.DataFrame:
    """Liczy metryki osobno dla detekcji, klasyfikacji aspektu i sentymentu."""
    true_aspect = data["manual_has_aspect"]
    model_aspect = data["model_has_aspect"]
    true_positive = int((true_aspect & model_aspect).sum())
    false_positive = int((~true_aspect & model_aspect).sum())
    false_negative = int((true_aspect & ~model_aspect).sum())
    true_negative = int((~true_aspect & ~model_aspect).sum())

    detection_precision = true_positive / (true_positive + false_positive)
    detection_recall = true_positive / (true_positive + false_negative)
    detection_f1 = 2 * detection_precision * detection_recall / (
        detection_precision + detection_recall
    )
    strict = data.loc[data["aspect_match"]].copy()

    rows = [
        ("records_in_source_file", len(data) + duplicate_count),
        ("exact_duplicate_records_ignored", duplicate_count),
        ("records_evaluated", len(data)),
        ("manual_aspect_present", int(true_aspect.sum())),
        ("manual_no_aspect", int((~true_aspect).sum())),
        ("model_aspect_detected", int(model_aspect.sum())),
        ("aspect_detection_true_positive", true_positive),
        ("aspect_detection_false_positive", false_positive),
        ("aspect_detection_false_negative", false_negative),
        ("aspect_detection_true_negative", true_negative),
        ("aspect_detection_precision", detection_precision),
        ("aspect_detection_recall", detection_recall),
        ("aspect_detection_f1", detection_f1),
        ("aspect_exact_match_count", int(data["aspect_match"].sum())),
        ("aspect_exact_match_rate_among_model_detections", data.loc[model_aspect, "aspect_match"].mean()),
        ("sentiment_records_same_aspect", len(strict)),
        ("sentiment_accuracy_same_aspect", accuracy_score(strict["manual_sentiment"], strict["model_sentiment_for_manual_aspect"])),
        ("sentiment_macro_f1_same_aspect", f1_score(strict["manual_sentiment"], strict["model_sentiment_for_manual_aspect"], labels=sentiment_labels, average="macro", zero_division=0)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).round({"value": 2})


def main() -> None:
    data, duplicate_count = load_annotations()
    data = add_evaluation_columns(data)

    output_columns = [
        "annotation_id", "sample_type", "review_id", "id", "aspect_text",
        "manual_aspect", "candidate_aspect", "model_aspects", "manual_sentiment",
        "model_sentiment_label", "model_sentiment_for_manual_aspect",
        "has_negation", "sentiment_method", "sampling_stratum", "manual_has_aspect",
        "model_has_aspect", "aspect_match", "sentiment_match_same_aspect", "manual_notes",
    ]
    data[output_columns].to_csv(rows_csv, index=False, encoding="utf-8-sig")
    build_summary(data, duplicate_count).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    aspect_confusion = pd.crosstab(
        data["manual_aspect"], data["candidate_aspect"].fillna(no_aspect_label), dropna=False
    )
    aspect_confusion.to_csv(aspect_confusion_csv, encoding="utf-8-sig")

    sentiment_data = data.loc[data["aspect_match"]]
    sentiment_confusion = pd.crosstab(
        sentiment_data["manual_sentiment"],
        sentiment_data["model_sentiment_for_manual_aspect"],
        dropna=False,
    ).reindex(index=sentiment_labels, columns=sentiment_labels, fill_value=0)
    sentiment_confusion.to_csv(sentiment_confusion_csv, encoding="utf-8-sig")

    print(f"Evaluated {len(data)} unique annotations")
    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
