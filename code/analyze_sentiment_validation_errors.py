from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from evaluate_sentiment_validation_sample import add_evaluation_columns, load_annotations


# Konfiguracja

root = Path(__file__).resolve().parents[1]
validation_dir = root / "processed" / "sentiment" / "validation" / "initial"

error_rows_csv = validation_dir / "manual_validation_error_rows.csv"
error_summary_csv = validation_dir / "manual_validation_error_summary.csv"
aspect_performance_csv = validation_dir / "manual_validation_aspect_performance.csv"
sentiment_performance_csv = validation_dir / "manual_validation_sentiment_performance.csv"
annotation_review_csv = validation_dir / "manual_validation_annotation_review_needed.csv"

sentiment_labels = ["negative", "neutral", "positive"]


# Klasyfikacja bledow


def add_error_types(data: pd.DataFrame) -> pd.DataFrame:
    """Nadaje kazdemu rekordowi jeden rozlaczny, glowny typ wyniku."""
    data = data.copy()
    conditions = [
        data["manual_has_aspect"] & ~data["model_has_aspect"],
        ~data["manual_has_aspect"] & data["model_has_aspect"],
        data["manual_has_aspect"] & data["model_has_aspect"] & ~data["aspect_match"],
        data["aspect_match"] & data["sentiment_match_same_aspect"].eq(False),
        ~data["manual_has_aspect"] & ~data["model_has_aspect"],
    ]
    choices = [
        "false_negative_aspect",
        "false_positive_aspect",
        "manual_aspect_not_detected",
        "wrong_sentiment",
        "correct_no_aspect",
    ]
    data["error_type"] = np.select(conditions, choices, default="correct")
    data["is_error"] = data["error_type"].ne("correct") & data["error_type"].ne(
        "correct_no_aspect"
    )
    data["model_aspects"] = data["model_aspects"].apply("; ".join)
    return data


# Podsumowania diagnostyczne


def build_aspect_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Pokazuje, ktore recznie oznaczone aspekty sa najczesciej przeoczone."""
    aspects = data.loc[data["manual_has_aspect"]].copy()
    summary = aspects.groupby("manual_aspect", as_index=False).agg(
        manual_rows=("annotation_id", "size"),
        detected_any_aspect=("model_has_aspect", "sum"),
        detected_manual_aspect=("aspect_match", "sum"),
    )
    summary["manual_aspect_recall"] = (
        summary["detected_manual_aspect"] / summary["manual_rows"]
    )
    return summary.sort_values(["manual_aspect_recall", "manual_rows"])


def build_sentiment_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Ocena polaryzacji tylko tam, gdzie model wykryl ten sam aspekt co osoba."""
    matching = data.loc[data["aspect_match"]].copy()
    rows: list[dict[str, object]] = []
    for group_name, group in [("all", matching), *matching.groupby("manual_aspect")]:
        rows.append(
            {
                "group": group_name,
                "rows": len(group),
                "accuracy": accuracy_score(
                    group["manual_sentiment"], group["model_sentiment_for_manual_aspect"]
                ),
                "macro_f1": f1_score(
                    group["manual_sentiment"],
                    group["model_sentiment_for_manual_aspect"],
                    labels=sentiment_labels,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["group"])


def main() -> None:
    data, _ = load_annotations()
    data = add_error_types(add_evaluation_columns(data))

    error_columns = [
        "annotation_id", "error_type", "sample_type", "sampling_stratum", "review_id",
        "id", "aspect_text", "manual_aspect", "model_aspects", "candidate_aspect",
        "manual_sentiment", "model_sentiment_for_manual_aspect", "model_sentiment_label",
        "has_negation", "sentiment_method", "manual_notes",
    ]
    data.loc[data["is_error"], error_columns].to_csv(
        error_rows_csv, index=False, encoding="utf-8-sig"
    )

    # Notatka zawierajaca nazwe innego aspektu moze oznaczac, ze osoba
    # oznaczajaca skorygowala decyzje w komentarzu, ale nie w glownej kolumnie.
    # Tych rekordow nie poprawiamy automatycznie: trafiaja do krotkiej rewizji.
    aspect_mismatches = data.loc[
        data["error_type"].eq("manual_aspect_not_detected")
        & data["manual_notes"].notna()
    ].copy()
    aspect_mismatches["note_matches_model_aspect"] = aspect_mismatches.apply(
        lambda row: row["manual_notes"] in row["model_aspects"].split("; "), axis=1
    )
    review_columns = [
        "annotation_id", "manual_aspect", "manual_notes", "model_aspects",
        "note_matches_model_aspect", "aspect_text",
    ]
    aspect_mismatches[review_columns].to_csv(
        annotation_review_csv, index=False, encoding="utf-8-sig"
    )

    error_summary = (
        data.groupby(["error_type", "sampling_stratum", "has_negation"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["error_type", "rows"], ascending=[True, False])
    )
    error_summary.to_csv(error_summary_csv, index=False, encoding="utf-8-sig")
    build_aspect_performance(data).round(2).to_csv(
        aspect_performance_csv, index=False, encoding="utf-8-sig"
    )
    build_sentiment_performance(data).round(2).to_csv(
        sentiment_performance_csv, index=False, encoding="utf-8-sig"
    )

    print(f"Saved {int(data['is_error'].sum())} errors to {error_rows_csv}")
    print(f"Saved {len(aspect_mismatches)} annotation review rows to {annotation_review_csv}")


if __name__ == "__main__":
    main()
