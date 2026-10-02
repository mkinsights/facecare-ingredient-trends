from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


# Konfiguracja benchmarku

root = Path(__file__).resolve().parents[1]
annotation_dir = root / "processed" / "sentiment" / "validation"
output_dir = root / "processed" / "sentiment" / "benchmark"

development_sample_paths = [
    annotation_dir / "manual_annotation_sample_200.csv",
    annotation_dir / "initial" / "manual_validation_sample_150.csv",
]
test_sample_csv = (
    annotation_dir
    / "tuned"
    / "manual_post_tuning_validation_sample_150.csv"
)
test_key_csv = (
    annotation_dir
    / "tuned"
    / "manual_post_tuning_validation_key_150.csv"
)
hybrid_evaluation_csv = (
    annotation_dir
    / "tuned"
    / "post_tuning_validation_evaluation_rows.csv"
)

model_path = output_dir / "aspect_conditioned_sentiment_model.joblib"
audit_csv = output_dir / "supervised_sentiment_data_audit.csv"
cv_results_csv = output_dir / "supervised_sentiment_cv_results.csv"
summary_csv = output_dir / "supervised_sentiment_benchmark_summary.csv"
class_metrics_csv = output_dir / "supervised_sentiment_class_metrics.csv"
confusion_csv = output_dir / "supervised_sentiment_confusion_matrix.csv"
predictions_csv = output_dir / "supervised_sentiment_test_predictions.csv"
metadata_json = output_dir / "supervised_sentiment_metadata.json"

random_state = 42
inner_folds = 4
candidates_c = [0.25, 0.5, 1.0, 2.0]
sentiment_labels = ["negative", "neutral", "positive"]


# Przygotowanie danych bez przecieku do testu


def normalize_text(value: object) -> str:
    """Normalizuje tekst wylacznie do kontroli duplikatow miedzy zbiorami."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\W+", " ", text).strip()


def model_text(data: pd.DataFrame) -> pd.Series:
    """Dodaje jawny kontekst aspektu do tekstu klasyfikowanego fragmentu."""
    aspect_token = "aspect_" + data["manual_aspect"].astype(str)
    return aspect_token + " " + data["aspect_text"].fillna("").astype(str)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Laduje rozwojowe adnotacje i zachowuje koncowa probe jako test."""
    development = pd.concat(
        [pd.read_csv(path).drop_duplicates() for path in development_sample_paths],
        ignore_index=True,
    )
    test = pd.read_csv(test_sample_csv).drop_duplicates()
    test_key = pd.read_csv(test_key_csv)

    required_columns = {
        "annotation_id",
        "id",
        "aspect_text",
        "manual_aspect",
        "manual_sentiment",
    }
    for name, data in [("development", development), ("test", test)]:
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
        if data[list(required_columns)].isna().any().any():
            raise ValueError(f"{name} data contains incomplete manual annotations")

    development["normalized_text"] = development["aspect_text"].map(normalize_text)
    test["normalized_text"] = test["aspect_text"].map(normalize_text)

    test_products = set(test["id"])
    test_texts = set(test["normalized_text"])
    development["overlaps_test_product"] = development["id"].isin(test_products)
    development["overlaps_test_text"] = development["normalized_text"].isin(test_texts)

    clean_development = development.loc[
        ~development["overlaps_test_product"]
        & ~development["overlaps_test_text"]
        & development["manual_aspect"].ne("no_aspect")
    ].copy()
    clean_test = test.loc[test["manual_aspect"].ne("no_aspect")].copy()
    clean_test = clean_test.merge(
        test_key[["annotation_id", "review_rating"]],
        on="annotation_id",
        how="left",
        validate="one_to_one",
    )

    if set(clean_development["id"]) & set(clean_test["id"]):
        raise ValueError("Product leakage remains between development and test data")
    if set(clean_development["normalized_text"]) & set(clean_test["normalized_text"]):
        raise ValueError("Normalized text leakage remains between development and test")
    return development, clean_development, clean_test


# Model i dobor parametru wylacznie na danych rozwojowych


def build_model(c_value: float) -> Pipeline:
    """Buduje lekki model odpowiedni dla malego korpusu tekstowego."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=25_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2_000,
        solver="lbfgs",
        random_state=random_state,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def purge_fold_text_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Usuwa z treningu identyczne teksty obecne w foldzie walidacyjnym."""
    validation_texts = set(validation["normalized_text"])
    return train.loc[~train["normalized_text"].isin(validation_texts)].copy()


def tune_model(development: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Dobiera C przez grupowa walidacje krzyzowa po produktach."""
    splitter = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )
    rows: list[dict[str, float]] = []
    best_c = candidates_c[0]
    best_macro_f1 = -1.0

    for c_value in candidates_c:
        observed = np.empty(len(development), dtype=object)
        for train_index, validation_index in splitter.split(
            development,
            development["manual_sentiment"],
            groups=development["id"],
        ):
            train_fold = purge_fold_text_overlap(
                development.iloc[train_index],
                development.iloc[validation_index],
            )
            validation_fold = development.iloc[validation_index]
            model = build_model(c_value)
            model.fit(
                model_text(train_fold),
                train_fold["manual_sentiment"],
            )
            observed[validation_index] = model.predict(model_text(validation_fold))

        macro_f1 = f1_score(
            development["manual_sentiment"],
            observed,
            labels=sentiment_labels,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "c_value": c_value,
                "accuracy": accuracy_score(development["manual_sentiment"], observed),
                "balanced_accuracy": balanced_accuracy_score(
                    development["manual_sentiment"], observed
                ),
                "macro_f1": macro_f1,
            }
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_c = c_value

    return best_c, pd.DataFrame(rows).round(2)


# Porownanie z baseline i obecnym modelem hybrydowym


def rating_to_sentiment(rating: pd.Series) -> pd.Series:
    """Mapuje gwiazdki opinii na trzy klasy uzywane w benchmarku."""
    numeric = pd.to_numeric(rating, errors="raise")
    return pd.Series(
        np.select(
            [numeric <= 2, numeric == 3],
            ["negative", "neutral"],
            default="positive",
        ),
        index=rating.index,
    )


def metric_row(
    model_name: str,
    actual: pd.Series,
    predicted: pd.Series,
    total_test_rows: int,
) -> dict[str, object]:
    """Liczy porownywalne metryki dla jednego modelu i zakresu danych."""
    return {
        "model": model_name,
        "evaluated_rows": len(actual),
        "coverage": len(actual) / total_test_rows,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=sentiment_labels,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate(
    model: Pipeline,
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ocenia model na zewnetrznej probie oraz na wspolnym zakresie hybrydy."""
    test = test.copy()
    probabilities = model.predict_proba(model_text(test))
    test["supervised_sentiment"] = model.predict(model_text(test))
    for index, label in enumerate(model.named_steps["classifier"].classes_):
        test[f"probability_{label}"] = probabilities[:, index]
    test["supervised_correct"] = test["supervised_sentiment"].eq(
        test["manual_sentiment"]
    )
    test["rating_baseline_sentiment"] = rating_to_sentiment(test["review_rating"])

    majority_label = development["manual_sentiment"].mode().iloc[0]
    test["majority_baseline_sentiment"] = majority_label

    hybrid = pd.read_csv(hybrid_evaluation_csv)[
        [
            "annotation_id",
            "aspect_match",
            "model_sentiment_for_manual_aspect",
        ]
    ]
    test = test.merge(hybrid, on="annotation_id", how="left", validate="one_to_one")
    covered = test.loc[
        test["aspect_match"].eq(True)
        & test["model_sentiment_for_manual_aspect"].notna()
    ].copy()

    summary_rows = [
        metric_row(
            "supervised_manual_labels_all_test",
            test["manual_sentiment"],
            test["supervised_sentiment"],
            len(test),
        ),
        metric_row(
            "review_rating_baseline_all_test",
            test["manual_sentiment"],
            test["rating_baseline_sentiment"],
            len(test),
        ),
        metric_row(
            "majority_baseline_all_test",
            test["manual_sentiment"],
            test["majority_baseline_sentiment"],
            len(test),
        ),
        metric_row(
            "supervised_on_hybrid_covered_subset",
            covered["manual_sentiment"],
            covered["supervised_sentiment"],
            len(test),
        ),
        metric_row(
            "current_hybrid_covered_subset",
            covered["manual_sentiment"],
            covered["model_sentiment_for_manual_aspect"],
            len(test),
        ),
    ]
    return test, pd.DataFrame(summary_rows).round(2)


# Zapis audytowalnych wynikow


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_development, development, test = load_data()
    best_c, cv_results = tune_model(development)

    model = build_model(best_c)
    model.fit(model_text(development), development["manual_sentiment"])
    predictions, summary = evaluate(model, development, test)

    audit = pd.DataFrame(
        [
            ("development_rows_before_purge", len(raw_development)),
            (
                "development_rows_overlapping_test_product",
                int(raw_development["overlaps_test_product"].sum()),
            ),
            (
                "development_rows_overlapping_test_text",
                int(raw_development["overlaps_test_text"].sum()),
            ),
            ("development_rows_after_purge_and_no_aspect_filter", len(development)),
            ("development_products_after_purge", development["id"].nunique()),
            ("test_rows_with_manual_aspect", len(test)),
            ("test_products", test["id"].nunique()),
        ],
        columns=["metric", "value"],
    )
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    cv_results.to_csv(
        cv_results_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )

    report = pd.DataFrame(
        classification_report(
            predictions["manual_sentiment"],
            predictions["supervised_sentiment"],
            labels=sentiment_labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(
        class_metrics_csv,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    pd.DataFrame(
        confusion_matrix(
            predictions["manual_sentiment"],
            predictions["supervised_sentiment"],
            labels=sentiment_labels,
        ),
        index=sentiment_labels,
        columns=sentiment_labels,
    ).to_csv(confusion_csv, encoding="utf-8-sig")

    prediction_columns = [
        "annotation_id",
        "id",
        "aspect_text",
        "manual_aspect",
        "manual_sentiment",
        "supervised_sentiment",
        "supervised_correct",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
        "rating_baseline_sentiment",
        "aspect_match",
        "model_sentiment_for_manual_aspect",
    ]
    predictions[prediction_columns].to_csv(
        predictions_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    joblib.dump(model, model_path)

    metadata = {
        "model": "aspect_conditioned_tfidf_logistic_regression",
        "best_c": best_c,
        "inner_cv": "stratified_group_by_product",
        "inner_folds": inner_folds,
        "test_policy": "product_and_normalized_text_disjoint_from_development",
        "development_rows": len(development),
        "test_rows": len(test),
        "sentiment_labels": sentiment_labels,
    }
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Selected C={best_c}")
    print(summary.to_string(index=False))
    print(f"Saved benchmark to {summary_csv}")


if __name__ == "__main__":
    main()
