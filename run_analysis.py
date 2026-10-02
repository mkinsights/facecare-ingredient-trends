from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STEPS = [
    ("products_analysis.py", []),
    ("sentiment_analysis.py", []),
    ("sentiment_model.py", []),
    ("evaluate_sentiment_validation_sample.py", []),
    ("analyze_sentiment_validation_errors.py", []),
    ("evaluate_sentiment_post_tuning_validation.py", []),
    ("evaluate_sentiment_stratified_validation.py", []),
    ("sentiment_threshold_calibration.py", []),
    ("sentiment_supervised_benchmark.py", []),
    ("sentiment_benchmark_statistical_validation.py", []),
    ("product_sentiment_integration_analysis.py", []),
    ("sentiment_targeted_ingredient_aspect_analysis.py", []),
    ("sentiment_multilevel_ingredient_aspect_analysis.py", ["--refit"]),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def count_rows(path: Path) -> int:
    return len(read_rows(path))


def find_row(rows: list[dict[str, str]], **values: str) -> dict[str, str]:
    found = [
        row
        for row in rows
        if all(row.get(column) == value for column, value in values.items())
    ]
    if len(found) != 1:
        raise RuntimeError(f"Nie znaleziono jednego wiersza dla: {values}")
    return found[0]


def check(label: str, value: object, expected: object) -> None:
    if value != expected:
        raise RuntimeError(f"{label}: jest {value!r}, powinno być {expected!r}")
    print(f"[OK] {label}: {value}")


def check_results() -> None:
    counts = {
        "surowe produkty": ("raw/products_analysis.csv", 1999),
        "surowe opinie": ("raw/product_reviews.csv", 12839),
        "składniki z Kosmopedii": ("raw/kosmopedia_ingredients.csv", 1200),
        "produkty po czyszczeniu": ("processed/products_analysis_processed.csv", 1814),
        "relacje składnik–aspekt": (
            "processed/sentiment/aspects/targeted_ingredient_aspect_results.csv",
            16,
        ),
        "modele wielopoziomowe": (
            "processed/sentiment/multilevel/multilevel_model_diagnostics.csv",
            16,
        ),
    }
    for label, (path, expected) in counts.items():
        check(label, count_rows(ROOT / path), expected)

    metadata = json.loads(
        (ROOT / "processed/sentiment/model/sentiment_model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    expected_metadata = {
        "training_reviews": 11609,
        "training_products": 1666,
        "oof_folds": 5,
        "aspect_mentions": 13307,
        "reviews_with_aspect": 7033,
    }
    for name, expected in expected_metadata.items():
        check(name, metadata[name], expected)

    models = read_rows(
        ROOT / "processed/summary/weighted_rating_regression_model_summary.csv"
    )
    check(
        "skorygowane R2 modelu bazowego",
        float(find_row(models, model="base")["adjusted_r_squared"]),
        0.17,
    )
    check(
        "skorygowane R2 modelu formuły",
        float(find_row(models, model="formula")["adjusted_r_squared"]),
        0.18,
    )

    aspect_rows = read_rows(
        ROOT / "processed/sentiment/aspects/targeted_ingredient_aspect_results.csv"
    )
    ceramides = find_row(
        aspect_rows,
        feature="is_ceramides",
        aspect="hydration_nourishment",
    )
    check("ceramidy–nawilżenie: efekt", float(ceramides["primary_coefficient"]), 0.05)
    check("ceramidy–nawilżenie: q", float(ceramides["q_value_fdr_bh"]), 0.01)

    multilevel_rows = read_rows(
        ROOT / "processed/sentiment/multilevel/multilevel_pooled_effects.csv"
    )
    ceramides = find_row(
        multilevel_rows,
        feature="is_ceramides",
        aspect="hydration_nourishment",
    )
    check(
        "ceramidy–nawilżenie: efekt wielopoziomowy",
        float(ceramides["coefficient"]),
        0.06,
    )
    check(
        "ceramidy–nawilżenie: q wielopoziomowe",
        float(ceramides["q_value_fdr_bh"]),
        0.02,
    )

    validation_rows = read_rows(
        ROOT
        / "processed/sentiment/validation/final/stratified_validation_evaluation_summary.csv"
    )
    validation = {row["metric"]: float(row["value"]) for row in validation_rows}
    check("F1 detekcji aspektu", validation["aspect_detection_f1"], 0.96)
    check("macro F1 wydźwięku", validation["sentiment_macro_f1_same_aspect"], 0.55)


def run_all() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"Projekt wymaga Pythona 3.11, a uruchomiono {sys.version_info.major}.{sys.version_info.minor}."
        )

    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "42")
    environment.setdefault("LOKY_MAX_CPU_COUNT", "1")

    for number, (script, arguments) in enumerate(STEPS, start=1):
        print(f"\n[{number}/{len(STEPS)}] {script}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "code" / script), *arguments],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    check_results()
    print("\nAnaliza zakończona.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Uruchamia analizę z pracy.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="sprawdza zapisane wyniki bez ponownego liczenia",
    )
    args = parser.parse_args()
    check_results() if args.check else run_all()


if __name__ == "__main__":
    main()
