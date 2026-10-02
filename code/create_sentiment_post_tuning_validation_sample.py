from __future__ import annotations

from pathlib import Path

import pandas as pd

import create_sentiment_validation_sample as sample_generator


# Konfiguracja kolejnej, niezaleznej proby walidacyjnej

root = Path(__file__).resolve().parents[1]
sentiment_dir = root / "processed" / "sentiment"
annotation_dir = sentiment_dir / "validation"
validation_dir = annotation_dir / "tuned"
model_dir = sentiment_dir / "model"

reviews_csv = sentiment_dir / "product_reviews.csv"
products_csv = sentiment_dir / "products_sentiment_base.csv"
mentions_csv = model_dir / "review_aspect_sentiment.csv"
development_key_csv = annotation_dir / "manual_annotation_key_200.csv"
first_validation_key_csv = (
    annotation_dir / "initial" / "manual_validation_key_150.csv"
)

sample_csv = validation_dir / "manual_post_tuning_validation_sample_150.csv"
key_csv = validation_dir / "manual_post_tuning_validation_key_150.csv"
summary_csv = validation_dir / "manual_post_tuning_validation_summary.csv"


# Losowanie i zapis


def main() -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)
    mentions = pd.read_csv(mentions_csv)
    reviews = pd.read_csv(reviews_csv)
    products = pd.read_csv(products_csv)

    # Obie wczesniejsze proby stanowia material rozwojowy i nie moga wejsc
    # do kolejnej oceny niezaleznej.
    excluded_key = pd.concat(
        [pd.read_csv(development_key_csv), pd.read_csv(first_validation_key_csv)],
        ignore_index=True,
    )
    sample_generator.random_state = 20260719
    blind_sample, key = sample_generator.build_validation_sample(
        mentions, reviews, products, excluded_key
    )

    annotation_ids = [f"R2{number:03d}" for number in range(1, len(blind_sample) + 1)]
    blind_sample["annotation_id"] = annotation_ids
    key["annotation_id"] = annotation_ids

    blind_sample.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    key.to_csv(key_csv, index=False, encoding="utf-8-sig")
    summary = (
        key.groupby(["sample_type", "sampling_stratum"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["sample_type", "sampling_stratum"])
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"Saved {len(blind_sample)} rows to {sample_csv}")
    print(f"Saved post-tuning key to {key_csv}")


if __name__ == "__main__":
    main()
