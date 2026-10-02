from __future__ import annotations

from pathlib import Path

import pandas as pd


# Konfiguracja

root = Path(__file__).resolve().parents[1]
CSV_FLOAT_FORMAT = "%.2f"
products_input_csv = root / "processed" / "products_analysis_processed.csv"
reviews_input_csv = root / "raw" / "product_reviews.csv"

sentiment_dir = root / "processed" / "sentiment"
sentiment_products_csv = sentiment_dir / "products_sentiment_base.csv"
sentiment_reviews_csv = sentiment_dir / "product_reviews.csv"

product_id_column = "id"
review_product_id_column = "product_id"
review_rating_column = "review_rating"


ingredient_flag_columns = [
    "is_niacinamide",
    "is_retinoids",
    "is_ceramides",
    "is_vitamin_c",
    "is_panthenol",
    "is_peptides",
    "is_hyaluronic_acid",
    "is_centella_asiatica",
    "is_salicylic_acid",
    "is_squalane",
    "is_aha",
]

formula_columns = [
    "ingredients_count",
    "active_count",
    "active_share",
    "active_position_score",
    "fragrance_count",
    "fragrance_allergen_count",
    "preservative_count",
    "is_alcohol_denat",
    "drying_alcohol_count",
    "is_drying_alcohol",
    "is_parfum",
    "is_colorant",
]

product_columns = [
    product_id_column,
    "brand",
    "category",
    "stars",
    "opinions",
    "log_opinions",
    "weighted_rating",
    "standardized_price_per_unit",
    "log_standardized_price",
    *formula_columns,
    *ingredient_flag_columns,
]

review_drop_columns = [
    "product_url",
    "review_index",
    "product_name",
]


# Funkcje pomocnicze


def require_columns(data: pd.DataFrame, columns: set[str], source_name: str) -> None:
    """Przerywa działanie, jeśli dane wejściowe nie mają wymaganych kolumn."""
    missing = columns - set(data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"{source_name} is missing required columns: {missing_columns}")


def build_sentiment_product_base(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy lekki zbiór produktów z cechami potrzebnymi do analizy sentymentu."""
    required_columns = {
        product_id_column,
        "brand",
        "category1",
        "category2",
        "stars",
        "opinions",
        "log_opinions",
        "weighted_rating",
        "standardized_price_per_unit",
        "log_standardized_price",
        *formula_columns,
        *ingredient_flag_columns,
    }
    require_columns(products, required_columns, "products_analysis_processed")

    products = products.copy()
    products["category"] = (
        products["category1"].fillna("").astype(str)
        + " - "
        + products["category2"].fillna("").astype(str)
    )
    return products[product_columns].copy()


def normalize_review_text(text: pd.Series) -> pd.Series:
    """Normalizuje tekst wyłącznie na potrzeby wykrywania duplikatów."""
    return (
        text.astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare_reviews(
    reviews: pd.DataFrame,
    allowed_product_ids: pd.Series,
) -> pd.DataFrame:
    """Przygotowuje opinie należące do analizowanej populacji produktów."""
    require_columns(
        reviews,
        {review_product_id_column, review_rating_column, "review_text"},
        "product_reviews",
    )
    reviews = reviews.drop(columns=review_drop_columns, errors="ignore").copy()
    reviews = reviews.rename(columns={review_product_id_column: product_id_column})

    # Lista produktów pochodzi z products_analysis_processed, dlatego dziedziczy
    # wszystkie zastosowane tam filtry kategorii i wykluczenia akcesoriów.
    reviews = reviews.loc[
        reviews[product_id_column].isin(set(allowed_product_ids.dropna()))
    ].copy()

    # Opinie bez oceny nie mają etykiety potrzebnej do walidacji sentymentu.
    reviews = reviews.loc[reviews[review_rating_column].notna()].copy()

    rating_text = reviews[review_rating_column].astype("string").str.strip()
    parsed_rating = rating_text.str.extract(
        r"^([1-5])(?:[.,]0+)?(?:\s*/\s*5)?$",
        expand=False,
    )
    invalid_rating = parsed_rating.isna()
    if invalid_rating.any():
        invalid_values = sorted(rating_text.loc[invalid_rating].unique().tolist())
        raise ValueError(
            "product_reviews contains unsupported review_rating values: "
            + ", ".join(invalid_values)
        )

    reviews[review_rating_column] = parsed_rating.astype("int64")
    reviews["review_text"] = reviews["review_text"].astype("string").str.strip()
    reviews = reviews.loc[
        reviews["review_text"].notna() & reviews["review_text"].ne("")
    ].copy()

    # Deduplikacja uwzględnia różnice wielkości liter i wielokrotne spacje,
    # ale zachowuje jednakowe wypowiedzi występujące przy różnych produktach.
    reviews["_normalized_review_text"] = normalize_review_text(
        reviews["review_text"]
    )
    conflicting_rating = (
        reviews.groupby(
            [product_id_column, "_normalized_review_text"]
        )[review_rating_column]
        .transform("nunique")
        .gt(1)
    )
    reviews = reviews.loc[~conflicting_rating].copy()
    reviews = reviews.drop_duplicates(
        subset=[
            product_id_column,
            review_rating_column,
            "_normalized_review_text",
        ],
        keep="first",
    )
    return reviews.drop(columns="_normalized_review_text").reset_index(drop=True)


# Uruchomienie etapu przygotowania danych do sentymentu


def main() -> None:
    """Zapisuje lekkie zbiory wejściowe do dalszej analizy sentymentu."""
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(products_input_csv)
    reviews = pd.read_csv(reviews_input_csv)

    sentiment_products = build_sentiment_product_base(products)
    prepared_reviews = prepare_reviews(
        reviews,
        sentiment_products[product_id_column],
    )

    sentiment_products.to_csv(
        sentiment_products_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    prepared_reviews.to_csv(
        sentiment_reviews_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )

    print(f"Saved {len(sentiment_products)} rows to {sentiment_products_csv}")
    print(f"Saved {len(prepared_reviews)} rows to {sentiment_reviews_csv}")


if __name__ == "__main__":
    main()
