from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_model import split_into_aspect_fragments


# Konfiguracja rozszerzonej walidacji warstwowej

root = Path(__file__).resolve().parents[1]
sentiment_dir = root / "processed" / "sentiment"
annotation_dir = sentiment_dir / "validation"
output_dir = annotation_dir / "final"
model_dir = sentiment_dir / "model"

mentions_csv = model_dir / "review_aspect_sentiment.csv"
reviews_csv = sentiment_dir / "product_reviews.csv"
products_csv = root / "processed" / "products_analysis_processed.csv"

excluded_key_paths = [
    annotation_dir / "manual_annotation_key_200.csv",
    annotation_dir / "initial" / "manual_validation_key_150.csv",
    annotation_dir
    / "tuned"
    / "manual_post_tuning_validation_key_150.csv",
]

sample_csv = output_dir / "manual_stratified_validation_sample_240.csv"
key_csv = output_dir / "manual_stratified_validation_key_240.csv"
summary_csv = output_dir / "manual_stratified_validation_summary.csv"

random_state = 20260720

# Liczebnosci celowo nie odzwierciedlaja struktury populacji. Mniejsze i slabiej
# zwalidowane kategorie sa nadreprezentowane, aby ocenic jakosc modelu.
candidate_category_quotas = {
    "Pielęgnacja twarzy": 65,
    "Oczyszczanie i demakijaż twarzy": 55,
    "Pielęgnacja ust": 45,
    "Tonizowanie twarzy": 35,
}
controls_per_category = 10
sentiment_shares = {
    "negative": 0.40,
    "neutral": 0.30,
    "positive": 0.30,
}


# Funkcje pomocnicze


def fragment_key(data: pd.DataFrame) -> pd.Series:
    """Tworzy niezmienny klucz fragmentu opinii."""
    return data["review_id"].astype(str) + "_" + data["fragment_index"].astype(str)


def load_excluded_keys() -> set[str]:
    """Laczy fragmenty ze wszystkich wczesniejszych prob badawczych."""
    frames = [pd.read_csv(path) for path in excluded_key_paths]
    excluded = pd.concat(frames, ignore_index=True)
    return set(fragment_key(excluded))


def allocate_sentiment_quotas(total: int) -> dict[str, int]:
    """Przelicza udzialy klas na liczby sumujace sie do zadanej wielkosci."""
    quotas = {
        sentiment: int(np.floor(total * share))
        for sentiment, share in sentiment_shares.items()
    }
    missing = total - sum(quotas.values())
    for sentiment in ["neutral", "negative", "positive"][:missing]:
        quotas[sentiment] += 1
    return quotas


def diversify_aspects(
    pool: pd.DataFrame,
    size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Losuje warstwe, ograniczajac dominacje jednego latwego aspektu."""
    if len(pool) < size:
        raise ValueError(f"Need {size} fragments, found {len(pool)}")

    shuffled = pool.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1)))
    aspect_order = shuffled["candidate_aspect"].value_counts().index.tolist()
    selected_indices: list[int] = []
    selected_set: set[int] = set()

    # Pobieranie rundami daje kazdemu dostepnemu aspektowi szanse wejscia
    # do proby, a dopiero pozniej dopelnia najliczniejsze warstwy.
    aspect_groups = {
        aspect: group.index.tolist()
        for aspect, group in shuffled.groupby("candidate_aspect", sort=False)
    }
    position = 0
    while len(selected_indices) < size:
        added = False
        for aspect in aspect_order:
            indices = aspect_groups[aspect]
            if position < len(indices):
                index = indices[position]
                if index not in selected_set:
                    selected_indices.append(index)
                    selected_set.add(index)
                    added = True
                    if len(selected_indices) == size:
                        break
        if not added:
            break
        position += 1

    if len(selected_indices) < size:
        remaining = shuffled.loc[~shuffled.index.isin(selected_set)]
        extra = rng.choice(
            remaining.index.to_numpy(),
            size=size - len(selected_indices),
            replace=False,
        )
        selected_indices.extend(extra.tolist())
    return pool.loc[selected_indices].copy()


# Losowanie fragmentow z wykrytym aspektem


def build_candidate_sample(
    mentions: pd.DataFrame,
    products: pd.DataFrame,
    excluded_keys: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Losuje category1 x sentyment, dbajac o zroznicowanie aspektow."""
    candidates = mentions.rename(
        columns={
            "aspect": "candidate_aspect",
            "sentiment_label": "model_sentiment_label",
        }
    ).copy()
    candidates["_fragment_key"] = fragment_key(candidates)
    candidates = candidates.loc[
        ~candidates["_fragment_key"].isin(excluded_keys)
    ].copy()
    candidates = candidates.merge(
        products[["id", "category1", "category2"]],
        on="id",
        how="left",
        validate="many_to_one",
    )
    candidates = candidates.loc[
        candidates["category1"].isin(candidate_category_quotas)
    ].copy()

    # Jeden fragment moze miec kilka wykrytych aspektow. W probie pozostaje
    # tylko raz; klucz techniczny zachowuje aspekt, wedlug ktorego go dobrano.
    candidates = candidates.sample(
        frac=1, random_state=random_state
    ).drop_duplicates("_fragment_key")

    selected: list[pd.DataFrame] = []
    selected_keys: set[str] = set()
    for category, category_quota in candidate_category_quotas.items():
        sentiment_quotas = allocate_sentiment_quotas(category_quota)
        for sentiment, quota in sentiment_quotas.items():
            pool = candidates.loc[
                candidates["category1"].eq(category)
                & candidates["model_sentiment_label"].eq(sentiment)
                & ~candidates["_fragment_key"].isin(selected_keys)
            ].copy()
            chosen = diversify_aspects(pool, quota, rng)
            chosen["sample_type"] = "candidate_aspect"
            chosen["sampling_stratum"] = (
                category + " | predicted_" + sentiment
            )
            selected.append(chosen)
            selected_keys.update(chosen["_fragment_key"])
    result = pd.concat(selected, ignore_index=True)
    if result["_fragment_key"].duplicated().any():
        raise ValueError("Candidate sample contains duplicate fragments")
    return result


# Kontrole bez wykrytego aspektu


def build_control_pool(
    reviews: pd.DataFrame,
    mentions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Rozbija opinie i zostawia fragmenty bez aspektu wykrytego przez model."""
    detected_keys = set(fragment_key(mentions))
    rows: list[dict[str, object]] = []
    for review in reviews.itertuples(index=False):
        for fragment_index, aspect_text in enumerate(
            split_into_aspect_fragments(str(review.review_text)),
            start=1,
        ):
            key = f"{review.review_id}_{fragment_index}"
            if key in detected_keys:
                continue
            rows.append(
                {
                    "review_id": review.review_id,
                    "id": review.id,
                    "review_rating": review.review_rating,
                    "fragment_index": fragment_index,
                    "aspect_text": aspect_text,
                    "_fragment_key": key,
                }
            )
    controls = pd.DataFrame(rows)
    return controls.merge(
        products[["id", "category1", "category2"]],
        on="id",
        how="left",
        validate="many_to_one",
    )


def build_control_sample(
    reviews: pd.DataFrame,
    mentions: pd.DataFrame,
    products: pd.DataFrame,
    excluded_keys: set[str],
    candidate_keys: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Losuje po 10 kontroli w kategorii, dzielac je wedlug ratingu."""
    controls = build_control_pool(reviews, mentions, products)
    controls = controls.loc[
        ~controls["_fragment_key"].isin(excluded_keys | candidate_keys)
    ].copy()
    selected: list[pd.DataFrame] = []
    for category in candidate_category_quotas:
        category_pool = controls.loc[controls["category1"].eq(category)]
        low = category_pool.loc[category_pool["review_rating"].le(3)]
        high = category_pool.loc[category_pool["review_rating"].ge(4)]
        low_size = controls_per_category // 2
        high_size = controls_per_category - low_size
        if len(low) < low_size or len(high) < high_size:
            raise ValueError(f"Not enough no-aspect controls for {category}")
        low_choice = rng.choice(low.index, size=low_size, replace=False)
        high_choice = rng.choice(high.index, size=high_size, replace=False)
        chosen = controls.loc[[*low_choice, *high_choice]].copy()
        chosen["sample_type"] = "no_aspect_control"
        chosen["sampling_stratum"] = category + " | no_detected_aspect"
        chosen["candidate_aspect"] = pd.NA
        chosen["model_sentiment_label"] = pd.NA
        chosen["sentiment_method"] = pd.NA
        chosen["matched_terms"] = pd.NA
        chosen["oof_fold"] = pd.NA
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


# Zapis slepej proby i klucza technicznego


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    mentions = pd.read_csv(mentions_csv)
    reviews = pd.read_csv(reviews_csv).reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))
    products = pd.read_csv(
        products_csv,
        usecols=["id", "category1", "category2"],
    )
    if products["id"].duplicated().any():
        raise ValueError("products_analysis_processed contains duplicate ids")

    excluded_keys = load_excluded_keys()
    candidates = build_candidate_sample(
        mentions, products, excluded_keys, rng
    )
    controls = build_control_sample(
        reviews,
        mentions,
        products,
        excluded_keys,
        set(candidates["_fragment_key"]),
        rng,
    )
    selected = pd.concat([candidates, controls], ignore_index=True, sort=False)
    selected = selected.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    expected_size = sum(candidate_category_quotas.values()) + (
        controls_per_category * len(candidate_category_quotas)
    )
    if len(selected) != expected_size:
        raise ValueError(f"Expected {expected_size} rows, got {len(selected)}")
    if selected["_fragment_key"].duplicated().any():
        raise ValueError("A fragment occurs more than once in the sample")
    if set(selected["_fragment_key"]) & excluded_keys:
        raise ValueError("The new sample overlaps an earlier annotation sample")

    annotation_ids = [
        f"H{number:03d}" for number in range(1, len(selected) + 1)
    ]
    selected.insert(0, "annotation_id", annotation_ids)

    blind = selected[
        [
            "annotation_id",
            "sample_type",
            "review_id",
            "id",
            "aspect_text",
        ]
    ].copy()
    blind["manual_aspect"] = pd.NA
    blind["manual_sentiment"] = pd.NA
    blind["has_negation"] = pd.NA
    blind["manual_notes"] = pd.NA

    key = selected[
        [
            "annotation_id",
            "sample_type",
            "sampling_stratum",
            "review_id",
            "id",
            "category1",
            "category2",
            "review_rating",
            "oof_fold",
            "fragment_index",
            "aspect_text",
            "candidate_aspect",
            "model_sentiment_label",
            "sentiment_method",
            "matched_terms",
        ]
    ].copy()

    blind.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    key.to_csv(key_csv, index=False, encoding="utf-8-sig")
    summary = (
        key.groupby(
            [
                "category1",
                "sample_type",
                "model_sentiment_label",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(["category1", "sample_type", "model_sentiment_label"])
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"Saved {len(blind)} rows to {sample_csv}")
    print(f"Saved technical key to {key_csv}")


if __name__ == "__main__":
    main()
