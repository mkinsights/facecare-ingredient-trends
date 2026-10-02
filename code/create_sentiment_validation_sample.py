from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_model import split_into_aspect_fragments


# Konfiguracja niezaleznej proby walidacyjnej

root = Path(__file__).resolve().parents[1]
sentiment_dir = root / "processed" / "sentiment"
annotation_dir = sentiment_dir / "validation"
validation_dir = annotation_dir / "initial"
model_dir = sentiment_dir / "model"

reviews_csv = sentiment_dir / "product_reviews.csv"
products_csv = sentiment_dir / "products_sentiment_base.csv"
mentions_csv = model_dir / "review_aspect_sentiment.csv"
development_key_csv = annotation_dir / "manual_annotation_key_200.csv"

validation_sample_csv = validation_dir / "manual_validation_sample_150.csv"
validation_key_csv = validation_dir / "manual_validation_key_150.csv"
validation_summary_csv = validation_dir / "manual_validation_sample_150_summary.csv"

random_state = 20260718
candidate_per_aspect = 10
hard_cases_per_aspect = 3
no_aspect_controls = 20


# Funkcje pomocnicze


def fragment_key(data: pd.DataFrame) -> pd.Series:
    """Tworzy staly identyfikator fragmentu opinii."""
    return data["review_id"].astype(str) + "_" + data["fragment_index"].astype(str)


def draw_unique_fragments(
    data: pd.DataFrame,
    size: int,
    unavailable_keys: set[str],
    selected_keys: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Losuje fragmenty, ktore nie byly uzyte w zadnej probie."""
    unavailable = unavailable_keys | selected_keys
    available = data.loc[~data["_fragment_key"].isin(unavailable)].copy()
    if size <= 0:
        return available.iloc[0:0].copy()
    if len(available) < size:
        raise ValueError(f"Need {size} available fragments, found {len(available)}")

    chosen_index = rng.choice(available.index.to_numpy(), size=size, replace=False)
    selected = available.loc[chosen_index].copy()
    selected_keys.update(selected["_fragment_key"].tolist())
    return selected


def build_no_aspect_controls(
    reviews: pd.DataFrame,
    mentions: pd.DataFrame,
    unavailable_keys: set[str],
    selected_keys: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Dobiera fragmenty bez wykrytego aspektu jako kontrole czulosci."""
    detected_keys = set(fragment_key(mentions))
    rows: list[dict[str, object]] = []

    for review in reviews.itertuples(index=False):
        fragments = split_into_aspect_fragments(str(review.review_text))
        for index, text in enumerate(fragments, start=1):
            key = f"{review.review_id}_{index}"
            if key in detected_keys:
                continue
            rows.append(
                {
                    "review_id": review.review_id,
                    "id": review.id,
                    "review_rating": review.review_rating,
                    "fragment_index": index,
                    "aspect_text": text,
                    "_fragment_key": key,
                    "sample_type": "no_aspect_control",
                    "sampling_stratum": "no_detected_aspect",
                    "candidate_aspect": pd.NA,
                    "model_sentiment_label": pd.NA,
                    "sentiment_method": pd.NA,
                    "matched_terms": pd.NA,
                    "oof_fold": pd.NA,
                }
            )

    controls = pd.DataFrame(rows)
    low_rating = controls.loc[controls["review_rating"] <= 3]
    high_rating = controls.loc[controls["review_rating"] >= 4]
    first = draw_unique_fragments(
        low_rating,
        no_aspect_controls // 2,
        unavailable_keys,
        selected_keys,
        rng,
    )
    second = draw_unique_fragments(
        high_rating,
        no_aspect_controls - len(first),
        unavailable_keys,
        selected_keys,
        rng,
    )
    return pd.concat([first, second], ignore_index=True)


# Dobor proby z wylaczeniem proby rozwojowej


def build_validation_sample(
    mentions: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    development_key: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buduje slepa probe oraz techniczny klucz do pozniejszej ewaluacji."""
    rng = np.random.default_rng(random_state)
    unavailable_keys = set(fragment_key(development_key))
    selected_keys: set[str] = set()

    candidates = mentions.copy()
    candidates["_fragment_key"] = fragment_key(candidates)
    candidates["sample_type"] = "candidate_aspect"
    candidates["sampling_stratum"] = "standard_candidate"
    candidates = candidates.rename(
        columns={"aspect": "candidate_aspect", "sentiment_label": "model_sentiment_label"}
    )

    selected_parts: list[pd.DataFrame] = []
    for aspect in sorted(candidates["candidate_aspect"].dropna().unique()):
        aspect_rows = candidates.loc[candidates["candidate_aspect"].eq(aspect)].copy()
        hard_cases = aspect_rows.loc[
            aspect_rows["model_sentiment_label"].ne("positive")
            | aspect_rows["sentiment_method"].ne("ordinal_model")
        ].copy()
        hard_cases["sampling_stratum"] = "hard_case"
        hard_available = hard_cases.loc[
            ~hard_cases["_fragment_key"].isin(unavailable_keys | selected_keys)
        ]
        hard_selection = draw_unique_fragments(
            hard_cases,
            min(hard_cases_per_aspect, len(hard_available)),
            unavailable_keys,
            selected_keys,
            rng,
        )
        standard_selection = draw_unique_fragments(
            aspect_rows,
            candidate_per_aspect - len(hard_selection),
            unavailable_keys,
            selected_keys,
            rng,
        )
        selected_parts.append(pd.concat([hard_selection, standard_selection], ignore_index=True))

    reviews = reviews.copy().reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))
    controls = build_no_aspect_controls(
        reviews,
        mentions,
        unavailable_keys,
        selected_keys,
        rng,
    )
    selected = pd.concat([*selected_parts, controls], ignore_index=True, sort=False)

    expected_rows = candidate_per_aspect * len(selected_parts) + no_aspect_controls
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(selected)}")
    if selected["_fragment_key"].duplicated().any():
        raise ValueError("A fragment appears more than once in the validation sample")
    if set(selected["_fragment_key"]) & unavailable_keys:
        raise ValueError("Validation sample overlaps with the development sample")

    selected = selected.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    selected.insert(0, "annotation_id", [f"V{number:03d}" for number in range(1, len(selected) + 1)])
    selected = selected.merge(
        products[["id", "category"]], on="id", how="left", validate="many_to_one"
    )

    blind_sample = selected[
        ["annotation_id", "sample_type", "review_id", "id", "aspect_text"]
    ].copy()
    blind_sample["manual_aspect"] = pd.NA
    blind_sample["manual_sentiment"] = pd.NA
    blind_sample["has_negation"] = pd.NA
    blind_sample["manual_notes"] = pd.NA

    key = selected[
        [
            "annotation_id", "sample_type", "sampling_stratum", "review_id", "id",
            "category", "review_rating", "oof_fold", "fragment_index", "aspect_text",
            "candidate_aspect", "model_sentiment_label", "sentiment_method", "matched_terms",
        ]
    ].copy()
    return blind_sample, key


# Zapis i kontrola jakosci


def main() -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)
    mentions = pd.read_csv(mentions_csv)
    reviews = pd.read_csv(reviews_csv)
    products = pd.read_csv(products_csv)
    development_key = pd.read_csv(development_key_csv)

    blind_sample, key = build_validation_sample(mentions, reviews, products, development_key)
    blind_sample.to_csv(validation_sample_csv, index=False, encoding="utf-8-sig")
    key.to_csv(validation_key_csv, index=False, encoding="utf-8-sig")

    summary = (
        key.groupby(["sample_type", "sampling_stratum"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["sample_type", "sampling_stratum"])
    )
    summary.to_csv(validation_summary_csv, index=False, encoding="utf-8-sig")

    print(f"Saved {len(blind_sample)} rows to {validation_sample_csv}")
    print(f"Saved validation key to {validation_key_csv}")


if __name__ == "__main__":
    main()
