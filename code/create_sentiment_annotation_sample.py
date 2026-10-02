from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_model import split_into_aspect_fragments


# Konfiguracja proby do recznego oznaczania

root = Path(__file__).resolve().parents[1]
sentiment_dir = root / "processed" / "sentiment"
model_dir = sentiment_dir / "model"
annotation_dir = sentiment_dir / "validation"

reviews_csv = sentiment_dir / "product_reviews.csv"
products_csv = sentiment_dir / "products_sentiment_base.csv"
mentions_csv = model_dir / "review_aspect_sentiment.csv"

annotation_sample_csv = annotation_dir / "manual_annotation_sample_200.csv"
annotation_key_csv = annotation_dir / "manual_annotation_key_200.csv"
annotation_summary_csv = annotation_dir / "manual_annotation_sample_200_summary.csv"

random_state = 42
sample_size = 200
candidate_per_aspect = 12
additional_hard_cases = 4
no_aspect_controls = 40


# Funkcje pomocnicze


def fragment_key(data: pd.DataFrame) -> pd.Series:
    """Tworzy identyfikator fragmentu niezalezny od przypisanego aspektu."""
    return (
        data["review_id"].astype(str)
        + "_"
        + data["fragment_index"].astype(str)
    )


def draw_unique_fragments(
    data: pd.DataFrame,
    size: int,
    used_fragment_keys: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Losuje fragmenty bez ponownego uzycia tego samego tekstu w probie."""
    available = data.loc[~data["_fragment_key"].isin(used_fragment_keys)].copy()
    if available.empty or size <= 0:
        return available.iloc[0:0].copy()

    take = min(size, len(available))
    selected_index = rng.choice(available.index.to_numpy(), size=take, replace=False)
    selected = available.loc[selected_index].copy()
    used_fragment_keys.update(selected["_fragment_key"].tolist())
    return selected


def build_no_aspect_controls(
    reviews: pd.DataFrame,
    mentions: pd.DataFrame,
    used_fragment_keys: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Tworzy kontrole do oceny aspektow przeoczonych przez slownik."""
    detected_keys = set(fragment_key(mentions))
    rows: list[dict[str, object]] = []

    for review in reviews.itertuples(index=False):
        for index, text in enumerate(
            split_into_aspect_fragments(str(review.review_text)),
            start=1,
        ):
            key = f"{review.review_id}_{index}"
            if key in detected_keys or key in used_fragment_keys:
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
    if len(controls) < no_aspect_controls:
        raise ValueError("Not enough fragments without a detected aspect")

    negative_or_neutral = controls.loc[controls["review_rating"] <= 3]
    positive = controls.loc[controls["review_rating"] >= 4]
    first_part = draw_unique_fragments(
        negative_or_neutral,
        no_aspect_controls // 2,
        used_fragment_keys,
        rng,
    )
    second_part = draw_unique_fragments(
        positive,
        no_aspect_controls - len(first_part),
        used_fragment_keys,
        rng,
    )
    controls = pd.concat([first_part, second_part], ignore_index=True)
    if len(controls) != no_aspect_controls:
        raise ValueError("Could not draw the requested number of control fragments")
    return controls


# Dobor proby warstwowej


def build_annotation_sample(
    mentions: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buduje slepa probe dla osoby oznaczajacej oraz osobny klucz modelu."""
    rng = np.random.default_rng(random_state)
    candidates = mentions.copy()
    candidates["_fragment_key"] = fragment_key(candidates)
    candidates["sample_type"] = "candidate_aspect"
    candidates["sampling_stratum"] = "standard_candidate"
    candidates = candidates.rename(
        columns={
            "aspect": "candidate_aspect",
            "sentiment_label": "model_sentiment_label",
        }
    )

    selected_parts: list[pd.DataFrame] = []
    used_fragment_keys: set[str] = set()
    aspects = sorted(candidates["candidate_aspect"].dropna().unique())

    for aspect in aspects:
        aspect_rows = candidates.loc[
            candidates["candidate_aspect"].eq(aspect)
        ].copy()
        hard_cases = aspect_rows.loc[
            aspect_rows["model_sentiment_label"].ne("positive")
            | aspect_rows["sentiment_method"].ne("ordinal_model")
        ].copy()
        hard_cases["sampling_stratum"] = "hard_case"
        hard_selection = draw_unique_fragments(
            hard_cases,
            size=4,
            used_fragment_keys=used_fragment_keys,
            rng=rng,
        )
        standard_selection = draw_unique_fragments(
            aspect_rows,
            size=candidate_per_aspect - len(hard_selection),
            used_fragment_keys=used_fragment_keys,
            rng=rng,
        )
        aspect_selection = pd.concat(
            [hard_selection, standard_selection],
            ignore_index=True,
        )
        if len(aspect_selection) != candidate_per_aspect:
            raise ValueError(f"Could not draw {candidate_per_aspect} rows for {aspect}")
        selected_parts.append(aspect_selection)

    candidate_sample = pd.concat(selected_parts, ignore_index=True)
    hard_pool = candidates.loc[
        candidates["model_sentiment_label"].ne("positive")
        | candidates["sentiment_method"].ne("ordinal_model")
    ].copy()
    hard_pool["sampling_stratum"] = "additional_hard_case"
    extra_sample = draw_unique_fragments(
        hard_pool,
        size=additional_hard_cases,
        used_fragment_keys=used_fragment_keys,
        rng=rng,
    )
    if len(extra_sample) != additional_hard_cases:
        raise ValueError("Could not draw additional hard cases")

    reviews = reviews.copy().reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))
    controls = build_no_aspect_controls(
        reviews,
        mentions,
        used_fragment_keys,
        rng,
    )

    selected = pd.concat(
        [candidate_sample, extra_sample, controls],
        ignore_index=True,
        sort=False,
    )
    if len(selected) != sample_size:
        raise ValueError(f"Expected {sample_size} rows, got {len(selected)}")
    if selected["_fragment_key"].duplicated().any():
        raise ValueError("A text fragment appears more than once in the sample")

    selected = selected.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )
    selected.insert(
        0,
        "annotation_id",
        [f"A{number:03d}" for number in range(1, len(selected) + 1)],
    )
    selected = selected.merge(
        products[["id", "category"]],
        on="id",
        how="left",
        validate="many_to_one",
    )

    blind_sample = selected[
        [
            "annotation_id",
            "sample_type",
            "review_id",
            "id",
            "aspect_text",
        ]
    ].copy()
    blind_sample["manual_aspect"] = pd.NA
    blind_sample["manual_sentiment"] = pd.NA
    blind_sample["has_negation"] = pd.NA
    blind_sample["manual_notes"] = pd.NA

    key = selected[
        [
            "annotation_id",
            "sample_type",
            "sampling_stratum",
            "review_id",
            "id",
            "category",
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
    return blind_sample, key


# Zapis


def main() -> None:
    annotation_dir.mkdir(parents=True, exist_ok=True)
    mentions = pd.read_csv(mentions_csv)
    reviews = pd.read_csv(reviews_csv)
    products = pd.read_csv(products_csv)

    blind_sample, key = build_annotation_sample(mentions, reviews, products)
    blind_sample.to_csv(
        annotation_sample_csv,
        index=False,
        encoding="utf-8-sig",
    )
    key.to_csv(
        annotation_key_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        key.groupby(["sample_type", "sampling_stratum"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["sample_type", "sampling_stratum"])
    )
    summary.to_csv(
        annotation_summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved {len(blind_sample)} rows to {annotation_sample_csv}")
    print(f"Saved annotation key to {annotation_key_csv}")


if __name__ == "__main__":
    main()
