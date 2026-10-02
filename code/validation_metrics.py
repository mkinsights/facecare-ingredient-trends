from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import evaluate_sentiment_post_tuning_validation as validation


SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def support(rows: int) -> str:
    if rows >= 50:
        return "stable"
    if rows >= 20:
        return "exploratory"
    return "insufficient"


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else np.nan


def model_quality_by_category(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in data.groupby("category1", sort=True):
        manual = group["manual_has_aspect"]
        model = group["model_has_aspect"]
        true_positive = int((manual & model).sum())
        false_positive = int((~manual & model).sum())
        false_negative = int((manual & ~model).sum())
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, true_positive + false_negative)
        detection_f1 = safe_ratio(2 * precision * recall, precision + recall)
        matched = group.loc[group["aspect_match"]]
        sentiment_f1 = (
            f1_score(
                matched["manual_sentiment"],
                matched["model_sentiment_for_manual_aspect"],
                labels=SENTIMENT_LABELS,
                average="macro",
                zero_division=0,
            )
            if len(matched)
            else np.nan
        )
        rows.append(
            {
                "category1": category,
                "validation_rows": len(group),
                "manual_aspect_rows": int(manual.sum()),
                "aspect_detection_precision": precision,
                "aspect_detection_recall": recall,
                "aspect_detection_f1": detection_f1,
                "primary_aspect_recall": group.loc[manual, "aspect_match"].mean(),
                "sentiment_rows_same_aspect": len(matched),
                "sentiment_accuracy_same_aspect": matched[
                    "sentiment_match_same_aspect"
                ].eq(True).mean(),
                "sentiment_macro_f1_same_aspect": sentiment_f1,
                "support_status": support(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("validation_rows", ascending=False)


def model_quality_by_category_aspect(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in sorted(data["category1"].unique()):
        category_data = data.loc[data["category1"].eq(category)]
        for aspect in sorted(validation.aspect_labels):
            gold = category_data["gold_aspects"].apply(lambda values: aspect in values)
            predicted = category_data["model_aspects"].apply(
                lambda values: aspect in values
            )
            matched = category_data.loc[gold & predicted]
            gold_mentions = int(gold.sum())
            if not gold_mentions:
                continue
            rows.append(
                {
                    "category1": category,
                    "aspect": aspect,
                    "gold_mentions": gold_mentions,
                    "detected_mentions": int((gold & predicted).sum()),
                    "aspect_recall": (gold & predicted).sum() / gold_mentions,
                    "sentiment_evaluable_mentions": len(matched),
                    "sentiment_accuracy": (
                        matched["manual_sentiment"]
                        .eq(matched["model_sentiment_for_manual_aspect"])
                        .mean()
                        if len(matched)
                        else np.nan
                    ),
                    "support_status": support(gold_mentions),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["category1", "gold_mentions"], ascending=[True, False]
    )


def model_quality_by_category_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = data.loc[data["aspect_match"]]
    for (category, sentiment), group in matched.groupby(
        ["category1", "manual_sentiment"], sort=True
    ):
        rows.append(
            {
                "category1": category,
                "manual_sentiment": sentiment,
                "evaluable_mentions": len(group),
                "class_recall_same_aspect": group[
                    "model_sentiment_for_manual_aspect"
                ].eq(sentiment).mean(),
                "support_status": support(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["category1", "evaluable_mentions"], ascending=[True, False]
    )
