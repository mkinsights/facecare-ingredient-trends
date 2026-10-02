from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression


# Konfiguracja wejścia, wyjścia i modelu

root = Path(__file__).resolve().parents[1]
CSV_FLOAT_FORMAT = "%.2f"
sentiment_dir = root / "processed" / "sentiment"
reviews_input_csv = sentiment_dir / "product_reviews.csv"
products_input_csv = sentiment_dir / "products_sentiment_base.csv"

model_dir = sentiment_dir / "model"
model_path = model_dir / "hybrid_sentiment_model.joblib"
metrics_csv = model_dir / "sentiment_model_metrics.csv"
class_metrics_csv = model_dir / "sentiment_class_metrics.csv"
confusion_matrix_csv = model_dir / "sentiment_confusion_matrix.csv"
validation_predictions_csv = model_dir / "sentiment_validation_predictions.csv"
reviews_scored_csv = model_dir / "reviews_sentiment_scored.csv"
review_aspects_csv = model_dir / "review_aspect_sentiment.csv"
aspect_summary_csv = model_dir / "aspect_sentiment_summary.csv"
product_aspect_summary_csv = model_dir / "product_aspect_sentiment_summary.csv"
product_sentiment_summary_csv = model_dir / "product_sentiment_summary.csv"
metadata_json = model_dir / "sentiment_model_metadata.json"
validation_product_ids_csv = model_dir / "validation_product_ids.csv"

random_state = 42
oof_folds = 5
ordinal_thresholds = (1, 2, 3, 4)
class_weight_power = 0.35

# Progi zostały dobrane wyłącznie na ręcznej próbie rozwojowej przez
# maksymalizację trzyklasowego macro F1. Trzy późniejsze próby służą tylko
# do niezależnej oceny tej kalibracji.
sentiment_negative_score_threshold = 0.10
sentiment_positive_score_threshold = 0.40


def sentiment_labels_from_scores(
    sentiment_scores: pd.Series | np.ndarray,
) -> np.ndarray:
    """Przekształca ciągły wynik na skalibrowane klasy sentymentu."""
    values = np.asarray(sentiment_scores, dtype=float)
    return np.select(
        [
            values < sentiment_negative_score_threshold,
            values < sentiment_positive_score_threshold,
        ],
        ["negative", "neutral"],
        default="positive",
    )


# Taksonomia aspektów opinii kosmetycznych

# Reguły wskazują temat wypowiedzi. Sentyment aspektu jest wyznaczany osobno
# przez model nadzorowany na fragmencie tekstu, w którym znaleziono wzmiankę.
aspect_definitions = {
    "effectiveness": {
        "group": "performance",
        "patterns": [
            r"\bdziała\w*",
            r"\bskutecz\w*",
            r"\befekt\w*",
            r"\brezultat\w*",
            r"\bpopraw\w*",
            r"\bwygład\w*",
            r"\bujędr\w*",
            r"\bzmarszcz\w*",
            r"\bprzebarwie\w*",
            r"\bwyprysk\w*",
            r"\bniedoskonał\w*",
            r"\btrądzik\w*",
            r"\belastyczn\w*",
            r"\bodklej\w*",
            r"\b(?:nie\s+robi|nic\s+nie\s+(?:robi|działa)|nie\s+widać|"
            r"nie\s+widzę|nie\s+zauważ\w*|nie\s+spełnia\w*)",
            r"\b(?:sprawdza\w*\s+się|spełnia\w*\s+(?:swoj\w*\s+)?funkcj\w*|"
            r"wróc\w*\s+do\s+(?:tego\s+)?produkt\w*|ma\s+moc)\b",
        ],
    },
    "hydration_nourishment": {
        "group": "performance",
        "patterns": [
            r"\bnawilż\w*",
            r"\bodżyw\w*",
            r"\bwysusz\w*",
            r"\bprzesusz\w*",
            r"\bsuch\w*",
            r"\bściąg\w*",
            r"\bnapię\w*",
        ],
    },
    "skin_tolerance": {
        "group": "performance",
        "patterns": [
            r"\bpodrażn\w*",
            r"\buczula\w*",
            r"\balerg\w*",
            r"\bszczyp\w*",
            r"\bpiecz\w*",
            r"\bpieczen\w*",
            r"\bzaczerwien\w*",
            r"\bkoi\w*",
            r"\błagodz\w*",
            r"\bulg\w*",
            r"\bwrażliw\w*",
        ],
    },
    "pore_clogging": {
        "group": "performance",
        "patterns": [
            r"\bzapych\w*",
            r"\bzatyk\w*",
            r"\bkomedogen\w*",
            r"\boczyszcz\w*\s+por\w*",
        ],
    },
    "cleansing_makeup_removal": {
        "group": "performance",
        "patterns": [
            r"\boczyszcz\w*",
            r"\bdomyw\w*",
            r"\bzmyw\w*",
            r"\bdemakijaż\w*",
            r"\bmakijaż\w*",
            r"\bmyci\w*",
            r"\bmyje\w*",
            r"\bpian\w*",
            r"\brozpuszcz\w*",
        ],
    },
    "texture_consistency": {
        "group": "sensory",
        "patterns": [
            r"\bkonsystenc\w*",
            r"\btekstur\w*",
            r"\bgęst\w*",
            r"\brzadk\w*",
            r"\bkremow\w*",
            r"\bżelow\w*",
            r"\bwodn\w*",
            r"\blepk\w*",
            r"\bklej\w*",
            r"\btłust\w*",
            r"\bwysch\w*",
            r"\bnasącz\w*",
            r"\brozryw\w*",
            (
                r"\blekk\w*\s+(?:konsystenc\w*|tekstur\w*|formuł\w*|"
                r"krem\w*|serum\w*|żel\w*|emulsj\w*|balsam\w*)"
            ),
            (
                r"\b(?:konsystenc\w*|tekstur\w*|formuł\w*)"
                r"[^.!?]{0,20}\blekk\w*"
            ),
            (
                r"\b(?:krem\w*|serum\w*|żel\w*|emulsj\w*|balsam\w*)"
                r"\s+(?:jest|ma|wydaje\s+się)\s+(?:bardzo\s+|super\s+)?"
                r"lekk\w*"
            ),
            (
                r"\b(?:krem\w*|serum\w*|żel\w*|emulsj\w*|balsam\w*)"
                r"\s*[-,:]\s*lekk\w*"
            ),
        ],
    },
    "absorption_finish": {
        "group": "sensory",
        "patterns": [
            r"\bwchłan\w*",
            r"\bmatow\w*",
            r"\bbłyszcz\w*",
            r"\bwykończen\w*",
            r"\bfilm\w*",
            r"\bwarstw\w*",
            r"\brolu\w*",
            r"\bbieli\w*",
        ],
    },
    "scent": {
        "group": "sensory",
        "patterns": [
            r"\bzapach\w*",
            r"\bpachn\w*",
            r"\baromat\w*",
            r"\bperfum\w*",
            r"\bbezzapach\w*",
        ],
    },
    "application": {
        "group": "sensory",
        "patterns": [
            r"\baplik\w*",
            r"\bnakład\w*",
            r"\brozprowadz\w*",
            r"\bdozow\w*",
            r"\bbezproblem\w*",
            (
                r"\b(?:ułatw|ulatw)\w*[^.!?]{0,40}"
                r"\b(?:aplik|stosowan|nakład|dozow|pielęgnac)\w*"
            ),
            (
                r"\b(?:opakowan|pomp|pipet|konsystenc)\w*[^.!?]{0,40}"
                r"\b(?:ułatw|ulatw)\w*"
            ),
        ],
    },
    "packaging": {
        "group": "packaging",
        "patterns": [
            r"\bopakowan\w*",
            r"\bbutelk\w*",
            r"\btubk\w*",
            r"\bsłoiczk\w*",
            r"\bpomp\w*",
            r"\bdozownik\w*",
            r"\batomizer\w*",
            r"\bzamknię\w*",
            r"\bampułk\w*",
            r"\b(?:opilk|poturbow|przeciek|wadliw)\w*",
        ],
    },
    "price_value": {
        "group": "economic",
        "patterns": [
            r"\bcen\w*",
            r"\bdrog\w*",
            r"\btani\w*",
            r"\bwart\w*",
            r"\bopłac\w*",
            r"\bpromocj\w*",
        ],
    },
    "efficiency": {
        "group": "economic",
        "patterns": [
            r"\bwydajn\w*",
            r"\bstarcza\w*",
            r"\bwystarcza\w*",
            r"\bzuży\w*",
            r"\bkropl\w*",
            r"\bmał\w+\s+iloś\w*",
        ],
    },
    "formula_composition": {
        "group": "formula",
        "patterns": [
            r"\bskład\w*",
            r"\bformuł\w*",
            r"\bsubstancj\w*",
            r"\bingredient\w*",
            r"\balkohol\w*",
            r"\bparaben\w*",
            r"\bkwas\w*",
            r"\bwitamin\w*",
            r"\bretinol\w*",
            r"\bniacynamid\w*",
            r"\bceramid\w*",
            r"\bpeptyd\w*",
            r"\bmikroplastik\w*",
            r"\bpolimer\w*",
        ],
    },
}


# Jednoznaczne frazy domenowe korygują sentyment tylko dla wskazanego aspektu.
# Nie obejmują ogólnych słów typu "dobry" lub "zły", aby nie zastępować modelu.
explicit_aspect_sentiment_patterns = {
    "effectiveness": {
        "positive": [
            r"\b(?:sprawdza\w*\s+się|spełnia\w*\s+(?:swoj\w*\s+)?funkcj\w*|"
            r"wróc\w*\s+do\s+(?:tego\s+)?produkt\w*|ma\s+moc)\b",
        ],
        "negative": [
            r"\b(?:nic|zupełnie\s+nic)\s+nie\s+(?:robi|działa)\b",
            r"\bnie\s+(?:widzę|widać|zauważ\w*)\b[^.!?]{0,40}"
            r"\b(?:efekt|rezultat|dział)\w*",
            r"\bnie\s+spełnia\w*\s+(?:moich\s+)?oczekiwa\w*",
            r"\bcudów\s+nie\s+zadziała\w*",
        ],
    },
    "hydration_nourishment": {
        "positive": [
            r"\bnie\s+(?:wysusz|przesusz|ściąg)\w*",
        ],
        "negative": [
            r"\bnie\s+nawilż\w*",
            r"\bbrak\w*\s+nawilż\w*",
            r"\b(?:wysusz|przesusz|ściąga)\w*\s+(?:skór\w*|usta\w*)",
        ],
    },
    "skin_tolerance": {
        "negative": [
            r"\b(?:podrażnia|uczula|piecze|szczypie|zaczerwienia)\w*",
        ],
    },
    "cleansing_makeup_removal": {
        "negative": [
            r"\bnie\s+(?:zmyw|domyw|oczyszcz)\w*",
            r"\bmało\s+zmyw\w*",
            r"\bnie\s+polecam\b[^.!?]{0,40}\b(?:demakijaż|zmyw)\w*",
        ],
    },
    "texture_consistency": {
        "negative": [
            r"\b(?:zbyt\s+)?(?:rzadk|gęst|lepk|klej)\w*",
        ],
    },
    "absorption_finish": {
        "positive": [
            r"\bnie\s+(?:zostawia|tworzy|robi)\w*[^.!?]{0,30}"
            r"\b(?:film|warstw)\w*",
            r"\bnie\s+bieli\w*",
        ],
        "negative": [
            r"\bnie\s+wchłan\w*",
            r"\brolu\w*",
            r"\b(?:zostawia|tworzy)\w*[^.!?]{0,30}"
            r"\b(?:film|warstw)\w*",
            r"\b(?:obklej|perłow)\w*",
        ],
    },
    "scent": {
        "negative": [
            r"\b(?:nieprzyjemn|irytują|dziwn)\w*\s+(?:zapach|pachn|aromat)\w*",
            r"\b(?:zapach|pachn|aromat)\w*[^.!?]{0,30}"
            r"\b(?:nieprzyjemn|irytują|zbyt\s+intensywn|za\s+mocn|dziwn)\w*",
            r"\bminus\s+za\b[^.!?]{0,40}\b(?:zapach|pachn|aromat)\w*",
        ],
    },
    "application": {
        "negative": [
            r"\b(?:trudn|niewygodn)\w*\s+(?:aplik|nakład|rozprowadz)\w*",
            r"\bnie\s+(?:da|moż)\w*[^.!?]{0,30}"
            r"\b(?:aplik|nakład|rozprowadz)\w*",
        ],
    },
    "packaging": {
        "negative": [
            r"\b(?:opakowan|pomp|pipet|butelk|tubk|ampułk)\w*[^.!?]{0,50}"
            r"\b(?:tragicz|wadliw|przeciek|nie\s+nadaj|przesta\w*\s+działa|"
            r"poturbow|ciężko\s+otworzyć)\w*",
            r"\bminus\s+za\b[^.!?]{0,40}"
            r"\b(?:opakowan|pomp|pipet|butelk|tubk|ampułk)\w*",
        ],
    },
    "price_value": {
        "negative": [
            r"\bnie\s+(?:jest\s+)?wart\w*\s+(?:swojej\s+)?cen\w*",
            r"\b(?:nie\s+opłaca|za\s+drog|zbyt\s+drog)\w*",
        ],
    },
    "efficiency": {
        "negative": [
            r"\b(?:mało|nie)\s+wydajn\w*",
        ],
    },
    "formula_composition": {
        "positive": [
            r"\bnie\s+zawiera\w*[^.!?]{0,30}"
            r"\b(?:alkohol|paraben|mikroplastik|silikon)\w*",
        ],
        "negative": [
            r"\b(?:średni|słab|kiepsk|zły)\w*\s+(?:skład|formuł)\w*",
            r"\bza\s+dużo\s+(?:alkohol|paraben)\w*",
        ],
    },
}


# Model porządkowy sentymentu


@dataclass
class OrdinalSentimentModel:
    """Modeluje kolejne progi skali ocen i zachowuje jej porządek."""

    vectorizer: FeatureUnion
    threshold_models: list[LogisticRegression]
    thresholds: tuple[int, ...] = ordinal_thresholds

    @classmethod
    def create(cls) -> "OrdinalSentimentModel":
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        max_features=45_000,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=55_000,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )
        return cls(vectorizer=vectorizer, threshold_models=[])

    def fit(self, texts: Iterable[str], ratings: np.ndarray) -> "OrdinalSentimentModel":
        text_values = pd.Series(texts, dtype="string").fillna("").astype(str)
        matrix = self.vectorizer.fit_transform(text_values)
        self.threshold_models = []
        for threshold in self.thresholds:
            binary_target = (ratings > threshold).astype(int)
            class_counts = np.bincount(binary_target, minlength=2)
            balanced_weights = len(binary_target) / (2.0 * class_counts)
            class_weights = {
                class_number: float(weight**class_weight_power)
                for class_number, weight in enumerate(balanced_weights)
            }
            model = LogisticRegression(
                C=1.5,
                class_weight=class_weights,
                max_iter=1_000,
                solver="liblinear",
                random_state=random_state,
            )
            model.fit(matrix, binary_target)
            self.threshold_models.append(model)
        return self

    def predict(self, texts: Iterable[str]) -> pd.DataFrame:
        text_values = pd.Series(texts, dtype="string").fillna("").astype(str)
        matrix = self.vectorizer.transform(text_values)
        cumulative = np.column_stack(
            [model.predict_proba(matrix)[:, 1] for model in self.threshold_models]
        )

        # P(ocena > k) musi maleć wraz ze wzrostem progu k.
        cumulative = np.minimum.accumulate(cumulative, axis=1)
        class_probabilities = np.column_stack(
            [
                1.0 - cumulative[:, 0],
                cumulative[:, 0] - cumulative[:, 1],
                cumulative[:, 1] - cumulative[:, 2],
                cumulative[:, 2] - cumulative[:, 3],
                cumulative[:, 3],
            ]
        )
        class_probabilities = np.clip(class_probabilities, 0.0, 1.0)
        probability_sum = class_probabilities.sum(axis=1, keepdims=True)
        class_probabilities = np.divide(
            class_probabilities,
            probability_sum,
            out=np.zeros_like(class_probabilities),
            where=probability_sum != 0,
        )

        rating_values = np.arange(1, 6)
        expected_rating = class_probabilities @ rating_values
        predicted_rating = np.clip(np.rint(expected_rating), 1, 5).astype(int)
        sentiment_score = (expected_rating - 3.0) / 2.0

        negative_probability = class_probabilities[:, :2].sum(axis=1)
        neutral_probability = class_probabilities[:, 2]
        positive_probability = class_probabilities[:, 3:].sum(axis=1)
        sentiment_label = sentiment_labels_from_scores(sentiment_score)

        prediction = pd.DataFrame(
            {
                "predicted_rating": predicted_rating,
                "expected_rating": expected_rating,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "negative_probability": negative_probability,
                "neutral_probability": neutral_probability,
                "positive_probability": positive_probability,
            }
        )
        for class_number in rating_values:
            prediction[f"rating_{class_number}_probability"] = class_probabilities[
                :, class_number - 1
            ]
        return prediction


# Przygotowanie i walidacja danych


def load_model_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wczytuje dane i sprawdza warunki niezbędne do treningu."""
    reviews = pd.read_csv(reviews_input_csv)
    products = pd.read_csv(products_input_csv)

    required_review_columns = {"id", "review_rating", "review_text"}
    missing_review_columns = required_review_columns - set(reviews.columns)
    if missing_review_columns:
        raise ValueError(
            "product_reviews.csv is missing columns: "
            + ", ".join(sorted(missing_review_columns))
        )

    reviews = reviews.copy()
    reviews["review_rating"] = pd.to_numeric(
        reviews["review_rating"], errors="raise"
    ).astype(int)
    invalid_rating = ~reviews["review_rating"].between(1, 5)
    if invalid_rating.any():
        invalid_values = sorted(reviews.loc[invalid_rating, "review_rating"].unique())
        raise ValueError(f"Ratings outside 1-5: {invalid_values}")

    reviews["review_text"] = reviews["review_text"].astype("string").str.strip()
    empty_text = reviews["review_text"].isna() | reviews["review_text"].eq("")
    if empty_text.any():
        raise ValueError(f"Empty review texts: {int(empty_text.sum())}")

    if "id" not in products.columns:
        raise ValueError("products_sentiment_base.csv is missing column: id")
    if products["id"].duplicated().any():
        raise ValueError("Duplicate product ids in products_sentiment_base.csv")

    outside_scope = ~reviews["id"].isin(products["id"])
    if outside_scope.any():
        raise ValueError(
            "Reviews outside the filtered product population: "
            f"{int(outside_scope.sum())}"
        )

    normalized_text = normalize_review_text(reviews["review_text"])
    duplicate_reviews = pd.DataFrame(
        {
            "id": reviews["id"],
            "review_rating": reviews["review_rating"],
            "normalized_text": normalized_text,
        }
    ).duplicated()
    if duplicate_reviews.any():
        raise ValueError(
            "Duplicate normalized reviews remain after preparation: "
            f"{int(duplicate_reviews.sum())}"
        )

    reviews = reviews.reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))
    return reviews, products


def normalize_review_text(text: pd.Series) -> pd.Series:
    """Normalizuje tekst do kontroli duplikatów i przecieku między foldami."""
    return (
        text.astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def purge_training_text_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Usuwa z treningu teksty występujące również w bieżącym foldzie testowym."""
    test_texts = set(normalize_review_text(test["review_text"]))
    overlaps_test = normalize_review_text(train["review_text"]).isin(test_texts)
    return train.loc[~overlaps_test].copy(), int(overlaps_test.sum())


def make_oof_splits(
    reviews: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tworzy odtwarzalne foldy z rozłącznymi produktami."""
    splitter = StratifiedGroupKFold(
        n_splits=oof_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(
        splitter.split(
            reviews["review_text"],
            reviews["review_rating"],
            groups=reviews["id"],
        )
    )

    fold_rows: list[pd.DataFrame] = []
    observed_test_indices: list[int] = []
    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        train_products = set(reviews.iloc[train_index]["id"])
        test_products = set(reviews.iloc[test_index]["id"])
        if train_products & test_products:
            raise RuntimeError(f"Product leakage detected in fold {fold_number}")

        observed_test_indices.extend(test_index.tolist())
        fold_products = pd.DataFrame(
            {
                "id": sorted(test_products),
                "oof_fold": fold_number,
            }
        )
        fold_rows.append(fold_products)

    if sorted(observed_test_indices) != list(range(len(reviews))):
        raise RuntimeError("OOF split does not cover every review exactly once")

    pd.concat(fold_rows, ignore_index=True).sort_values("id").to_csv(
        validation_product_ids_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return splits


def three_class_label(rating: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(rating, dtype=float)
    return np.select(
        [values <= 2, values == 3],
        ["negative", "neutral"],
        default="positive",
    )


def evaluate_predictions(
    actual_rating: np.ndarray,
    prediction: pd.DataFrame,
    baseline_prediction: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Liczy metryki adekwatne do porządkowej i niezbalansowanej skali."""
    predicted_rating = prediction["predicted_rating"].to_numpy()
    expected_rating = prediction["expected_rating"].to_numpy()
    actual_sentiment = three_class_label(actual_rating)
    predicted_sentiment = prediction["sentiment_label"].to_numpy()
    spearman_result = spearmanr(actual_rating, expected_rating)

    class_probabilities = prediction[
        [f"rating_{rating}_probability" for rating in range(1, 6)]
    ].to_numpy()
    observed_probability = class_probabilities[
        np.arange(len(actual_rating)),
        np.asarray(actual_rating, dtype=int) - 1,
    ]
    confidence = class_probabilities.max(axis=1)
    exact_match = predicted_rating == actual_rating
    calibration_bins = np.minimum((confidence * 10).astype(int), 9)
    expected_calibration_error = 0.0
    for bin_number in range(10):
        in_bin = calibration_bins == bin_number
        if not in_bin.any():
            continue
        expected_calibration_error += float(
            in_bin.mean()
            * abs(exact_match[in_bin].mean() - confidence[in_bin].mean())
        )

    one_hot_rating = np.eye(5)[np.asarray(actual_rating, dtype=int) - 1]
    multiclass_brier = float(
        np.mean(np.sum((class_probabilities - one_hot_rating) ** 2, axis=1))
    )
    metrics = [
        ("test_reviews", float(len(actual_rating))),
        ("mae_expected_rating", mean_absolute_error(actual_rating, expected_rating)),
        (
            "rmse_expected_rating",
            mean_squared_error(actual_rating, expected_rating) ** 0.5,
        ),
        ("mae_rounded_rating", mean_absolute_error(actual_rating, predicted_rating)),
        ("exact_rating_accuracy", accuracy_score(actual_rating, predicted_rating)),
        (
            "within_one_star_accuracy",
            float(np.mean(np.abs(actual_rating - predicted_rating) <= 1)),
        ),
        (
            "quadratic_weighted_kappa",
            cohen_kappa_score(actual_rating, predicted_rating, weights="quadratic"),
        ),
        ("rating_spearman", float(spearman_result.statistic)),
        (
            "rating_balanced_accuracy",
            balanced_accuracy_score(actual_rating, predicted_rating),
        ),
        (
            "rating_macro_f1",
            f1_score(
                actual_rating,
                predicted_rating,
                labels=[1, 2, 3, 4, 5],
                average="macro",
                zero_division=0,
            ),
        ),
        (
            "sentiment_balanced_accuracy",
            balanced_accuracy_score(actual_sentiment, predicted_sentiment),
        ),
        (
            "sentiment_macro_f1",
            f1_score(
                actual_sentiment,
                predicted_sentiment,
                labels=["negative", "neutral", "positive"],
                average="macro",
                zero_division=0,
            ),
        ),
        (
            "multiclass_log_loss",
            log_loss(actual_rating, class_probabilities, labels=[1, 2, 3, 4, 5]),
        ),
        ("multiclass_brier_score", multiclass_brier),
        ("expected_calibration_error", expected_calibration_error),
        ("mean_observed_class_probability", float(observed_probability.mean())),
        (
            "training_majority_rating_accuracy",
            accuracy_score(actual_rating, baseline_prediction),
        ),
        (
            "training_majority_rating_mae",
            mean_absolute_error(actual_rating, baseline_prediction),
        ),
    ]
    metrics_frame = pd.DataFrame(metrics, columns=["metric", "value"])

    matrix = confusion_matrix(
        actual_rating,
        predicted_rating,
        labels=[1, 2, 3, 4, 5],
    )
    confusion_frame = pd.DataFrame(
        matrix,
        index=[f"actual_{rating}" for rating in range(1, 6)],
        columns=[f"predicted_{rating}" for rating in range(1, 6)],
    ).reset_index(names="actual_rating")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual_rating,
        predicted_rating,
        labels=[1, 2, 3, 4, 5],
        zero_division=0,
    )
    predicted_counts = pd.Series(predicted_rating).value_counts()
    class_metrics = pd.DataFrame(
        {
            "rating": range(1, 6),
            "support": support,
            "actual_share": support / len(actual_rating),
            "predicted_count": [
                int(predicted_counts.get(rating, 0)) for rating in range(1, 6)
            ],
            "predicted_share": [
                float((predicted_rating == rating).mean())
                for rating in range(1, 6)
            ],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return metrics_frame, confusion_frame, class_metrics


# Wykrywanie aspektów i sentymentu aspektowego


def compile_aspect_patterns() -> dict[str, list[re.Pattern[str]]]:
    return {
        aspect: [re.compile(pattern, flags=re.IGNORECASE) for pattern in definition["patterns"]]
        for aspect, definition in aspect_definitions.items()
    }


def split_into_aspect_fragments(text: str) -> list[str]:
    """Dzieli opinię na zdania i człony kontrastowe zachowujące kontekst."""
    sentence_parts = re.split(r"(?<=[.!?])\s+|[;\n]+", text)
    fragments: list[str] = []
    for sentence in sentence_parts:
        contrast_parts = re.split(
            r"\s+(?:ale|jednak|natomiast|lecz)\s+",
            sentence,
            flags=re.IGNORECASE,
        )
        fragments.extend(
            fragment.strip(" \t,.-")
            for fragment in contrast_parts
            if fragment.strip(" \t,.-")
        )
    return fragments


def extract_aspect_mentions(reviews: pd.DataFrame) -> pd.DataFrame:
    """Tworzy długi zbiór: jedna wzmianka o aspekcie w jednym fragmencie."""
    compiled_patterns = compile_aspect_patterns()
    rows: list[dict[str, object]] = []

    for review in reviews.itertuples(index=False):
        for fragment_index, fragment in enumerate(
            split_into_aspect_fragments(str(review.review_text)),
            start=1,
        ):
            for aspect, patterns in compiled_patterns.items():
                matched_terms = sorted(
                    {
                        match.group(0).lower()
                        for pattern in patterns
                        for match in pattern.finditer(fragment)
                    }
                )
                if not matched_terms:
                    continue
                rows.append(
                    {
                        "review_id": review.review_id,
                        "id": review.id,
                        "review_rating": review.review_rating,
                        "aspect": aspect,
                        "aspect_group": aspect_definitions[aspect]["group"],
                        "fragment_index": fragment_index,
                        "aspect_text": fragment,
                        "matched_terms": "; ".join(matched_terms),
                    }
                )

    return pd.DataFrame(rows)


def score_aspect_mentions(
    mentions: pd.DataFrame,
    model: OrdinalSentimentModel,
) -> pd.DataFrame:
    """Nadaje sentyment unikatowym fragmentom i dołącza go do wzmianek."""
    if mentions.empty:
        return mentions

    unique_fragments = mentions[["aspect_text"]].drop_duplicates().reset_index(drop=True)
    fragment_predictions = model.predict(unique_fragments["aspect_text"])
    unique_fragments = pd.concat(
        [unique_fragments, fragment_predictions],
        axis=1,
    )
    scored_mentions = mentions.merge(
        unique_fragments,
        on="aspect_text",
        how="left",
        validate="many_to_one",
    )
    scored_mentions["sentiment_method"] = "ordinal_model"
    scored_mentions = apply_explicit_aspect_sentiment_rules(scored_mentions)
    scored_mentions = apply_pore_clogging_rules(scored_mentions)
    scored_mentions = apply_skin_tolerance_negation_rules(scored_mentions)
    scored_mentions = apply_absorption_finish_negation_rules(scored_mentions)
    return apply_application_facilitation_rules(scored_mentions)


def apply_explicit_aspect_sentiment_rules(mentions: pd.DataFrame) -> pd.DataFrame:
    """Koryguje jednoznaczne negatywne i pozytywne oceny lokalnego aspektu."""
    for aspect, patterns_by_label in explicit_aspect_sentiment_patterns.items():
        is_aspect = mentions["aspect"].eq(aspect)
        if not is_aspect.any():
            continue

        aspect_text = mentions.loc[is_aspect, "aspect_text"].astype("string")
        # Najpierw ustawiamy negatywne reguły ogólne, a potem bardziej
        # szczegółowe pozytywne konstrukcje z negacją, np. "nie wysusza".
        for label in ("negative", "positive"):
            patterns = patterns_by_label.get(label, [])
            if not patterns:
                continue
            combined_pattern = "(?:" + "|".join(patterns) + ")"
            matching_indices = aspect_text.index[
                aspect_text.str.contains(
                    combined_pattern,
                    case=False,
                    regex=True,
                    na=False,
                )
            ]
            if label == "positive":
                probabilities = (0.0, 0.0, 0.0, 0.4, 0.6)
            else:
                probabilities = (0.6, 0.4, 0.0, 0.0, 0.0)
            set_rule_based_sentiment(
                mentions,
                matching_indices,
                class_probabilities=probabilities,
                label=label,
                method=f"explicit_{aspect}_{label}_rule",
            )
    return mentions


def apply_application_facilitation_rules(
    mentions: pd.DataFrame,
) -> pd.DataFrame:
    """Traktuje ułatwioną aplikację jako pozytywną cechę użytkową."""
    is_application = mentions["aspect"].eq("application")
    if not is_application.any():
        return mentions

    application_text = mentions.loc[is_application, "aspect_text"].astype("string")
    has_facilitation = application_text.str.contains(
        r"\b(?:ułatw|ulatw)\w*",
        case=False,
        regex=True,
        na=False,
    )
    has_local_negation = application_text.str.contains(
        r"\b(?:nie|wcale\s+nie)\s+(?:ułatw|ulatw)\w*",
        case=False,
        regex=True,
        na=False,
    )
    positive_indices = application_text.index[
        has_facilitation & ~has_local_negation
    ]
    set_rule_based_sentiment(
        mentions,
        positive_indices,
        class_probabilities=(0.0, 0.0, 0.0, 0.4, 0.6),
        label="positive",
        method="application_facilitation_rule",
    )
    return mentions


def apply_absorption_finish_negation_rules(
    mentions: pd.DataFrame,
) -> pd.DataFrame:
    """Nadaje pozytywny kierunek zanegowanemu rolowaniu produktu."""
    is_absorption_finish = mentions["aspect"].eq("absorption_finish")
    if not is_absorption_finish.any():
        return mentions

    finish_text = mentions.loc[is_absorption_finish, "aspect_text"].astype("string")
    no_rolling = (
        finish_text.str.contains(
            r"\b(?:nie|nigdy)\s+(?:\w+\s+){0,2}rolu\w*",
            case=False,
            regex=True,
            na=False,
        )
        | finish_text.str.contains(
            r"\bbez\s+rolow\w*",
            case=False,
            regex=True,
            na=False,
        )
        | finish_text.str.contains(
            r"\bnie\s+(?:mam|miał\w*)\b[^.!?]{0,50}\brolow\w*",
            case=False,
            regex=True,
            na=False,
        )
    )
    negated_indices = finish_text.index[no_rolling]
    set_rule_based_sentiment(
        mentions,
        negated_indices,
        class_probabilities=(0.0, 0.0, 0.0, 0.4, 0.6),
        label="positive",
        method="absorption_finish_negation_rule",
    )
    return mentions


def apply_skin_tolerance_negation_rules(mentions: pd.DataFrame) -> pd.DataFrame:
    """Nadaje pozytywny kierunek lokalnie zanegowanym reakcjom skóry."""
    is_skin_tolerance = mentions["aspect"].eq("skin_tolerance")
    if not is_skin_tolerance.any():
        return mentions

    tolerance_text = mentions.loc[is_skin_tolerance, "aspect_text"].astype("string")
    adverse_reaction = (
        r"(?:podrażn\w*|uczul\w*|alerg\w*|szczyp\w*|piecz\w*|"
        r"pieczen\w*|zaczerwien\w*)"
    )
    has_local_negation = (
        tolerance_text.str.contains(
            rf"\b(?:nie|nigdy|wcale|ani)\s+"
            rf"(?:(?:jest|są|był\w*|został\w*|mam|miał\w*|powod\w*|"
            rf"wywoł\w*|odczuw\w*|zauważ\w*|wystąp\w*|pojawi\w*)\s+)?"
            rf"(?:(?:u|na)\s+(?:mnie|skórze|oczach)\s+)?"
            rf"(?:żadn\w*\s+)?{adverse_reaction}",
            case=False,
            regex=True,
            na=False,
        )
        | tolerance_text.str.contains(
            rf"\b(?:bez|brak|zero)\s+(?:żadn\w*\s+)?{adverse_reaction}",
            case=False,
            regex=True,
            na=False,
        )
    )
    negated_indices = tolerance_text.index[has_local_negation]
    set_rule_based_sentiment(
        mentions,
        negated_indices,
        class_probabilities=(0.0, 0.0, 0.0, 0.4, 0.6),
        label="positive",
        method="skin_tolerance_negation_rule",
    )
    return mentions


def apply_pore_clogging_rules(mentions: pd.DataFrame) -> pd.DataFrame:
    """Koryguje kierunek fraz, których znaczenie zależy głównie od negacji."""
    is_pore_clogging = mentions["aspect"].eq("pore_clogging")
    if not is_pore_clogging.any():
        return mentions

    pore_text = mentions.loc[is_pore_clogging, "aspect_text"].astype("string")
    is_uncertain = pore_text.str.contains(
        r"\b(?:nie\s+wiem|trudno\s+powiedzieć|chyba|może)\b"
        r"[^.!?]{0,40}\b(?:zapych|zatyk)\w*",
        case=False,
        regex=True,
        na=False,
    )
    explicit_product_clogging = pore_text.str.contains(
        r"(?:\b(?:produkt|krem|serum|żel|emulsj|balsam|mnie)\b"
        r"[^.!?]{0,30}\b(?:zapych|zatyk)\w*|"
        r"\b(?:zapych|zatyk)\w*\s+pory\b)",
        case=False,
        regex=True,
        na=False,
    )
    skin_predisposition = pore_text.str.contains(
        r"(?:\b(?:skłonn\w*|tendencj\w*)\s+do\s+zapych\w*|"
        r"\błatwo\s+zapych\w*)",
        case=False,
        regex=True,
        na=False,
    )
    is_uncertain |= skin_predisposition & ~explicit_product_clogging

    is_positive = (
        pore_text.str.contains(
        r"\b(?:nie|nigdy|wcale)\s+(?:\w+\s+){0,3}(?:zapych|zatyk)\w*",
        case=False,
        regex=True,
        na=False,
        )
        | pore_text.str.contains(
            r"(?:\bnie\s+(?:powoduje|wywołuje|zauważ\w*)\b"
            r"[^.!?]{0,60}\b(?:zapych|zatyk)\w*|"
            r"\bbez\s+(?:zapych|zatyk)\w*)",
            case=False,
            regex=True,
            na=False,
        )
    ) & ~is_uncertain

    pore_indices = pore_text.index
    positive_indices = pore_indices[is_positive]
    uncertain_indices = pore_indices[is_uncertain]
    negative_indices = pore_indices[~is_positive & ~is_uncertain]

    set_rule_based_sentiment(
        mentions,
        positive_indices,
        class_probabilities=(0.0, 0.0, 0.0, 0.4, 0.6),
        label="positive",
        method="pore_clogging_negation_rule",
    )
    set_rule_based_sentiment(
        mentions,
        uncertain_indices,
        class_probabilities=(0.0, 0.0, 1.0, 0.0, 0.0),
        label="neutral",
        method="pore_clogging_uncertainty_rule",
    )
    set_rule_based_sentiment(
        mentions,
        negative_indices,
        class_probabilities=(0.6, 0.4, 0.0, 0.0, 0.0),
        label="negative",
        method="pore_clogging_affirmative_rule",
    )
    return mentions


def set_rule_based_sentiment(
    mentions: pd.DataFrame,
    indices: pd.Index,
    class_probabilities: tuple[float, float, float, float, float],
    label: str,
    method: str,
) -> None:
    """Ustawia spójny profil oceny dla jednoznacznej reguły dziedzinowej."""
    if len(indices) == 0:
        return

    rating_values = np.arange(1, 6)
    probabilities = np.asarray(class_probabilities, dtype=float)
    expected_rating = float(probabilities @ rating_values)
    mentions.loc[indices, "predicted_rating"] = int(round(expected_rating))
    mentions.loc[indices, "expected_rating"] = expected_rating
    mentions.loc[indices, "sentiment_score"] = (expected_rating - 3.0) / 2.0
    mentions.loc[indices, "sentiment_label"] = label
    mentions.loc[indices, "negative_probability"] = probabilities[:2].sum()
    mentions.loc[indices, "neutral_probability"] = probabilities[2]
    mentions.loc[indices, "positive_probability"] = probabilities[3:].sum()
    for class_number, probability in enumerate(probabilities, start=1):
        mentions.loc[indices, f"rating_{class_number}_probability"] = probability
    mentions.loc[indices, "sentiment_method"] = method


# Agregaty badawcze


def collapse_review_aspects(mentions: pd.DataFrame) -> pd.DataFrame:
    """Sprowadza wzmianki do jednej obserwacji opinia-aspekt."""
    if mentions.empty:
        return pd.DataFrame()

    review_aspects = (
        mentions.groupby(
            ["review_id", "id", "aspect", "aspect_group"],
            as_index=False,
        )
        .agg(
            mentions_count=("aspect", "size"),
            expected_rating=("expected_rating", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            negative_probability=("negative_probability", "mean"),
            neutral_probability=("neutral_probability", "mean"),
            positive_probability=("positive_probability", "mean"),
            rule_based_share=(
                "sentiment_method",
                lambda values: float((values != "ordinal_model").mean()),
            ),
            neutral_uncertainty_rule_share=(
                "sentiment_method",
                lambda values: float(
                    values.eq("pore_clogging_uncertainty_rule").mean()
                ),
            ),
        )
    )
    review_aspects["sentiment_label"] = sentiment_labels_from_scores(
        review_aspects["sentiment_score"]
    )
    review_aspects.loc[
        review_aspects["neutral_uncertainty_rule_share"].eq(1),
        "sentiment_label",
    ] = "neutral"
    return review_aspects


def summarize_aspects(mentions: pd.DataFrame) -> pd.DataFrame:
    """Podsumowuje aspekty, nadając każdej opinii równą wagę w danym aspekcie."""
    if mentions.empty:
        return pd.DataFrame()

    review_aspects = collapse_review_aspects(mentions)
    summary = (
        review_aspects.groupby(["aspect", "aspect_group"], as_index=False)
        .agg(
            mentions_count=("mentions_count", "sum"),
            reviews_count=("review_id", "size"),
            products_count=("id", "nunique"),
            mean_expected_rating=("expected_rating", "mean"),
            mean_sentiment_score=("sentiment_score", "mean"),
            median_sentiment_score=("sentiment_score", "median"),
            negative_share=(
                "sentiment_label",
                lambda values: float((values == "negative").mean()),
            ),
            neutral_share=(
                "sentiment_label",
                lambda values: float((values == "neutral").mean()),
            ),
            positive_share=(
                "sentiment_label",
                lambda values: float((values == "positive").mean()),
            ),
            rule_based_share=("rule_based_share", "mean"),
        )
        .sort_values("mentions_count", ascending=False)
    )
    return summary


def summarize_product_aspects(mentions: pd.DataFrame) -> pd.DataFrame:
    """Agreguje sentyment aspektów do produktu po poziomie pojedynczej opinii."""
    if mentions.empty:
        return pd.DataFrame()

    review_aspects = collapse_review_aspects(mentions)
    return (
        review_aspects.groupby(["id", "aspect", "aspect_group"], as_index=False)
        .agg(
            mentions_count=("mentions_count", "sum"),
            reviews_count=("review_id", "size"),
            mean_expected_rating=("expected_rating", "mean"),
            mean_sentiment_score=("sentiment_score", "mean"),
            negative_share=(
                "sentiment_label",
                lambda values: float((values == "negative").mean()),
            ),
            positive_share=(
                "sentiment_label",
                lambda values: float((values == "positive").mean()),
            ),
            rule_based_share=("rule_based_share", "mean"),
        )
        .sort_values(["id", "mentions_count"], ascending=[True, False])
    )


def summarize_products(
    reviews_scored: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Łączy sentyment opinii z przygotowaną bazą produktów."""
    review_summary = (
        reviews_scored.groupby("id", as_index=False)
        .agg(
            reviews_analyzed=("review_id", "size"),
            mean_observed_review_rating=("review_rating", "mean"),
            mean_expected_review_rating=("expected_rating", "mean"),
            mean_sentiment_score=("sentiment_score", "mean"),
            median_sentiment_score=("sentiment_score", "median"),
            sentiment_score_std=("sentiment_score", "std"),
            negative_review_share=(
                "sentiment_label",
                lambda values: float((values == "negative").mean()),
            ),
            neutral_review_share=(
                "sentiment_label",
                lambda values: float((values == "neutral").mean()),
            ),
            positive_review_share=(
                "sentiment_label",
                lambda values: float((values == "positive").mean()),
            ),
        )
    )
    result = products.merge(
        review_summary,
        on="id",
        how="left",
        validate="one_to_one",
    )
    result["reviews_analyzed"] = result["reviews_analyzed"].fillna(0).astype(int)
    return result


def generate_oof_outputs(
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    """Generuje badawcze predykcje dla opinii niewidzianych podczas treningu."""
    review_outputs: list[pd.DataFrame] = []
    aspect_outputs: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, int]] = []

    for fold_number, (train_index, test_index) in enumerate(
        make_oof_splits(reviews),
        start=1,
    ):
        train = reviews.iloc[train_index].copy()
        test = reviews.iloc[test_index].copy()
        train, purged_rows = purge_training_text_overlap(train, test)
        if train.empty:
            raise RuntimeError(f"No training reviews remain in fold {fold_number}")

        model = OrdinalSentimentModel.create().fit(
            train["review_text"],
            train["review_rating"].to_numpy(),
        )
        prediction = model.predict(test["review_text"])
        fold_output = pd.concat(
            [
                test[
                    ["review_id", "id", "review_rating", "review_text"]
                ].reset_index(drop=True),
                prediction,
            ],
            axis=1,
        )
        fold_output.insert(4, "oof_fold", fold_number)
        fold_output["_baseline_rating"] = int(
            train["review_rating"].mode().iloc[0]
        )
        review_outputs.append(fold_output)

        fold_mentions = extract_aspect_mentions(test)
        fold_mentions = score_aspect_mentions(fold_mentions, model)
        if not fold_mentions.empty:
            fold_mentions.insert(3, "oof_fold", fold_number)
            aspect_outputs.append(fold_mentions)

        fold_metadata.append(
            {
                "fold": fold_number,
                "training_reviews": int(len(train)),
                "validation_reviews": int(len(test)),
                "training_products": int(train["id"].nunique()),
                "validation_products": int(test["id"].nunique()),
                "purged_training_text_overlaps": purged_rows,
            }
        )

    reviews_scored = (
        pd.concat(review_outputs, ignore_index=True)
        .sort_values("review_id")
        .reset_index(drop=True)
    )
    if len(reviews_scored) != len(reviews):
        raise RuntimeError("OOF predictions do not cover every review")
    if reviews_scored["review_id"].duplicated().any():
        raise RuntimeError("A review received more than one OOF prediction")

    if aspect_outputs:
        aspect_mentions = (
            pd.concat(aspect_outputs, ignore_index=True)
            .sort_values(["review_id", "fragment_index", "aspect"])
            .reset_index(drop=True)
        )
    else:
        aspect_mentions = pd.DataFrame()
    return reviews_scored, aspect_mentions, fold_metadata


# Trening, predykcja i zapis wyników


def main() -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    reviews, products = load_model_data()
    reviews_scored, aspect_mentions, fold_metadata = generate_oof_outputs(reviews)

    baseline_prediction = reviews_scored.pop("_baseline_rating").to_numpy()
    validation_prediction = reviews_scored.drop(
        columns=["review_id", "id", "review_rating", "review_text", "oof_fold"]
    )
    metrics, rating_confusion, class_metrics = evaluate_predictions(
        reviews_scored["review_rating"].to_numpy(),
        validation_prediction,
        baseline_prediction,
    )

    reviews_scored.to_csv(
        validation_predictions_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    reviews_scored.to_csv(
        reviews_scored_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    metrics.to_csv(
        metrics_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    class_metrics.to_csv(
        class_metrics_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    rating_confusion.to_csv(
        confusion_matrix_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )

    aspect_mentions.to_csv(
        review_aspects_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )

    aspect_summary = summarize_aspects(aspect_mentions)
    aspect_summary.to_csv(
        aspect_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    product_aspect_summary = summarize_product_aspects(aspect_mentions)
    product_aspect_summary.to_csv(
        product_aspect_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    product_summary = summarize_products(reviews_scored, products)
    product_summary.to_csv(
        product_sentiment_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )

    # Model produkcyjny jest trenowany osobno na całości i służy tylko do
    # przyszłych, nowych opinii. Wyniki badawcze powyżej pochodzą wyłącznie z OOF.
    final_model = OrdinalSentimentModel.create().fit(
        reviews["review_text"],
        reviews["review_rating"].to_numpy(),
    )
    joblib.dump(
        {
            "vectorizer": final_model.vectorizer,
            "threshold_models": final_model.threshold_models,
            "thresholds": final_model.thresholds,
            "aspect_definitions": aspect_definitions,
            "random_state": random_state,
            "class_weight_power": class_weight_power,
            "sentiment_negative_score_threshold": (
                sentiment_negative_score_threshold
            ),
            "sentiment_positive_score_threshold": (
                sentiment_positive_score_threshold
            ),
            "training_rows": len(reviews),
        },
        model_path,
    )
    # Weryfikacja pełnego cyklu serializacji. Model finalny nie zasila wyników
    # badawczych (te pochodzą wyłącznie z predykcji OOF), ale pozostaje gotowy
    # do późniejszych predykcji i zgodny z opisem odtwarzalności.
    joblib.load(model_path)

    metadata = {
        "model_type": "ordinal_tfidf_logistic_regression_with_rule_based_aspects",
        "training_reviews": int(len(reviews)),
        "training_products": int(reviews["id"].nunique()),
        "validation_reviews": int(len(reviews)),
        "validation_products": int(reviews["id"].nunique()),
        "oof_folds": oof_folds,
        "random_state": random_state,
        "class_weight_power": class_weight_power,
        "sentiment_negative_score_threshold": (
            sentiment_negative_score_threshold
        ),
        "sentiment_positive_score_threshold": (
            sentiment_positive_score_threshold
        ),
        "validation_split": "stratified_group_oof_by_product",
        "validation_product_ids_file": validation_product_ids_csv.name,
        "purged_training_text_overlaps": int(
            sum(
                fold["purged_training_text_overlaps"]
                for fold in fold_metadata
            )
        ),
        "folds": fold_metadata,
        "rating_distribution": {
            str(key): int(value)
            for key, value in reviews["review_rating"].value_counts().sort_index().items()
        },
        "aspects": {
            aspect: definition["group"]
            for aspect, definition in aspect_definitions.items()
        },
        "aspect_mentions": int(len(aspect_mentions)),
        "reviews_with_aspect": int(aspect_mentions["review_id"].nunique())
        if not aspect_mentions.empty
        else 0,
    }
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Validation metrics saved to {metrics_csv}")
    print(f"Final model saved to {model_path}")
    print(f"Scored {len(reviews_scored)} reviews with grouped OOF models")
    print(
        f"Detected {len(aspect_mentions)} aspect mentions in "
        f"{metadata['reviews_with_aspect']} reviews"
    )


if __name__ == "__main__":
    main()
