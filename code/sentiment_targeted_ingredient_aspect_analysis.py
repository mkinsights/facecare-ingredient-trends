from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sentiment_model import collapse_review_aspects


# Konfiguracja ukierunkowanych hipotez

root = Path(__file__).resolve().parents[1]
mentions_csv = (
    root / "processed" / "sentiment" / "model" / "review_aspect_sentiment.csv"
)
products_csv = root / "processed" / "products_analysis_processed.csv"
output_dir = (
    root / "processed" / "sentiment" / "aspects"
)

results_csv = output_dir / "targeted_ingredient_aspect_results.csv"
diagnostics_csv = output_dir / "targeted_ingredient_aspect_diagnostics.csv"
methodology_csv = output_dir / "targeted_ingredient_aspect_methodology.csv"

csv_float_format = "%.2f"
maximum_review_weight = 10
minimum_category_size = 5

hypotheses = [
    {
        "family": "hydration_ingredients",
        "feature": "is_ceramides",
        "feature_type": "binary",
        "aspect": "hydration_nourishment",
        "expected_direction": "positive",
    },
    {
        "family": "hydration_ingredients",
        "feature": "is_hyaluronic_acid",
        "feature_type": "binary",
        "aspect": "hydration_nourishment",
        "expected_direction": "positive",
    },
    {
        "family": "hydration_ingredients",
        "feature": "is_panthenol",
        "feature_type": "binary",
        "aspect": "hydration_nourishment",
        "expected_direction": "positive",
    },
    {
        "family": "effectiveness_ingredients",
        "feature": "is_retinoids",
        "feature_type": "binary",
        "aspect": "effectiveness",
        "expected_direction": "positive",
    },
    {
        "family": "effectiveness_ingredients",
        "feature": "is_vitamin_c",
        "feature_type": "binary",
        "aspect": "effectiveness",
        "expected_direction": "positive",
    },
    {
        "family": "effectiveness_ingredients",
        "feature": "is_niacinamide",
        "feature_type": "binary",
        "aspect": "effectiveness",
        "expected_direction": "positive",
    },
    {
        "family": "salicylic_acid_performance",
        "feature": "is_salicylic_acid",
        "feature_type": "binary",
        "aspect": "cleansing_makeup_removal",
        "expected_direction": "positive",
    },
    {
        "family": "salicylic_acid_performance",
        "feature": "is_salicylic_acid",
        "feature_type": "binary",
        "aspect": "pore_clogging",
        "expected_direction": "positive",
    },
    {
        "family": "salicylic_acid_performance",
        "feature": "is_salicylic_acid",
        "feature_type": "binary",
        "aspect": "skin_tolerance",
        "expected_direction": "undirected",
    },
    {
        "family": "formula_tolerance_sensory",
        "feature": "is_parfum",
        "feature_type": "binary",
        "aspect": "scent",
        "expected_direction": "undirected",
    },
    {
        "family": "formula_tolerance_sensory",
        "feature": "fragrance_count",
        "feature_type": "continuous",
        "aspect": "scent",
        "expected_direction": "undirected",
    },
    {
        "family": "formula_tolerance_sensory",
        "feature": "is_parfum",
        "feature_type": "binary",
        "aspect": "skin_tolerance",
        "expected_direction": "negative",
    },
    {
        "family": "formula_tolerance_sensory",
        "feature": "fragrance_allergen_count",
        "feature_type": "continuous",
        "aspect": "skin_tolerance",
        "expected_direction": "negative",
    },
    {
        "family": "formula_tolerance_sensory",
        "feature": "is_drying_alcohol",
        "feature_type": "binary",
        "aspect": "skin_tolerance",
        "expected_direction": "negative",
    },
    {
        "family": "formula_tolerance_sensory",
        "feature": "is_drying_alcohol",
        "feature_type": "binary",
        "aspect": "hydration_nourishment",
        "expected_direction": "negative",
    },
    {
        "family": "active_intensity",
        "feature": "active_count",
        "feature_type": "continuous",
        "aspect": "effectiveness",
        "expected_direction": "positive",
    },
]


# Dane na poziomie produktu i aspektu


def require_columns(
    data: pd.DataFrame,
    columns: set[str],
    source_name: str,
) -> None:
    """Sprawdza komplet wymaganych kolumn."""
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Zapisuje CSV zgodnie ze standardem projektu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format=csv_float_format,
    )


def probability_display(value: float) -> str:
    """Zachowuje czytelność bardzo małych wartości p i q."""
    if pd.isna(value):
        return ""
    if value < 0.01:
        return "<0.01"
    return f"{value:.2f}"


def collapse_rare_categories(values: pd.Series) -> pd.Series:
    """Łączy małe category2, ograniczając liczbę niestabilnych parametrów."""
    normalized = values.fillna("__missing__").astype(str)
    counts = normalized.value_counts()
    return normalized.where(
        normalized.map(counts).ge(minimum_category_size),
        "__rare__",
    )


def load_product_aspects() -> pd.DataFrame:
    """Buduje jedną obserwację dla produktu i aspektu."""
    mentions = pd.read_csv(mentions_csv)
    products = pd.read_csv(products_csv)
    if products["id"].duplicated().any():
        raise ValueError("products_analysis_processed contains duplicated ids")

    features = sorted({hypothesis["feature"] for hypothesis in hypotheses})
    metadata = [
        "id",
        "brand",
        "category2",
        "log_standardized_price",
        "log_opinions",
        "ingredients_count",
        *features,
    ]
    require_columns(products, set(metadata), "products_analysis_processed")

    review_aspects = collapse_review_aspects(mentions)
    product_aspects = (
        review_aspects.groupby(["id", "aspect"], as_index=False)
        .agg(
            product_mean_sentiment_score=("sentiment_score", "mean"),
            product_aspect_reviews=("review_id", "size"),
        )
        .merge(
            products[metadata],
            on="id",
            how="left",
            validate="many_to_one",
        )
    )
    if product_aspects[["brand", "category2"]].isna().any().any():
        raise ValueError("Missing product metadata after aspect merge")
    return product_aspects


def support_status(
    products_count: int,
    feature_type: str,
    present_count: int,
    absent_count: int,
    unique_values: int,
) -> str:
    """Oznacza, czy dana hipoteza ma stabilne pokrycie danych."""
    if feature_type == "binary":
        minimum_group = min(present_count, absent_count)
        if products_count >= 200 and minimum_group >= 30:
            return "stable"
        if products_count >= 100 and minimum_group >= 10:
            return "exploratory"
        return "insufficient"
    if products_count >= 200 and unique_values >= 5:
        return "stable"
    if products_count >= 100 and unique_values >= 3:
        return "exploratory"
    return "insufficient"


# Projekt modelu i diagnostyka


def prepare_design(
    product_aspects: pd.DataFrame,
    hypothesis: dict[str, str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Buduje model z kontrolą rynku, category2 i złożoności formuły."""
    feature = hypothesis["feature"]
    aspect = hypothesis["aspect"]
    control_columns = [
        "log_standardized_price",
        "log_opinions",
    ]
    if feature != "active_count":
        control_columns.append("ingredients_count")

    analysis = product_aspects.loc[
        product_aspects["aspect"].eq(aspect),
        [
            "id",
            "brand",
            "category2",
            "product_mean_sentiment_score",
            "product_aspect_reviews",
            feature,
            *control_columns,
        ],
    ].copy()
    numeric_columns = [
        "product_mean_sentiment_score",
        "product_aspect_reviews",
        feature,
        *control_columns,
    ]
    analysis[numeric_columns] = analysis[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna(
        subset=numeric_columns
    )
    analysis["category2"] = collapse_rare_categories(analysis["category2"])
    analysis["brand"] = analysis["brand"].fillna("__missing__").astype(str)

    numeric_design = analysis[[feature, *control_columns]].astype(float).copy()
    continuous_columns = control_columns.copy()
    if hypothesis["feature_type"] == "continuous":
        continuous_columns.insert(0, feature)
    for column in continuous_columns:
        standard_deviation = numeric_design[column].std(ddof=0)
        if standard_deviation <= 0:
            raise ValueError(f"No variation in model column: {column}")
        numeric_design[column] = (
            numeric_design[column] - numeric_design[column].mean()
        ) / standard_deviation

    category_dummies = pd.get_dummies(
        analysis["category2"],
        prefix="category2",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=analysis.index, name="intercept"),
            numeric_design,
            category_dummies,
        ],
        axis=1,
    )
    return (
        analysis,
        analysis["product_mean_sentiment_score"],
        design,
        [feature, *control_columns],
    )


def maximum_vif(design: pd.DataFrame, numeric_columns: list[str]) -> float:
    """Liczy maksymalny VIF dla interpretowanych zmiennych liczbowych."""
    values = sm.add_constant(
        design[numeric_columns].astype(float),
        has_constant="add",
    ).to_numpy()
    return float(
        max(
            variance_inflation_factor(values, index)
            for index in range(1, values.shape[1])
        )
    )


def descriptive_statistics(
    analysis: pd.DataFrame,
    feature: str,
    feature_type: str,
) -> dict[str, float]:
    """Liczy opisowe różnice grup lub korelację dla licznika."""
    values = analysis[feature]
    outcome = analysis["product_mean_sentiment_score"]
    present = values.gt(0)
    statistics = {
        "feature_present_products": int(present.sum()),
        "feature_absent_products": int((~present).sum()),
        "feature_unique_values": int(values.nunique()),
        "mean_sentiment_with_feature": np.nan,
        "mean_sentiment_without_feature": np.nan,
        "raw_mean_difference": np.nan,
        "raw_spearman_rho": np.nan,
        "raw_spearman_p_value": np.nan,
    }
    if feature_type == "binary":
        with_feature = outcome.loc[present]
        without_feature = outcome.loc[~present]
        statistics["mean_sentiment_with_feature"] = float(with_feature.mean())
        statistics["mean_sentiment_without_feature"] = float(
            without_feature.mean()
        )
        statistics["raw_mean_difference"] = float(
            with_feature.mean() - without_feature.mean()
        )
    else:
        rho, p_value = spearmanr(values, outcome)
        statistics["raw_spearman_rho"] = float(rho)
        statistics["raw_spearman_p_value"] = float(p_value)
    return statistics


# Estymacja hipotez i analiza wrażliwości


def fit_hypothesis(
    product_aspects: pd.DataFrame,
    hypothesis: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Dopasowuje główny OLS i ważony model wrażliwości."""
    analysis, outcome, design, numeric_columns = prepare_design(
        product_aspects,
        hypothesis,
    )
    feature = hypothesis["feature"]
    matrix = design.to_numpy(dtype=float)
    design_rank = int(np.linalg.matrix_rank(matrix))
    if design_rank < design.shape[1]:
        raise ValueError(
            f"Rank-deficient design for {feature} -> {hypothesis['aspect']}"
        )

    cluster_options = {
        "groups": analysis["brand"],
        "use_correction": True,
    }
    primary = sm.OLS(outcome, design).fit(
        cov_type="cluster",
        cov_kwds=cluster_options,
        use_t=True,
    )
    weights = analysis["product_aspect_reviews"].clip(
        lower=1,
        upper=maximum_review_weight,
    )
    sensitivity = sm.WLS(outcome, design, weights=weights).fit(
        cov_type="cluster",
        cov_kwds=cluster_options,
        use_t=True,
    )

    primary_confidence = primary.conf_int().loc[feature]
    sensitivity_confidence = sensitivity.conf_int().loc[feature]
    descriptive = descriptive_statistics(
        analysis,
        feature,
        hypothesis["feature_type"],
    )
    status = support_status(
        len(analysis),
        hypothesis["feature_type"],
        descriptive["feature_present_products"],
        descriptive["feature_absent_products"],
        descriptive["feature_unique_values"],
    )
    result = {
        **hypothesis,
        "support_status": status,
        "products_count": len(analysis),
        "brands_count": analysis["brand"].nunique(),
        "category2_count": analysis["category2"].nunique(),
        **descriptive,
        "primary_coefficient": float(primary.params[feature]),
        "primary_standard_error": float(primary.bse[feature]),
        "primary_ci_95_lower": float(primary_confidence.iloc[0]),
        "primary_ci_95_upper": float(primary_confidence.iloc[1]),
        "primary_p_value": float(primary.pvalues[feature]),
        "sensitivity_coefficient": float(sensitivity.params[feature]),
        "sensitivity_standard_error": float(sensitivity.bse[feature]),
        "sensitivity_ci_95_lower": float(sensitivity_confidence.iloc[0]),
        "sensitivity_ci_95_upper": float(sensitivity_confidence.iloc[1]),
        "sensitivity_p_value": float(sensitivity.pvalues[feature]),
    }
    diagnostics = {
        "family": hypothesis["family"],
        "feature": feature,
        "aspect": hypothesis["aspect"],
        "products_count": len(analysis),
        "design_columns": design.shape[1],
        "design_rank": design_rank,
        "condition_number": float(np.linalg.cond(matrix)),
        "maximum_numeric_vif": maximum_vif(design, numeric_columns),
        "primary_r_squared": float(primary.rsquared),
        "primary_adjusted_r_squared": float(primary.rsquared_adj),
        "sensitivity_r_squared": float(sensitivity.rsquared),
        "sensitivity_adjusted_r_squared": float(
            sensitivity.rsquared_adj
        ),
    }
    return result, diagnostics


def add_inference_flags(results: pd.DataFrame) -> pd.DataFrame:
    """Dodaje FDR, zgodność kierunku i kryterium odpornego wyniku."""
    results = results.copy()
    results["q_value_fdr_bh"] = np.nan
    for indices in results.groupby("family").groups.values():
        index = list(indices)
        results.loc[index, "q_value_fdr_bh"] = multipletests(
            results.loc[index, "primary_p_value"],
            method="fdr_bh",
        )[1]

    results["direction_consistent"] = (
        np.sign(results["primary_coefficient"])
        == np.sign(results["sensitivity_coefficient"])
    )
    results["direction_matches_hypothesis"] = (
        results["expected_direction"].eq("undirected")
        | (
            results["expected_direction"].eq("positive")
            & results["primary_coefficient"].gt(0)
        )
        | (
            results["expected_direction"].eq("negative")
            & results["primary_coefficient"].lt(0)
        )
    )
    results["sensitivity_ci_excludes_zero"] = (
        results["sensitivity_ci_95_lower"].gt(0)
        | results["sensitivity_ci_95_upper"].lt(0)
    )
    results["robust_result"] = (
        results["support_status"].ne("insufficient")
        & results["q_value_fdr_bh"].le(0.05)
        & results["direction_consistent"]
        & results["direction_matches_hypothesis"]
        & results["sensitivity_ci_excludes_zero"]
    )
    results["primary_p_value_display"] = results["primary_p_value"].map(
        probability_display
    )
    results["q_value_fdr_bh_display"] = results["q_value_fdr_bh"].map(
        probability_display
    )
    return results.sort_values(
        ["robust_result", "family", "q_value_fdr_bh", "feature", "aspect"],
        ascending=[False, True, True, True, True],
        ignore_index=True,
    )


# Uruchomienie



def main() -> None:
    """Uruchamia wszystkie z góry zdefiniowane hipotezy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    product_aspects = load_product_aspects()

    result_rows = []
    diagnostic_rows = []
    for hypothesis in hypotheses:
        result, diagnostics = fit_hypothesis(product_aspects, hypothesis)
        result_rows.append(result)
        diagnostic_rows.append(diagnostics)

    results = add_inference_flags(pd.DataFrame(result_rows))
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["family", "feature", "aspect"],
        ignore_index=True,
    )
    save_csv(results, results_csv)
    save_csv(diagnostics, diagnostics_csv)

    methodology = pd.DataFrame(
        [
            ("observation_unit", "one product-aspect with equal product weight"),
            ("outcome", "product mean OOF sentiment_score for the selected aspect"),
            (
                "primary_model",
                "OLS with category2 fixed effects and brand-clustered errors",
            ),
            (
                "controls",
                "log price, log opinions and ingredients_count; active_count omits ingredients_count due collinearity",
            ),
            (
                "continuous_effect",
                "coefficient for a one-standard-deviation increase",
            ),
            ("binary_effect", "presence versus absence of the feature"),
            (
                "sensitivity",
                "WLS by aspect review count capped at 10; brand-clustered errors",
            ),
            (
                "multiple_testing",
                "Benjamini-Hochberg FDR within each prespecified hypothesis family",
            ),
            (
                "dryness_measurement",
                "proxied by hydration_nourishment and skin_tolerance; no separate dryness aspect exists",
            ),
            (
                "causal_interpretation",
                "not permitted; estimates are adjusted associations",
            ),
        ],
        columns=["parameter", "value"],
    )
    save_csv(methodology, methodology_csv)

    print(f"Tested hypotheses: {len(results)}")
    print(f"Stable hypotheses: {int(results['support_status'].eq('stable').sum())}")
    print(f"Robust results: {int(results['robust_result'].sum())}")
    print(f"Saved targeted analysis to {output_dir}")


if __name__ == "__main__":
    main()
