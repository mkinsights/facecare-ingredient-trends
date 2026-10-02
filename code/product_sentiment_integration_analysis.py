from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Konfiguracja analizy zintegrowanej

root = Path(__file__).resolve().parents[1]
products_csv = root / "processed" / "products_analysis_processed.csv"
sentiment_csv = (
    root / "processed" / "sentiment" / "model" / "product_sentiment_summary.csv"
)
segmentation_root = root / "processed" / "segmentation"
output_dir = root / "processed" / "sentiment" / "product"

coverage_csv = output_dir / "sentiment_product_coverage.csv"
product_correlation_csv = output_dir / "sentiment_product_correlation.csv"
product_correlation_long_csv = (
    output_dir / "sentiment_product_correlation_with_score.csv"
)
ingredient_correlation_csv = (
    output_dir / "sentiment_ingredient_family_correlation.csv"
)
regression_coefficients_csv = output_dir / "sentiment_regression_coefficients.csv"
regression_models_csv = output_dir / "sentiment_regression_model_summary.csv"
segment_tests_csv = output_dir / "segment_sentiment_omnibus_tests.csv"
methodology_csv = output_dir / "product_sentiment_integration_methodology.csv"

csv_float_format = "%.2f"
maximum_review_weight = 10
minimum_category_size = 5

active_flags = [
    "is_niacinamide",
    "is_ceramides",
    "is_hyaluronic_acid",
    "is_retinoids",
    "is_vitamin_c",
    "is_peptides",
    "is_centella_asiatica",
    "is_salicylic_acid",
    "is_panthenol",
    "is_squalane",
    "is_aha",
]

continuous_formula_features = [
    "log_standardized_price",
    "log_opinions",
    "ingredients_count",
    "active_share",
    "fragrance_allergen_count",
    "preservative_count",
    "drying_alcohol_count",
]

binary_formula_features = [
    "is_parfum",
    "is_colorant",
]

regression_specs = {
    "market": [
        "log_standardized_price",
        "log_opinions",
    ],
    "formula": continuous_formula_features + binary_formula_features,
    "ingredient_families": (
        continuous_formula_features + binary_formula_features + active_flags
    ),
    "active_position_sensitivity": [
        "log_standardized_price",
        "log_opinions",
        "active_position_score",
        "fragrance_allergen_count",
        "preservative_count",
        "drying_alcohol_count",
        *binary_formula_features,
    ],
    "fragrance_count_sensitivity": [
        "log_standardized_price",
        "log_opinions",
        "ingredients_count",
        "active_share",
        "fragrance_count",
        "preservative_count",
        "drying_alcohol_count",
        *binary_formula_features,
    ],
    "active_count_sensitivity": [
        "log_standardized_price",
        "log_opinions",
        "active_count",
        "fragrance_allergen_count",
        "preservative_count",
        "drying_alcohol_count",
        *binary_formula_features,
    ],
}

segmentation_scopes = {
    "global": segmentation_root / "segmentation_results.csv",
    "pielegnacja_twarzy": (
        segmentation_root / "pielegnacja_twarzy" / "segmentation_results.csv"
    ),
    "oczyszczanie_i_demakijaz_twarzy": (
        segmentation_root
        / "oczyszczanie_i_demakijaz_twarzy"
        / "segmentation_results.csv"
    ),
}

# Walidacja i przygotowanie danych


def require_columns(
    data: pd.DataFrame,
    columns: set[str],
    source_name: str,
) -> None:
    """Przerywa analizę, jeśli źródło nie zawiera wymaganych kolumn."""
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")


def save_csv(data: pd.DataFrame, path: Path) -> None:
    """Zapisuje CSV w jednolitym formacie liczbowym projektu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format=csv_float_format,
    )


def probability_display(value: float) -> str:
    """Chroni małe wartości p przed nieczytelnym zapisem 0,00."""
    if pd.isna(value):
        return ""
    if value < 0.01:
        return "<0.01"
    return f"{value:.2f}"


def add_fdr(
    data: pd.DataFrame,
    p_value_column: str,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Dodaje korektę Benjamini-Hochberga do rodziny porównań."""
    result = data.copy()
    result["q_value_fdr_bh"] = np.nan
    groups = (
        result.groupby(group_column).groups
        if group_column is not None
        else {"all": result.index}
    )
    for indices in groups.values():
        index = list(indices)
        valid = result.loc[index, p_value_column].notna()
        valid_index = result.loc[index].index[valid]
        if len(valid_index):
            result.loc[valid_index, "q_value_fdr_bh"] = multipletests(
                result.loc[valid_index, p_value_column],
                method="fdr_bh",
            )[1]
    result["p_value_display"] = result[p_value_column].map(probability_display)
    result["q_value_fdr_bh_display"] = result["q_value_fdr_bh"].map(
        probability_display
    )
    result["fdr_significant_0_05"] = result["q_value_fdr_bh"].le(0.05)
    return result


def load_integrated_products() -> pd.DataFrame:
    """Łączy aktualne cechy produktów z wynikiem sentymentu OOF."""
    products = pd.read_csv(products_csv)
    sentiment = pd.read_csv(sentiment_csv)
    for source_name, data in [
        ("products_analysis_processed", products),
        ("product_sentiment_summary", sentiment),
    ]:
        require_columns(data, {"id"}, source_name)
        if data["id"].duplicated().any():
            duplicated = int(data["id"].duplicated(keep=False).sum())
            raise ValueError(f"{source_name} contains {duplicated} duplicated ids")

    sentiment_columns = [
        "id",
        "reviews_analyzed",
        "mean_observed_review_rating",
        "mean_expected_review_rating",
        "mean_sentiment_score",
        "median_sentiment_score",
        "sentiment_score_std",
        "negative_review_share",
        "neutral_review_share",
        "positive_review_share",
    ]
    require_columns(sentiment, set(sentiment_columns), "product_sentiment_summary")
    integrated = products.merge(
        sentiment[sentiment_columns],
        on="id",
        how="left",
        validate="one_to_one",
    )
    integrated = integrated.rename(
        columns={"mean_sentiment_score": "sentiment_score"}
    )
    integrated["has_sentiment_score"] = (
        integrated["sentiment_score"].notna().astype(int)
    )

    return integrated


def build_coverage(products: pd.DataFrame) -> pd.DataFrame:
    """Pokazuje pokrycie wynikiem sentymentu globalnie i według category1."""
    rows = []
    groups = [("all_products", products)]
    groups.extend(
        (str(category), group)
        for category, group in products.groupby("category1", dropna=False)
    )
    for category, group in groups:
        scored = group["sentiment_score"].notna()
        rows.append(
            {
                "category1": category,
                "products_total": len(group),
                "products_with_sentiment": int(scored.sum()),
                "sentiment_coverage_percent": float(scored.mean() * 100),
                "median_reviews_analyzed": float(
                    group.loc[scored, "reviews_analyzed"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["products_total", "category1"],
        ascending=[False, True],
        ignore_index=True,
    )


# Korelacje z główną miarą sentymentu


def build_correlation(
    products: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Liczy macierz Spearmana na produktach z dostępnym sentymentem."""
    require_columns(products, set(columns), "integrated products")
    numeric = products[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman").round(4)


def build_score_correlations_long(
    products: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Testuje każdą korelację ze score i stosuje korektę wielokrotną."""
    rows = []
    for variable in columns:
        if variable == "sentiment_score":
            continue
        pair = products[["sentiment_score", variable]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(pair) < 3 or pair[variable].nunique() < 2:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(
                pair["sentiment_score"],
                pair[variable],
            )
        rows.append(
            {
                "variable": variable,
                "products_count": len(pair),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    result = add_fdr(pd.DataFrame(rows), "p_value")
    return result.sort_values(
        ["q_value_fdr_bh", "variable"],
        na_position="last",
        ignore_index=True,
    )



def save_scope_correlations(products: pd.DataFrame) -> None:
    """Powtarza główną macierz dla dwóch wcześniej analizowanych category1."""
    scopes = {
        "pielegnacja_twarzy": "Pielęgnacja twarzy",
        "oczyszczanie_i_demakijaz_twarzy": (
            "Oczyszczanie i demakijaż twarzy"
        ),
    }
    columns = [
        "sentiment_score",
        "weighted_rating",
        "log_opinions",
        "log_standardized_price",
        "ingredients_count",
        "active_count",
        "active_position_score",
        "fragrance_count",
        "fragrance_allergen_count",
        "preservative_count",
        "drying_alcohol_count",
        "is_parfum",
        "is_colorant",
    ]
    for slug, category in scopes.items():
        scoped = products.loc[products["category1"].eq(category)]
        matrix = build_correlation(scoped, columns)
        matrix.to_csv(
            output_dir / f"{slug}_sentiment_product_correlation.csv",
            index=True,
            encoding="utf-8-sig",
            float_format=csv_float_format,
        )


# Regresja: cechy produktu a sentiment_score


def collapse_rare_categories(values: pd.Series) -> pd.Series:
    """Łączy małe category2, aby nie tworzyć niestabilnych efektów stałych."""
    normalized = values.fillna("__missing__").astype(str)
    counts = normalized.value_counts()
    return normalized.where(counts.reindex(normalized).to_numpy() >= minimum_category_size, "__rare__")


def prepare_regression_design(
    products: pd.DataFrame,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Standaryzuje liczniki, zachowuje flagi 0/1 i dodaje efekty category2."""
    required = {
        "sentiment_score",
        "category2",
        "brand",
        "reviews_analyzed",
        *predictors,
    }
    require_columns(products, required, "integrated products")
    analysis = products[
        [
            "sentiment_score",
            "category2",
            "brand",
            "reviews_analyzed",
            *predictors,
        ]
    ].copy()
    numeric_columns = ["sentiment_score", "reviews_analyzed", *predictors]
    analysis[numeric_columns] = analysis[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna(
        subset=numeric_columns
    )
    analysis["category2"] = collapse_rare_categories(analysis["category2"])
    analysis["brand"] = analysis["brand"].fillna("__missing__").astype(str)

    variable_predictors = [
        predictor
        for predictor in predictors
        if analysis[predictor].nunique(dropna=True) > 1
    ]
    continuous = [
        predictor
        for predictor in variable_predictors
        if predictor not in binary_formula_features + active_flags
    ]
    standardized = analysis[variable_predictors].astype(float).copy()
    for column in continuous:
        standard_deviation = standardized[column].std(ddof=0)
        if standard_deviation > 0:
            standardized[column] = (
                standardized[column] - standardized[column].mean()
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
            standardized,
            category_dummies,
        ],
        axis=1,
    )
    return analysis, analysis["sentiment_score"], design, variable_predictors


def maximum_vif(
    design: pd.DataFrame,
    predictors: list[str],
) -> float:
    """Liczy maksymalny VIF wśród interpretowanych predyktorów."""
    if len(predictors) <= 1:
        return 1.0
    predictor_design = sm.add_constant(
        design[predictors].astype(float),
        has_constant="add",
    )
    values = predictor_design.to_numpy()
    return float(
        max(
            variance_inflation_factor(values, index)
            for index in range(1, values.shape[1])
        )
    )


def extract_regression_rows(
    fitted: object,
    predictors: list[str],
    prefix: str,
) -> dict[str, dict[str, float]]:
    """Wyciąga współczynniki i przedziały ufności z dopasowanego modelu."""
    confidence = fitted.conf_int()
    return {
        predictor: {
            f"{prefix}_coefficient": float(fitted.params[predictor]),
            f"{prefix}_standard_error": float(fitted.bse[predictor]),
            f"{prefix}_ci_95_lower": float(confidence.loc[predictor].iloc[0]),
            f"{prefix}_ci_95_upper": float(confidence.loc[predictor].iloc[1]),
            f"{prefix}_p_value": float(fitted.pvalues[predictor]),
        }
        for predictor in predictors
    }


def fit_sentiment_regressions(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dopasowuje modele OLS oraz ważoną analizę wrażliwości."""
    coefficient_rows = []
    model_rows = []
    for model_name, predictors in regression_specs.items():
        analysis, outcome, design, used_predictors = prepare_regression_design(
            products,
            predictors,
        )
        matrix = design.to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(matrix))
        if rank < design.shape[1]:
            raise ValueError(
                f"Regression {model_name} is rank deficient: "
                f"{rank}/{design.shape[1]}"
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
        weights = analysis["reviews_analyzed"].clip(
            lower=1,
            upper=maximum_review_weight,
        )
        sensitivity = sm.WLS(outcome, design, weights=weights).fit(
            cov_type="cluster",
            cov_kwds=cluster_options,
            use_t=True,
        )
        primary_values = extract_regression_rows(
            primary,
            used_predictors,
            "primary",
        )
        sensitivity_values = extract_regression_rows(
            sensitivity,
            used_predictors,
            "sensitivity",
        )
        for predictor in used_predictors:
            row = {
                "model": model_name,
                "predictor": predictor,
                "predictor_type": (
                    "binary"
                    if predictor in binary_formula_features + active_flags
                    else "continuous_standardized"
                ),
                "products_count": len(analysis),
            }
            row.update(primary_values[predictor])
            row.update(sensitivity_values[predictor])
            row["direction_consistent"] = bool(
                np.sign(row["primary_coefficient"])
                == np.sign(row["sensitivity_coefficient"])
            )
            coefficient_rows.append(row)

        model_rows.append(
            {
                "model": model_name,
                "products_count": len(analysis),
                "brands_count": analysis["brand"].nunique(),
                "category2_count": analysis["category2"].nunique(),
                "predictors_count": len(used_predictors),
                "design_columns": design.shape[1],
                "design_rank": rank,
                "condition_number": float(np.linalg.cond(matrix)),
                "maximum_predictor_vif": maximum_vif(
                    design,
                    used_predictors,
                ),
                "primary_r_squared": float(primary.rsquared),
                "primary_adjusted_r_squared": float(primary.rsquared_adj),
                "primary_aic": float(primary.aic),
                "primary_bic": float(primary.bic),
                "sensitivity_r_squared": float(sensitivity.rsquared),
                "sensitivity_adjusted_r_squared": float(
                    sensitivity.rsquared_adj
                ),
            }
        )

    coefficients = add_fdr(
        pd.DataFrame(coefficient_rows),
        "primary_p_value",
        "model",
    )
    coefficients["sensitivity_ci_excludes_zero"] = (
        coefficients["sensitivity_ci_95_lower"].gt(0)
        | coefficients["sensitivity_ci_95_upper"].lt(0)
    )
    coefficients["robust_result"] = (
        coefficients["fdr_significant_0_05"]
        & coefficients["direction_consistent"]
        & coefficients["sensitivity_ci_excludes_zero"]
    )
    coefficients = coefficients.sort_values(
        ["model", "q_value_fdr_bh", "predictor"],
        na_position="last",
        ignore_index=True,
    )
    return coefficients, pd.DataFrame(model_rows)



def build_segment_diagnostics(products: pd.DataFrame) -> pd.DataFrame:
    """Testuje eksploracyjnie różnice wydźwięku między segmentami K-means."""
    rows = []
    for scope, path in segmentation_scopes.items():
        segments = pd.read_csv(path, usecols=["id", "segment"])
        if segments["id"].duplicated().any():
            raise ValueError(f"{scope} segmentation contains duplicated ids")
        data = segments.merge(
            products[["id", "sentiment_score"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        groups = [
            group["sentiment_score"].dropna().to_numpy()
            for _, group in data.groupby("segment")
        ]
        groups = [values for values in groups if len(values)]
        if len(groups) < 2:
            continue
        statistic, p_value = kruskal(*groups)
        products_count = sum(len(values) for values in groups)
        groups_count = len(groups)
        rows.append(
            {
                "scope": scope,
                "test": "Kruskal-Wallis",
                "products_count": products_count,
                "segments_count": groups_count,
                "statistic": statistic,
                "p_value": p_value,
                "epsilon_squared": max(
                    (statistic - groups_count + 1)
                    / (products_count - groups_count),
                    0,
                ),
                "interpretation": "exploratory_not_used_for_main_conclusions",
            }
        )
    return add_fdr(pd.DataFrame(rows), "p_value")


# Uruchomienie pełnej integracji


def main() -> None:
    """Uruchamia korelacje i regresję wydźwięku na poziomie produktu."""
    output_dir.mkdir(parents=True, exist_ok=True)
    products = load_integrated_products()
    scored_products = products.loc[
        products["sentiment_score"].notna()
    ].copy()

    product_columns = [
        "sentiment_score",
        "weighted_rating",
        "log_opinions",
        "log_standardized_price",
        "ingredients_count",
        "active_count",
        "active_share",
        "active_position_score",
        "fragrance_count",
        "fragrance_allergen_count",
        "preservative_count",
        "drying_alcohol_count",
        "is_parfum",
        "is_colorant",
    ]
    ingredient_columns = [
        "sentiment_score",
        "weighted_rating",
        "log_opinions",
        "log_standardized_price",
        "ingredients_count",
        "active_count",
        "active_position_score",
        *active_flags,
    ]
    product_correlation = build_correlation(scored_products, product_columns)
    ingredient_correlation = build_correlation(
        scored_products,
        ingredient_columns,
    )
    score_correlations = build_score_correlations_long(
        scored_products,
        product_columns,
    )
    coefficients, model_summary = fit_sentiment_regressions(scored_products)
    segment_diagnostics = build_segment_diagnostics(products)
    coverage = build_coverage(products)

    save_csv(coverage, coverage_csv)
    product_correlation.to_csv(
        product_correlation_csv,
        index=True,
        encoding="utf-8-sig",
        float_format=csv_float_format,
    )
    ingredient_correlation.to_csv(
        ingredient_correlation_csv,
        index=True,
        encoding="utf-8-sig",
        float_format=csv_float_format,
    )
    save_csv(score_correlations, product_correlation_long_csv)
    save_csv(coefficients, regression_coefficients_csv)
    save_csv(model_summary, regression_models_csv)
    save_csv(segment_diagnostics, segment_tests_csv)
    save_scope_correlations(scored_products)


    methodology = pd.DataFrame(
        [
            ("primary_outcome", "sentiment_score"),
            (
                "sentiment_definition",
                "mean OOF review-level sentiment_score for one product",
            ),
            ("primary_observation_unit", "one product with equal weight"),
            (
                "correlation",
                "Spearman; weighted_rating used only for convergent validation",
            ),
            (
                "primary_regression",
                "OLS with category2 fixed effects and brand-clustered errors",
            ),
            (
                "regression_scaling",
                "continuous predictors z-standardized; binary predictors kept 0/1",
            ),
            (
                "collinearity_control",
                "active_share is primary; active_count, position and broad fragrance count use separate sensitivity models",
            ),
            (
                "sensitivity_regression",
                "WLS by reviews_analyzed capped at 10; brand-clustered errors",
            ),
            (
                "multiple_testing",
                "Benjamini-Hochberg FDR within each model",
            ),
            (
                "segment_diagnostic",
                "exploratory Kruskal-Wallis tests; omitted from main conclusions because silhouette was low",
            ),
            (
                "interpretation",
                "associational, not causal; sentiment availability is reported",
            ),
        ],
        columns=["parameter", "value"],
    )
    save_csv(methodology, methodology_csv)

    print(
        f"Integrated {len(products)} products; "
        f"{len(scored_products)} have sentiment_score."
    )
    print(
        f"Robust full-model coefficients: "
        f"{int(coefficients.loc[coefficients['model'].eq('ingredient_families'), 'robust_result'].sum())}"
    )
    print(f"Saved product-sentiment integration to {output_dir}")


if __name__ == "__main__":
    main()
