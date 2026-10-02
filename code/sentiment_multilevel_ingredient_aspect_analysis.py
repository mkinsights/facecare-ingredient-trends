from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from sentiment_model import collapse_review_aspects
from sentiment_targeted_ingredient_aspect_analysis import (
    hypotheses,
    probability_display,
    save_csv,
)


# Konfiguracja modelu wielopoziomowego

root = Path(__file__).resolve().parents[1]
mentions_csv = (
    root / "processed" / "sentiment" / "model" / "review_aspect_sentiment.csv"
)
products_csv = root / "processed" / "products_analysis_processed.csv"
product_level_results_csv = (
    root
    / "processed"
    / "sentiment"
    / "aspects"
    / "targeted_ingredient_aspect_results.csv"
)
output_dir = (
    root / "processed" / "sentiment" / "multilevel"
)

pooled_effects_csv = output_dir / "multilevel_pooled_effects.csv"
category_effects_csv = output_dir / "multilevel_category_effects.csv"
combination_results_csv = output_dir / "multilevel_combination_results.csv"
interaction_tests_csv = output_dir / "multilevel_interaction_tests.csv"
diagnostics_csv = output_dir / "multilevel_model_diagnostics.csv"
methodology_csv = output_dir / "multilevel_methodology.csv"

minimum_binary_products_per_group = 10
minimum_continuous_category_products = 30
minimum_continuous_unique_values = 3
normal_critical_value = norm.ppf(0.975)


# Przygotowanie poziomu opinia–aspekt


def require_columns(
    data: pd.DataFrame,
    columns: set[str],
    source_name: str,
) -> None:
    """Sprawdza komplet kolumn wymaganych przez model."""
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")


def load_review_aspects() -> pd.DataFrame:
    """Łączy sentyment aspektowy opinii z aktualnymi cechami produktów."""
    mentions = pd.read_csv(mentions_csv)
    products = pd.read_csv(products_csv)
    if products["id"].duplicated().any():
        raise ValueError("products_analysis_processed contains duplicated ids")

    features = sorted({hypothesis["feature"] for hypothesis in hypotheses})
    metadata_columns = [
        "id",
        "brand",
        "category2",
        "log_standardized_price",
        "log_opinions",
        "ingredients_count",
        *features,
    ]
    require_columns(products, set(metadata_columns), "processed products")

    review_aspects = collapse_review_aspects(mentions)
    data = review_aspects.merge(
        products[metadata_columns],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if data[["brand", "category2"]].isna().any().any():
        raise ValueError("Missing product metadata after review-aspect merge")
    return data


def supported_interaction_categories(
    data: pd.DataFrame,
    feature: str,
    feature_type: str,
) -> list[str]:
    """Wybiera category2 z wystarczającym wsparciem dla interakcji."""
    products = data.drop_duplicates("id")
    supported = []
    for category, group in products.groupby("category2"):
        values = group[feature]
        if feature_type == "binary":
            present = int(values.gt(0).sum())
            absent = int(values.eq(0).sum())
            eligible = (
                present >= minimum_binary_products_per_group
                and absent >= minimum_binary_products_per_group
            )
        else:
            eligible = (
                len(group) >= minimum_continuous_category_products
                and values.nunique() >= minimum_continuous_unique_values
            )
        if eligible:
            supported.append(str(category))
    return supported


def prepare_model_data(
    review_aspects: pd.DataFrame,
    hypothesis: dict[str, str],
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    """Standaryzuje kontrolki i koduje wspierane poziomy category2."""
    feature = hypothesis["feature"]
    aspect = hypothesis["aspect"]
    controls = ["log_standardized_price", "log_opinions"]
    if feature != "active_count":
        controls.append("ingredients_count")

    columns = [
        "review_id",
        "id",
        "brand",
        "category2",
        "sentiment_score",
        feature,
        *controls,
    ]
    data = review_aspects.loc[
        review_aspects["aspect"].eq(aspect),
        columns,
    ].copy()
    numeric_columns = ["sentiment_score", feature, *controls]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=numeric_columns
    )

    supported = supported_interaction_categories(
        data,
        feature,
        hypothesis["feature_type"],
    )
    data["category_model"] = data["category2"].where(
        data["category2"].isin(supported),
        "__other__",
    )
    product_counts = (
        data.drop_duplicates("id")["category_model"].value_counts()
    )
    supported_reference = [
        category for category in supported if category in product_counts.index
    ]
    if supported_reference:
        reference = max(
            supported_reference,
            key=lambda category: int(product_counts.loc[category]),
        )
    else:
        reference = str(product_counts.index[0])
    category_levels = [
        reference,
        *sorted(
            category
            for category in data["category_model"].unique()
            if category != reference
        ),
    ]
    data["category_model"] = pd.Categorical(
        data["category_model"],
        categories=category_levels,
        ordered=True,
    )

    continuous_columns = controls.copy()
    if hypothesis["feature_type"] == "continuous":
        continuous_columns.insert(0, feature)
    model_feature = feature
    for column in continuous_columns:
        standard_deviation = data[column].std(ddof=0)
        if standard_deviation <= 0:
            raise ValueError(f"No variation in model column: {column}")
        standardized_name = f"{column}_z"
        data[standardized_name] = (
            data[column] - data[column].mean()
        ) / standard_deviation
        if column == feature:
            model_feature = standardized_name

    data["product_id"] = data["id"].astype(str)
    data["brand_group"] = data["brand"].fillna("__missing__").astype(str)
    standardized_controls = [f"{column}_z" for column in controls]
    return data, supported, standardized_controls, model_feature


# Estymacja MixedLM


def fit_mixed_model(
    formula: str,
    data: pd.DataFrame,
) -> tuple[object, str, list[str]]:
    """Dopasowuje MixedLM i stosuje drugi optymalizator przy braku zbieżności."""
    captured_warnings: list[str] = []
    last_error: Exception | None = None
    for optimizer in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data,
                    groups=data["brand_group"],
                    re_formula="1",
                    vc_formula={"product": "0 + C(product_id)"},
                )
                result = model.fit(
                    reml=False,
                    method=optimizer,
                    maxiter=500,
                    disp=False,
                )
            captured_warnings.extend(
                str(item.message) for item in caught
            )
            if result.converged:
                return result, optimizer, captured_warnings
        except Exception as error:
            last_error = error
            captured_warnings.append(f"{optimizer}: {type(error).__name__}: {error}")
    if last_error is not None:
        raise RuntimeError(
            "MixedLM failed: " + " | ".join(captured_warnings)
        ) from last_error
    raise RuntimeError("MixedLM did not converge: " + " | ".join(captured_warnings))


def fixed_effect_covariance(result: object) -> pd.DataFrame:
    """Zwraca macierz kowariancji ograniczoną do efektów stałych."""
    fixed_names = list(result.fe_params.index)
    return result.cov_params().loc[fixed_names, fixed_names]


def linear_contrast(
    result: object,
    contrast: pd.Series,
) -> dict[str, float]:
    """Liczy efekt, błąd, CI i p-value dla kombinacji parametrów stałych."""
    fixed_parameters = result.fe_params
    covariance = fixed_effect_covariance(result)
    vector = contrast.reindex(fixed_parameters.index, fill_value=0).to_numpy()
    coefficient = float(vector @ fixed_parameters.to_numpy())
    variance = float(vector @ covariance.to_numpy() @ vector)
    standard_error = float(np.sqrt(max(variance, 0)))
    z_value = (
        coefficient / standard_error if standard_error > 0 else np.nan
    )
    p_value = (
        float(2 * norm.sf(abs(z_value))) if not pd.isna(z_value) else np.nan
    )
    return {
        "coefficient": coefficient,
        "standard_error": standard_error,
        "ci_95_lower": coefficient - normal_critical_value * standard_error,
        "ci_95_upper": coefficient + normal_critical_value * standard_error,
        "z_value": z_value,
        "p_value": p_value,
    }


def interaction_parameter(
    parameter_names: list[str],
    model_feature: str,
    category: str,
) -> str | None:
    """Odszukuje nazwę parametru interakcji dla konkretnej kategorii."""
    prefix = f"{model_feature}:C(category_model)"
    suffix = f"[T.{category}]"
    return next(
        (
            name
            for name in parameter_names
            if name.startswith(prefix) and name.endswith(suffix)
        ),
        None,
    )


def category_slope(
    result: object,
    model_feature: str,
    category: str,
    reference: str,
) -> dict[str, float]:
    """Wyznacza efekt składnika w wybranym category2."""
    contrast = pd.Series(0.0, index=result.fe_params.index)
    contrast[model_feature] = 1.0
    if category != reference:
        interaction = interaction_parameter(
            list(result.fe_params.index),
            model_feature,
            category,
        )
        if interaction is None:
            raise ValueError(
                f"Missing interaction parameter for category: {category}"
            )
        contrast[interaction] = 1.0
    return linear_contrast(result, contrast)


def average_category_slope(
    result: object,
    model_feature: str,
    reference: str,
    product_weights: pd.Series,
) -> dict[str, float]:
    """Liczy uśredniony efekt, ważąc kategorie liczbą produktów."""
    contrast = pd.Series(0.0, index=result.fe_params.index)
    contrast[model_feature] = 1.0
    normalized_weights = product_weights / product_weights.sum()
    for category, weight in normalized_weights.items():
        if category == reference:
            continue
        interaction = interaction_parameter(
            list(result.fe_params.index),
            model_feature,
            str(category),
        )
        if interaction is not None:
            contrast[interaction] = float(weight)
    return linear_contrast(result, contrast)


def interaction_wald_test(
    result: object,
    model_feature: str,
    supported_categories: list[str],
    reference: str,
) -> dict[str, float]:
    """Testuje różnice nachyleń wyłącznie między wspieranymi category2."""
    interaction_names = []
    for category in supported_categories:
        if category == reference:
            continue
        interaction = interaction_parameter(
            list(result.fe_params.index),
            model_feature,
            category,
        )
        if interaction is not None:
            interaction_names.append(interaction)
    if not interaction_names:
        return {
            "interaction_parameters": 0,
            "wald_statistic": np.nan,
            "degrees_of_freedom": 0,
            "p_value": np.nan,
        }
    coefficients = result.fe_params.loc[interaction_names].to_numpy()
    covariance = fixed_effect_covariance(result).loc[
        interaction_names,
        interaction_names,
    ].to_numpy()
    inverse = np.linalg.pinv(covariance)
    statistic = float(coefficients.T @ inverse @ coefficients)
    degrees_of_freedom = int(np.linalg.matrix_rank(covariance))
    return {
        "interaction_parameters": len(interaction_names),
        "wald_statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": float(chi2.sf(statistic, degrees_of_freedom)),
    }


def random_effect_diagnostics(result: object) -> dict[str, float]:
    """Rozkłada wariancję na markę, produkt i poziom opinii."""
    brand_variance = (
        float(result.cov_re.iloc[0, 0]) if not result.cov_re.empty else 0.0
    )
    product_variance = float(result.vcomp[0]) if len(result.vcomp) else 0.0
    residual_variance = float(result.scale)
    total_variance = brand_variance + product_variance + residual_variance
    return {
        "brand_variance": brand_variance,
        "product_variance": product_variance,
        "residual_variance": residual_variance,
        "brand_icc": (
            brand_variance / total_variance if total_variance > 0 else np.nan
        ),
        "product_and_brand_icc": (
            (brand_variance + product_variance) / total_variance
            if total_variance > 0
            else np.nan
        ),
    }


# Jeden model hipotezy


def fit_hypothesis_model(
    review_aspects: pd.DataFrame,
    hypothesis: dict[str, str],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Dopasowuje model i zwraca efekt średni, kategorie oraz diagnostykę."""
    data, supported, controls, model_feature = prepare_model_data(
        review_aspects,
        hypothesis,
    )
    fixed_terms = [
        f"{model_feature} * C(category_model)",
        *controls,
    ]
    formula = "sentiment_score ~ " + " + ".join(fixed_terms)
    result, optimizer, model_warnings = fit_mixed_model(formula, data)

    reference = str(data["category_model"].cat.categories[0])
    product_weights = (
        data.drop_duplicates("id")["category_model"].value_counts()
    )
    pooled_statistics = average_category_slope(
        result,
        model_feature,
        reference,
        product_weights,
    )
    pooled_row = {
        **hypothesis,
        "review_aspects_count": len(data),
        "products_count": data["id"].nunique(),
        "brands_count": data["brand_group"].nunique(),
        "category2_count": data["category2"].nunique(),
        "supported_interaction_categories": len(supported),
        **pooled_statistics,
    }

    category_rows = []
    outcome_standard_deviation = data["sentiment_score"].std(ddof=0)
    categories_to_report = [
        *supported,
        *(["__other__"] if "__other__" in product_weights.index else []),
    ]
    for category in categories_to_report:
        statistics = category_slope(
            result,
            model_feature,
            category,
            reference,
        )
        category_data = data.loc[data["category_model"].eq(category)]
        category_products = category_data.drop_duplicates("id")
        values = category_products[hypothesis["feature"]]
        feature_present = category_data[hypothesis["feature"]].gt(0)
        feature_absent = category_data[hypothesis["feature"]].eq(0)
        contrast_definition = (
            "with ingredient vs without ingredient"
            if hypothesis["feature_type"] == "binary"
            else "change per 1 SD of ingredient count"
        )
        category_rows.append(
            {
                **hypothesis,
                "category2": category,
                "category_support": (
                    "supported" if category in supported else "pooled_other"
                ),
                "review_aspects_count": len(category_data),
                "opinions_count": category_data["review_id"].nunique(),
                "products_count": len(category_products),
                "feature_present_products": int(values.gt(0).sum()),
                "feature_absent_products": int(values.eq(0).sum()),
                "feature_present_opinions": int(
                    category_data.loc[feature_present, "review_id"].nunique()
                ),
                "feature_absent_opinions": int(
                    category_data.loc[feature_absent, "review_id"].nunique()
                ),
                "contrast_definition": contrast_definition,
                "adjusted_sentiment_difference": statistics["coefficient"],
                "adjusted_difference_ci_95_lower": statistics["ci_95_lower"],
                "adjusted_difference_ci_95_upper": statistics["ci_95_upper"],
                "effect_size": (
                    statistics["coefficient"] / outcome_standard_deviation
                    if outcome_standard_deviation > 0
                    else np.nan
                ),
                "effect_size_metric": (
                    "adjusted difference / SD of aspect sentiment_score"
                ),
                **statistics,
            }
        )

    interaction_row = {
        **hypothesis,
        "reference_category": reference,
        "supported_categories": "; ".join(supported),
        **interaction_wald_test(
            result,
            model_feature,
            supported,
            reference,
        ),
    }
    variance = random_effect_diagnostics(result)
    diagnostics_row = {
        **hypothesis,
        "model_formula": formula,
        "optimizer": optimizer,
        "converged": bool(result.converged),
        "review_aspects_count": len(data),
        "reviews_count": data["review_id"].nunique(),
        "products_count": data["id"].nunique(),
        "brands_count": data["brand_group"].nunique(),
        "fixed_effects_count": len(result.fe_params),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        **variance,
        "boundary_warning": any(
            "boundary" in warning.lower() for warning in model_warnings
        ),
        "warnings": " | ".join(dict.fromkeys(model_warnings)),
    }
    return pooled_row, category_rows, interaction_row, diagnostics_row


# FDR i zgodność z modelem produktowym


def add_grouped_fdr(
    data: pd.DataFrame,
    p_value_column: str,
) -> pd.DataFrame:
    """Koryguje FDR w obrębie wcześniej określonej rodziny hipotez."""
    result = data.copy()
    result["q_value_fdr_bh"] = np.nan
    for indices in result.groupby("family").groups.values():
        index = list(indices)
        valid = result.loc[index, p_value_column].notna()
        valid_index = result.loc[index].index[valid]
        if len(valid_index):
            result.loc[valid_index, "q_value_fdr_bh"] = multipletests(
                result.loc[valid_index, p_value_column],
                method="fdr_bh",
            )[1]
    result["p_value_display"] = result[p_value_column].map(
        probability_display
    )
    result["q_value_fdr_bh_display"] = result["q_value_fdr_bh"].map(
        probability_display
    )
    result["fdr_significant_0_05"] = result["q_value_fdr_bh"].le(0.05)
    return result


def add_category_effect_fdr(data: pd.DataFrame) -> pd.DataFrame:
    """Koryguje tylko interpretowalne efekty wspieranych category2."""
    result = data.copy()
    result["q_value_fdr_bh"] = np.nan
    supported = result["category_support"].eq("supported")
    for indices in result.loc[supported].groupby("family").groups.values():
        index = list(indices)
        result.loc[index, "q_value_fdr_bh"] = multipletests(
            result.loc[index, "p_value"],
            method="fdr_bh",
        )[1]
    result["p_value_display"] = result["p_value"].map(
        probability_display
    )
    result["q_value_fdr_bh_display"] = result["q_value_fdr_bh"].map(
        probability_display
    )
    result["fdr_significant_0_05"] = (
        supported & result["q_value_fdr_bh"].le(0.05)
    )
    return result


def add_product_level_comparison(pooled: pd.DataFrame) -> pd.DataFrame:
    """Porównuje MixedLM z wcześniejszym modelem równoważącym produkty."""
    product_results = pd.read_csv(product_level_results_csv)[
        [
            "feature",
            "aspect",
            "primary_coefficient",
            "primary_ci_95_lower",
            "primary_ci_95_upper",
            "q_value_fdr_bh",
            "robust_result",
        ]
    ].rename(
        columns={
            "primary_coefficient": "product_level_coefficient",
            "primary_ci_95_lower": "product_level_ci_95_lower",
            "primary_ci_95_upper": "product_level_ci_95_upper",
            "q_value_fdr_bh": "product_level_q_value_fdr_bh",
            "robust_result": "product_level_robust_result",
        }
    )
    product_level_columns = [
        column
        for column in product_results.columns
        if column not in {"feature", "aspect"}
    ]
    # Przy ponownym uruchomieniu można użyć zapisanych wyników. Usuwamy stare
    # kolumny porównania, aby nie tworzyć ich kolejnych kopii.
    result = pooled.drop(
        columns=[column for column in product_level_columns if column in pooled],
        errors="ignore",
    ).merge(
        product_results,
        on=["feature", "aspect"],
        how="left",
        validate="one_to_one",
    )
    result["direction_consistent_with_product_level"] = (
        np.sign(result["coefficient"])
        == np.sign(result["product_level_coefficient"])
    )
    result["convergent_result"] = (
        result["fdr_significant_0_05"]
        & result["product_level_robust_result"].fillna(False)
        & result["direction_consistent_with_product_level"]
    )
    return result



# Uruchomienie całej analizy

def checkpoint_records(path: Path) -> list[dict[str, object]]:
    """Wczytuje zapisane rekordy, umożliwiając wznowienie obliczeń."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def hypothesis_key(values: dict[str, object]) -> tuple[str, str]:
    """Tworzy stabilny identyfikator jednej pary składnik–aspekt."""
    return str(values["feature"]), str(values["aspect"])


def main() -> None:
    """Dopasowuje modele partiami i finalizuje wyniki po ostatniej partii."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Maximum number of unfinished models to fit in this run.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="liczy wszystkie modele od początku",
    )
    arguments = parser.parse_args()

    output_dir.mkdir(parents=True, exist_ok=True)
    review_aspects = load_review_aspects()

    pooled_rows = [] if arguments.refit else checkpoint_records(pooled_effects_csv)
    category_rows = [] if arguments.refit else checkpoint_records(category_effects_csv)
    interaction_rows = (
        [] if arguments.refit else checkpoint_records(interaction_tests_csv)
    )
    diagnostic_rows = [] if arguments.refit else checkpoint_records(diagnostics_csv)
    completed = {hypothesis_key(row) for row in diagnostic_rows}
    pending = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis_key(hypothesis) not in completed
    ]
    selected = (
        pending[: arguments.max_models]
        if arguments.max_models is not None
        else pending
    )

    for index, hypothesis in enumerate(selected, start=1):
        print(
            f"[{len(completed) + index}/{len(hypotheses)}] "
            f"{hypothesis['feature']} -> {hypothesis['aspect']}",
            flush=True,
        )
        pooled, categories, interaction, diagnostics = fit_hypothesis_model(
            review_aspects,
            hypothesis,
        )
        pooled_rows.append(pooled)
        category_rows.extend(categories)
        interaction_rows.append(interaction)
        diagnostic_rows.append(diagnostics)

        save_csv(pd.DataFrame(pooled_rows), pooled_effects_csv)
        save_csv(pd.DataFrame(category_rows), category_effects_csv)
        save_csv(pd.DataFrame(interaction_rows), interaction_tests_csv)
        save_csv(pd.DataFrame(diagnostic_rows), diagnostics_csv)

    remaining = len(pending) - len(selected)
    if remaining > 0:
        print(f"Checkpoint saved; models remaining: {remaining}", flush=True)
        return

    pooled = add_grouped_fdr(
        pd.DataFrame(pooled_rows),
        "p_value",
    )
    pooled = add_product_level_comparison(pooled)
    category_effects = add_category_effect_fdr(pd.DataFrame(category_rows))
    combination_results = category_effects.loc[
        category_effects["category_support"].eq("supported"),
        [
            "family",
            "feature",
            "feature_type",
            "aspect",
            "category2",
            "contrast_definition",
            "adjusted_sentiment_difference",
            "adjusted_difference_ci_95_lower",
            "adjusted_difference_ci_95_upper",
            "effect_size",
            "effect_size_metric",
            "products_count",
            "feature_present_products",
            "feature_absent_products",
            "opinions_count",
            "feature_present_opinions",
            "feature_absent_opinions",
            "p_value",
            "q_value_fdr_bh",
            "fdr_significant_0_05",
        ],
    ].copy()
    interactions = add_grouped_fdr(
        pd.DataFrame(interaction_rows),
        "p_value",
    )
    diagnostics = pd.DataFrame(diagnostic_rows)

    save_csv(pooled, pooled_effects_csv)
    save_csv(category_effects, category_effects_csv)
    save_csv(combination_results, combination_results_csv)
    save_csv(interactions, interaction_tests_csv)
    save_csv(diagnostics, diagnostics_csv)

    methodology = pd.DataFrame(
        [
            ("observation_unit", "one review-aspect observation"),
            (
                "hierarchy",
                "reviews nested in products; products nested in brands",
            ),
            (
                "category2",
                "fixed effects with ingredient × category2 interactions",
            ),
            (
                "random_effects",
                "brand random intercept and product variance component",
            ),
            (
                "outcome",
                "OOF aspect-level sentiment_score collapsed to review-aspect",
            ),
            (
                "controls",
                "standardized log price, log opinions and ingredients_count; active_count omits ingredients_count",
            ),
            (
                "interaction_support",
                "binary: at least 10 products with and without feature; continuous: 30 products and 3 values",
            ),
            (
                "average_effect",
                "category slopes weighted by product counts, not review counts",
            ),
            (
                "heterogeneity_test",
                "joint Wald test among supported category2 only; pooled other excluded",
            ),
            (
                "multiple_testing",
                "Benjamini-Hochberg FDR within prespecified hypothesis family",
            ),
            (
                "effect_size",
                "adjusted category contrast divided by the SD of aspect sentiment_score in the model sample",
            ),
            (
                "validation",
                "direction compared with equal-product product-level model",
            ),
            (
                "causal_interpretation",
                "not permitted; estimates are adjusted associations",
            ),
        ],
        columns=["parameter", "value"],
    )
    save_csv(methodology, methodology_csv)

    print(
        f"Converged models: {int(diagnostics['converged'].sum())}/"
        f"{len(diagnostics)}",
        flush=True,
    )
    print(
        f"Convergent pooled results: {int(pooled['convergent_result'].sum())}",
        flush=True,
    )
    print(f"Saved multilevel analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
