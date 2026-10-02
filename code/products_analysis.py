from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


# Konfiguracja

# Główne ścieżki wejścia i wyjścia. Skrypt czyta dane surowe z katalogu raw,
# a wszystkie pliki wynikowe zapisuje w katalogu processed.
root = Path(__file__).resolve().parents[1]
CSV_FLOAT_FORMAT = "%.2f"
input_csv = root / "raw" / "products_analysis.csv"
kosmopedia_csv = root / "raw" / "kosmopedia_ingredients.csv"
output_csv = root / "processed" / "products_analysis_processed.csv"
ingredient_frequency_csv = root / "processed" / "ingredient_frequency.csv"
ingredient_frequency_unmapped_csv = (
    root / "processed" / "ingredient_frequency_unmapped.csv"
)
category_summary_csv = root / "processed" / "summary" / "category_summary.csv"
brand_summary_csv = root / "processed" / "summary" / "brand_summary.csv"
ingredient_flag_summary_csv = (
    root / "processed" / "summary" / "ingredient_flag_summary.csv"
)
ingredient_family_group_comparison_csv = (
    root / "processed" / "summary" / "ingredient_family_group_comparison.csv"
)
ingredient_family_mannwhitney_tests_csv = (
    root / "processed" / "summary" / "ingredient_family_mannwhitney_tests.csv"
)
ingredient_family_regression_models_csv = (
    root / "processed" / "summary" / "ingredient_family_regression_models.csv"
)
weighted_rating_regression_coefficients_csv = (
    root / "processed" / "summary" / "weighted_rating_regression_coefficients.csv"
)
weighted_rating_regression_model_summary_csv = (
    root / "processed" / "summary" / "weighted_rating_regression_model_summary.csv"
)
weighted_rating_regression_ingredient_effects_csv = (
    root / "processed" / "summary" / "weighted_rating_regression_ingredient_effects.csv"
)
weighted_rating_active_sensitivity_csv = (
    root / "processed" / "summary" / "weighted_rating_active_sensitivity.csv"
)
product_characteristics_correlation_csv = (
    root / "processed" / "summary" / "product_characteristics_correlation.csv"
)
product_formula_correlation_csv = (
    root / "processed" / "summary" / "product_formula_correlation.csv"
)
active_intensity_correlation_csv = (
    root / "processed" / "summary" / "active_intensity_correlation.csv"
)
ingredient_family_correlation_csv = (
    root / "processed" / "summary" / "ingredient_family_correlation.csv"
)
category_specific_summary_dir = root / "processed" / "summary" / "by_category"
subcategory_specific_summary_dir = (
    root / "processed" / "summary" / "by_subcategory"
)
segmentation_dir = root / "processed" / "segmentation"
segmentation_dataset_csv = segmentation_dir / "segmentation_dataset.csv"
segmentation_results_csv = segmentation_dir / "segmentation_results.csv"
segment_summary_csv = segmentation_dir / "segment_summary.csv"
segmentation_model_metrics_csv = segmentation_dir / "segmentation_model_metrics.csv"
face_care_segmentation_dir = segmentation_dir / "pielegnacja_twarzy"
face_care_segmentation_dataset_csv = face_care_segmentation_dir / "segmentation_dataset.csv"
face_care_segmentation_results_csv = face_care_segmentation_dir / "segmentation_results.csv"
face_care_segment_summary_csv = face_care_segmentation_dir / "segment_summary.csv"
face_care_segmentation_model_metrics_csv = (
    face_care_segmentation_dir / "segmentation_model_metrics.csv"
)
cleansing_segmentation_dir = segmentation_dir / "oczyszczanie_i_demakijaz_twarzy"
cleansing_segmentation_dataset_csv = cleansing_segmentation_dir / "segmentation_dataset.csv"
cleansing_segmentation_results_csv = cleansing_segmentation_dir / "segmentation_results.csv"
cleansing_segment_summary_csv = cleansing_segmentation_dir / "segment_summary.csv"
cleansing_segmentation_model_metrics_csv = (
    cleansing_segmentation_dir / "segmentation_model_metrics.csv"
)
ingredient_functional_groups_csv = (
    root / "processed" / "ingredient_functional_groups.csv"
)
inci_mapping_csv = root / "processed" / "inci_mapping.csv"

# Stałe nazw kolumn. Dzięki temu zmiana nazwy kolumny w jednym miejscu nie
# wymaga ręcznego szukania jej w całym pliku.
ingredients_column = "ingredients"
capacity_column = "capacity"
unit_column = "unit"
standardized_price_per_unit_column = "standardized_price_per_unit"
log_standardized_price_column = "log_standardized_price"
ingredients_count_column = "ingredients_count"
active_count_column = "active_count"
active_share_column = "active_share"
active_position_score_column = "active_position_score"
fragrance_count_column = "fragrance_count"
fragrance_allergen_count_column = "fragrance_allergen_count"
preservative_count_column = "preservative_count"
is_alcohol_denat_column = "is_alcohol_denat"
drying_alcohol_count_column = "drying_alcohol_count"
is_drying_alcohol_column = "is_drying_alcohol"
is_parfum_column = "is_parfum"
is_colorant_column = "is_colorant"
ingredient_frequency_column = "frequency"
ingredient_variants_column = "variants"
functional_group_code_column = "functional_group_code"
mapping_original_column = "Oryginalna_Nazwa"
mapping_clean_column = "Oczyszczona_wersja_skladnika"
ignore_mapping_value = "IGNORE"
kosmopedia_function_code_column = "aktywna_grupa_funkcyjna_kod"
category_column = "category"
category_1_column = "category1"
category_2_column = "category2"
product_url_column = "url"
brand_column = "brand"
price_column = "price"
rating_column = "stars"
weighted_rating_column = "weighted_rating"
opinions_column = "opinions"
log_opinions_column = "log_opinions"
segment_column = "segment"
segment_label_column = "segment_label"
face_care_category_1 = "Pielęgnacja twarzy"
cleansing_category_1 = "Oczyszczanie i demakijaż twarzy"

binary_feature_weight = 0.5
segmentation_k_values = tuple(range(3, 9))
segmentation_continuous_columns = (
    log_standardized_price_column,
    log_opinions_column,
    ingredients_count_column,
    active_share_column,
    active_position_score_column,
    fragrance_allergen_count_column,
    preservative_count_column,
    drying_alcohol_count_column,
)

kosmopedia_match_columns = (
    "inci_name",
    "nazwa_zwyczajowa",
    "identyfikacja_chemiczna",
)

# Zakres analizy: zostają tylko produkty z kategorii twarzy, bez akcesoriów.
included_category_prefix = "Pielęgnacja i higiena-Twarz-"
excluded_category_parts = (
    "Akcesoria do twarzy",
    "Akcesoria do depilacji",
    "Pęsety",
    "Zestawy kosmetyków",
)

# Krótkołańcuchowe, lotne alkohole mogą zwiększać odtłuszczanie i wysuszanie
# skóry, ale sam skład INCI nie pozwala ocenić ich stężenia ani działania całej
# formulacji. Nie zaliczamy tu alkoholi tłuszczowych, np. Cetyl Alcohol.
potentially_drying_alcohols = frozenset(
    {
        "ALCOHOL",
        "ALCOHOL DENAT",
        "ETHANOL",
        "ETHYL ALCOHOL",
        "ISOPROPANOL",
        "ISOPROPYL ALCOHOL",
        "PROPYL ALCOHOL",
        "SD ALCOHOL 40",
        "SD ALCOHOL 40-B",
    }
)

# Nazwy wymagające indywidualnego oznakowania jako alergeny zapachowe według
# załącznika III rozporządzenia (WE) nr 1223/2009, zaktualizowanego
# rozporządzeniem (UE) 2023/1545. Zestaw zawiera również kanoniczne warianty
# występujące w lokalnym inci_mapping.
eu_fragrance_allergens = frozenset(
    {
        "3-PROPYLIDENEPHTHALIDE",
        "6-METHYL COUMARIN",
        "ACETYL CEDRENE",
        "ALPHA-ISOMETHYL IONONE",
        "ALPHA-TERPINENE",
        "AMYL CINNAMAL",
        "AMYL SALICYLATE",
        "AMYLCINNAMYL ALCOHOL",
        "ANETHOLE",
        "ANISE ALCOHOL",
        "BENZALDEHYDE",
        "BENZYL ALCOHOL",
        "BENZYL BENZOATE",
        "BENZYL CINNAMATE",
        "BENZYL SALICYLATE",
        "BETA-CARYOPHYLLENE",
        "CAMPHOR",
        "CANANGA ODORATA FLOWER EXTRACT",
        "CANANGA ODORATA FLOWER OIL",
        "CANANGA ODORATA OIL",
        "CARVONE",
        "CEDRUS ATLANTICA OIL-EXTRACT",
        "CINNAMAL",
        "CINNAMOMUM CASSIA LEAF OIL",
        "CINNAMOMUM ZEYLANICUM BARK OIL",
        "CINNAMYL ALCOHOL",
        "CITRAL",
        "CITRONELLOL",
        "CITRUS AURANTIUM AMARA FLOWER OIL",
        "CITRUS AURANTIUM AMARA PEEL OIL",
        "CITRUS AURANTIUM BERGAMIA FRUIT OIL",
        "CITRUS AURANTIUM BERGAMIA PEEL OIL",
        "CITRUS AURANTIUM DULCIS FLOWER OIL",
        "CITRUS AURANTIUM DULCIS PEEL OIL",
        "CITRUS AURANTIUM PEEL OIL",
        "CITRUS BERGAMIA OIL",
        "CITRUS LIMON OIL",
        "CITRUS LIMON PEEL OIL",
        "CITRUS SINENSIS PEEL OIL",
        "COMARIN",
        "COUMARIN",
        "CYMBOPOGON CITRATUS LEAF OIL",
        "CYMBOPOGON FLEXUOSUS HERB OIL",
        "CYMBOPOGON FLEXUOSUS OIL",
        "CYMBOPOGON SCHOENANTHUS OIL",
        "DIMETHYL PHENETHYL ACETATE",
        "EUCALYPTUS GLOBULUS LEAF OIL",
        "EUCALYPTUS GLOBULUS LEAF/TWIG OIL",
        "EUCALYPTUS GLOBULUS OIL",
        "EUGENIA CARYOPHYLLUS BUD OIL",
        "EUGENIA CARYOPHYLLUS FLOWER OIL",
        "EUGENIA CARYOPHYLLUS LEAF OIL",
        "EUGENIA CARYOPHYLLUS STEM OIL",
        "EUGENOL",
        "EUGENYL ACETATE",
        "EVERNIA FURFURACEA EXTRACT",
        "EVERNIA PRUNASTRI EXTRACT",
        "FARNESOL",
        "GERANIOL",
        "GERANYL ACETATE",
        "HEXAMETHYLINDANOPYRAN",
        "HEXADECANOLACTONE",
        "HEXYL CINNAMAL",
        "HYDROXYCITRONELLAL",
        "HYDROXYISOHEXYL 3-CYCLOHEXENE CARBOXALDEHYDE",
        "ISOEUGENOL",
        "ISOEUGENYL ACETATE",
        "JASMINE OIL",
        "JASMINUM GRANDIFLORUM FLOWER EXTRACT",
        "JASMINUM OFFICINALE FLOWER EXTRACT",
        "JASMINUM OFFICINALE OIL",
        "JUNIPERUS VIRGINIANA OIL",
        "JUNIPERUS VIRGINIANA WOOD OIL",
        "LAURUS NOBILIS LEAF OIL",
        "LAVANDULA ANGUSTIFOLIA FLOWER OIL",
        "LAVANDULA ANGUSTIFOLIA OIL",
        "LAVANDULA HYBRIDA OIL",
        "LAVANDULA INTERMEDIA OIL",
        "LAVANDULA OIL",
        "LEMONGRASS OIL",
        "LIMONENE",
        "LINALOOL",
        "LINALYL ACETATE",
        "LIPPIA CITRIODORA ABSOLUTE",
        "MENTHA PIPERITA OIL",
        "MENTHA VIRIDIS LEAF OIL",
        "MENTHOL",
        "METHYL 2-OCTYNOATE",
        "METHYL SALICYLATE",
        "MYROXYLON PEREIRAE OIL",
        "NARCISSUS JONQUILLA EXTRACT",
        "NARCISSUS POETICUS EXTRACT",
        "NARCISSUS PSEUDONARCISSUS FLOWER EXTRACT",
        "NARCISSUS TAZETTA EXTRACT",
        "PELARGONIUM GRAVEOLENS FLOWER OIL",
        "PINENE",
        "PINUS MUGO LEAF OIL",
        "PINUS MUGO TWIG LEAF EXTRACT",
        "PINUS MUGO TWIG OIL",
        "PINUS PUMILA NEEDLE EXTRACT",
        "POGOSTEMON CABLIN OIL",
        "ROSA ALBA FLOWER EXTRACT",
        "ROSA ALBA FLOWER OIL",
        "ROSA CANINA FLOWER OIL",
        "ROSA CENTIFOLIA FLOWER EXTRACT",
        "ROSA CENTIFOLIA FLOWER OIL",
        "ROSA DAMASCENA FLOWER EXTRACT",
        "ROSA DAMASCENA FLOWER OIL",
        "ROSA GALLICA FLOWER OIL",
        "ROSA MOSCHATA FLOWER OIL",
        "ROSA RUGOSA FLOWER OIL",
        "ROSE FLOWER OIL",
        "ROSE KETONES",
        "SALICYLALDEHYDE",
        "SANTALOL",
        "SANTALUM ALBUM OIL",
        "SCLAREOL",
        "TERPINEOL",
        "TERPINOLENE",
        "TETRA-METHYL ACETYLOCTAHYDRONAPHTHALENES",
        "TETRAMETHYL ACETYLOCTAHYDRONAPHTHALENES",
        "TRIMETHYLBENZENEPROPANOL",
        "TRIMETHYLCYCLOPENTENYL METHYLISOPENTENOL",
        "TURPENTINE",
        "VANILLIN",
    }
)

ingredient_family_columns = {
    "is_niacinamide": {"NIACINAMIDE", "NICOTINAMIDE"},
    "is_ceramides": ("CERAMIDE",),
    "is_hyaluronic_acid": ("HYALURON",),
    "is_retinoids": (
        "RETIN",
        "HYDROXYPINACOLONE RETINOATE",
    ),
    "is_vitamin_c": ("ASCORB", "VITAMIN C"),
    "is_peptides": ("PEPTIDE",),
    "is_centella_asiatica": (
        "CENTELLA",
        "ASIATICOSIDE",
        "MADECASSOSIDE",
        "ASIATIC ACID",
        "MADECASSIC ACID",
    ),
    "is_salicylic_acid": (
        "SALICYLIC ACID",
        "SODIUM SALICYLATE",
        "BETAINE SALICYLATE",
        "CAPRYLOYL SALICYLIC ACID",
        "SALICYLOYL PHYTOSPHINGOSINE",
    ),
    "is_panthenol": ("PANTHENOL", "PANTHENYL"),
    "is_squalane": ("SQUALANE",),
    "is_aha": (
        "GLYCOLIC ACID",
        "LACTIC ACID",
        "MANDELIC ACID",
    ),
}

ingredient_family_exact_matches = {
    "is_niacinamide": {"NIACINAMIDE", "NICOTINAMIDE"},
    "is_aha": {
        "GLYCOLIC ACID",
        "LACTIC ACID",
        "MANDELIC ACID",
    },
}

segmentation_binary_columns = (
    is_parfum_column,
    is_colorant_column,
    *tuple(ingredient_family_columns),
)


@dataclass(frozen=True)
class InciDictionary:
    """Słownik INCI używany w całej analizie.

    lookup mapuje wariant źródłowy z Oryginalna_Nazwa na zestaw nazw
    oczyszczonych. canonical_by_initial przyspiesza rozpoznawanie sklejonych
    składów, w których brakuje przecinków albo innych separatorów.
    """

    lookup: dict[str, frozenset[str]]
    canonical_by_initial: dict[str, tuple[str, ...]]


# Funkcje pomocnicze


def require_columns(
    data: pd.DataFrame,
    columns: set[str],
    source_name: str,
) -> None:
    """Przerywa działanie, jeśli plik wejściowy nie ma wymaganych kolumn."""
    missing_columns = columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required {source_name} columns: "
            + ", ".join(sorted(missing_columns))
        )


def normalize_lookup_text(value: object) -> str:
    """Ujednolica zapis tekstu bez zmiany znaczenia składnika.

    Ta funkcja celowo nie zgaduje synonimów ani nie poprawia nazw chemicznych.
    Normalizuje tylko format: wielkość liter, spacje i odstępy przy ukośnikach.
    Znaczenie składnika zawsze pochodzi z inci_mapping.csv.
    """
    if pd.isna(value):
        return ""

    normalized = str(value).replace("\xa0", " ").strip()
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n,.;:*[]{}").upper()


def split_ingredient_candidates(value: object) -> list[str]:
    """Dzieli skład na kandydatów do mapowania, bez zmiany nazw składników."""
    if pd.isna(value):
        return []

    text = str(value).replace("\xa0", " ")
    text = re.sub(
        r"\[\s*(?:MAY CONTAIN\s*/\s*PEUT CONTENIR\s*/\s*)?\+/-\s*:?\s*",
        ", ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bMAY CONTAIN\s*/\s*PEUT CONTENIR\s*:\s*",
        ", ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[•;|]+", ", ", text)

    # Gwiazdki zwykle oznaczają przypisy albo brakujące separatory. Zamieniamy
    # je na spacje, żeby późniejsze dopasowanie fraz ze słownika mogło odzyskać
    # sąsiadujące składniki.
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"\s+\.\s+", ", ", text)
    text = re.sub(r"\.(?=\s+[A-Za-zÀ-ž])", ", ", text)

    candidates = []
    for candidate in re.split(r",\s+|,(?=[A-Za-zÀ-ž])", text):
        normalized = normalize_lookup_text(candidate)
        if normalized:
            candidates.append(normalized)
    return candidates


def parse_number(value: str) -> float:
    """Zamienia polski przecinek dziesiętny i standardową kropkę na float."""
    return float(value.replace(",", "."))


def format_capacity(value: float) -> int | float:
    """Zostawia pełne pojemności jako int, a ułamkowe delikatnie zaokrągla."""
    return int(value) if value.is_integer() else round(value, 4)


def normalize_unit(value: str) -> str:
    """Ujednolica jednostkę po rozdzieleniu pola capacity."""
    unit = value.strip(".").lower()
    return "para" if unit == "par" else unit


# Zakres produktów i metadane


def filter_products_with_ingredients(products: pd.DataFrame) -> pd.DataFrame:
    """Usuwa produkty bez składu albo z literalną wartością 'brak danych'."""
    require_columns(products, {ingredients_column}, "products")
    ingredients = products[ingredients_column].fillna("").astype(str).str.strip()
    keep = ingredients.ne("") & ingredients.str.lower().ne("brak danych")
    return products.loc[keep].reset_index(drop=True)


def filter_face_care_products(products: pd.DataFrame) -> pd.DataFrame:
    """Zostawia pielęgnację twarzy i usuwa kategorie akcesoriów."""
    require_columns(products, {category_column}, "products")
    categories = products[category_column].fillna("")
    included = categories.str.startswith(included_category_prefix)
    excluded = categories.str.contains("|".join(excluded_category_parts), regex=True)
    return products.loc[included & ~excluded].reset_index(drop=True)


def apply_product_scope(products: pd.DataFrame) -> pd.DataFrame:
    """Stosuje filtry wierszy bez modyfikowania oryginalnego pola ingredients."""
    products = filter_products_with_ingredients(products)
    return filter_face_care_products(products)


def split_category(value: object) -> pd.Series:
    """Rozbija ścieżkę kategorii na category1 i category2."""
    if pd.isna(value):
        return pd.Series({category_1_column: "", category_2_column: ""})

    parts = [part.strip() for part in str(value).split("-")]
    return pd.Series(
        {
            category_1_column: parts[2] if len(parts) > 2 else "",
            category_2_column: parts[3] if len(parts) > 3 else "",
        }
    )


def transform_category_columns(products: pd.DataFrame) -> pd.DataFrame:
    """Zastępuje category kolumnami category1/category2 w tym samym miejscu."""
    require_columns(products, {category_column}, "products")
    category_parts = products[category_column].apply(split_category)
    category_position = products.columns.get_loc(category_column)
    products = products.drop(columns=[category_column])
    return pd.concat(
        [
            products.iloc[:, :category_position],
            category_parts,
            products.iloc[:, category_position:],
        ],
        axis=1,
    )


def split_capacity_value(value: object) -> pd.Series:
    """Rozdziela capacity na liczbową pojemność całkowitą i jednostkę."""
    if pd.isna(value):
        return pd.Series({capacity_column: pd.NA, unit_column: ""})

    capacity = re.sub(r"\s+", " ", str(value).strip())
    number_pattern = r"(\d+(?:[,.]\d+)?)"
    unit_pattern = r"([^\W\d_]+\.?)"

    multiplied_match = re.fullmatch(
        rf"{number_pattern}\s*x\s*{number_pattern}\s*{unit_pattern}",
        capacity,
        flags=re.IGNORECASE,
    )
    if multiplied_match:
        # Przykład: "7 x 2 ml" zapisujemy jako 14 ml.
        count = parse_number(multiplied_match.group(1))
        amount = parse_number(multiplied_match.group(2))
        unit = normalize_unit(multiplied_match.group(3))
        return pd.Series(
            {
                capacity_column: format_capacity(count * amount),
                unit_column: unit,
            }
        )

    per_item_match = re.fullmatch(
        rf"{number_pattern}\s+.+?\s+po\s+{number_pattern}\s*{unit_pattern}",
        capacity,
        flags=re.IGNORECASE,
    )
    if per_item_match:
        # Przykład: "7 ampułek po 2 ml" również zapisujemy jako 14 ml.
        count = parse_number(per_item_match.group(1))
        amount = parse_number(per_item_match.group(2))
        unit = normalize_unit(per_item_match.group(3))
        return pd.Series(
            {
                capacity_column: format_capacity(count * amount),
                unit_column: unit,
            }
        )

    simple_match = re.fullmatch(
        rf"{number_pattern}\s*{unit_pattern}",
        capacity,
        flags=re.IGNORECASE,
    )
    if simple_match:
        amount = parse_number(simple_match.group(1))
        unit = normalize_unit(simple_match.group(2))
        return pd.Series(
            {
                capacity_column: format_capacity(amount),
                unit_column: unit,
            }
        )

    return pd.Series({capacity_column: pd.NA, unit_column: ""})


def transform_capacity_column(products: pd.DataFrame) -> pd.DataFrame:
    """Zastępuje surowe capacity kolumnami capacity i unit."""
    require_columns(products, {capacity_column}, "products")
    capacity_parts = products[capacity_column].apply(split_capacity_value)
    capacity_position = products.columns.get_loc(capacity_column)
    products = products.drop(columns=[capacity_column])
    return pd.concat(
        [
            products.iloc[:, :capacity_position],
            capacity_parts,
            products.iloc[:, capacity_position:],
        ],
        axis=1,
    )


def calculate_standardized_price_per_unit(row: pd.Series) -> float | pd.NA:
    """Przelicza cenę jednostkową na porównywalną bazę."""
    price = row[price_column]
    capacity = row[capacity_column]
    unit = row[unit_column]
    if pd.isna(price) or pd.isna(capacity) or capacity == 0:
        return pd.NA

    if unit in {"ml", "g"}:
        return round(price / capacity * 100, 2)
    if unit == "szt":
        return round(price / capacity, 2)
    if unit == "para":
        return round(price / (capacity * 2), 2)
    return pd.NA


def add_standardized_price_per_unit(products: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cenę jednostkową na wspólnej bazie dla ml, g, szt. i par."""
    require_columns(
        products,
        {price_column, capacity_column, unit_column},
        "products",
    )
    products = products.copy()
    standardized_price_per_unit = products.apply(
        calculate_standardized_price_per_unit,
        axis=1,
    )
    if standardized_price_per_unit_column in products.columns:
        products = products.drop(columns=[standardized_price_per_unit_column])

    standardized_price_position = products.columns.get_loc(price_column) + 1
    if "price_per_unit" in products.columns:
        standardized_price_position = products.columns.get_loc("price_per_unit") + 1
    products.insert(
        standardized_price_position,
        standardized_price_per_unit_column,
        standardized_price_per_unit,
    )
    return products


def add_log_standardized_price(products: pd.DataFrame) -> pd.DataFrame:
    """Dodaje logarytmiczną cenę jednostkową odporniejszą na wartości skrajne."""
    require_columns(products, {standardized_price_per_unit_column}, "products")
    products = products.copy()
    log_standardized_price = np.log1p(
        products[standardized_price_per_unit_column]
    ).round(4)
    if log_standardized_price_column in products.columns:
        products = products.drop(columns=[log_standardized_price_column])

    log_price_position = products.columns.get_loc(standardized_price_per_unit_column) + 1
    products.insert(
        log_price_position,
        log_standardized_price_column,
        log_standardized_price,
    )
    return products


def normalize_product_columns(products: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje metadane produktu bez zmieniania oryginalnego ingredients."""
    products = transform_category_columns(products)
    products = products.drop(columns=[product_url_column], errors="ignore")
    products = transform_capacity_column(products)
    products = add_standardized_price_per_unit(products)
    return add_log_standardized_price(products)



def add_weighted_rating(products: pd.DataFrame) -> pd.DataFrame:
    """Dodaje bayesowską ocenę produktu ważoną liczbą opinii."""
    require_columns(
        products,
        {rating_column, opinions_column},
        "products",
    )
    products = products.copy()
    global_rating_mean = products[rating_column].mean()
    opinion_stability_threshold = products[opinions_column].median()
    denominator = products[opinions_column] + opinion_stability_threshold
    weighted_rating = (
        products[opinions_column].div(denominator).mul(products[rating_column])
        + opinion_stability_threshold / denominator * global_rating_mean
    ).round(4)
    weighted_rating = weighted_rating.where(denominator.ne(0), global_rating_mean)

    if weighted_rating_column in products.columns:
        products = products.drop(columns=[weighted_rating_column])

    weighted_rating_position = products.columns.get_loc(opinions_column) + 1
    products.insert(
        weighted_rating_position,
        weighted_rating_column,
        weighted_rating,
    )
    return products


def add_log_opinions(products: pd.DataFrame) -> pd.DataFrame:
    """Dodaje logarytmiczną miarę popularności produktu."""
    require_columns(products, {opinions_column}, "products")
    products = products.copy()
    log_opinions = np.log1p(products[opinions_column]).round(4)

    if log_opinions_column in products.columns:
        products = products.drop(columns=[log_opinions_column])

    log_opinions_position = products.columns.get_loc(opinions_column) + 1
    products.insert(
        log_opinions_position,
        log_opinions_column,
        log_opinions,
    )
    return products


# Słownik INCI i mapowanie składników produktu


def build_inci_mapping_lookup(
    inci_mapping_path: Path = inci_mapping_csv,
) -> InciDictionary:
    """Buduje mapowanie Oryginalna_Nazwa -> Oczyszczona_wersja_skladnika.

    Jeden wariant źródłowy może prowadzić do kilku składników oczyszczonych,
    dlatego wartości w lookup są zestawami, a nie pojedynczym tekstem.
    """
    mapping = pd.read_csv(inci_mapping_path)
    require_columns(
        mapping,
        {mapping_original_column, mapping_clean_column},
        "INCI mapping",
    )

    mutable_lookup: dict[str, set[str]] = {}
    canonical_names = set()

    for _, row in mapping.iterrows():
        # Obie strony mapy normalizujemy tak samo, żeby różnice w spacjach czy
        # wielkości liter nie blokowały dopasowania.
        source_name = normalize_lookup_text(row[mapping_original_column])
        clean_name = normalize_lookup_text(row[mapping_clean_column])
        if not source_name or not clean_name:
            continue

        mutable_lookup.setdefault(source_name, set()).add(clean_name)
        if clean_name != ignore_mapping_value:
            canonical_names.add(clean_name)

    # Nazwa oczyszczona też może pojawić się bezpośrednio w składzie produktu.
    # Dodajemy ją jako legalny klucz wejściowy.
    for canonical_name in canonical_names:
        mutable_lookup.setdefault(canonical_name, set()).add(canonical_name)

    lookup = {
        source_name: frozenset(clean_names)
        for source_name, clean_names in mutable_lookup.items()
    }
    canonical_by_initial: dict[str, list[str]] = {}
    for canonical_name in canonical_names:
        canonical_by_initial.setdefault(canonical_name[0], []).append(canonical_name)

    return InciDictionary(
        lookup=lookup,
        canonical_by_initial={
            initial: tuple(sorted(names, key=len, reverse=True))
            for initial, names in canonical_by_initial.items()
        },
    )


def non_ignored_names(names: frozenset[str] | set[str]) -> set[str]:
    """Usuwa wpisy słownika oznaczone jawnie jako IGNORE."""
    return {name for name in names if name != ignore_mapping_value}


def normalize_function_codes(value: object) -> tuple[str, ...]:
    """Ujednolica kody grup funkcyjnych z Kosmopedii."""
    if pd.isna(value):
        return ()

    aliases = {
        "fragrance": "fragrances",
        "preservatives": "preservative",
        "filters": "uv filters",
        "dyes": "dyes",
    }
    codes = []
    for code in re.split(r"[,;]", str(value)):
        normalized = re.sub(r"\s+", " ", code).strip().lower()
        if not normalized or normalized == "nan":
            continue
        codes.append(aliases.get(normalized, normalized))

    return tuple(dict.fromkeys(codes))


def kosmopedia_name_keys(value: object) -> set[str]:
    """Tworzy warianty nazwy, po których można szukać składnika w Kosmopedii."""
    if pd.isna(value):
        return set()

    value = str(value)
    keys = {
        normalize_lookup_text(value),
        normalize_lookup_text(re.sub(r"\([^)]*\)", " ", value)),
    }
    return {key for key in keys if key and key not in {"BRAK", "NAN"}}


def build_kosmopedia_function_lookup(
    kosmopedia_path: Path = kosmopedia_csv,
) -> dict[str, frozenset[str]]:
    """Buduje mapę nazwa składnika -> kody grup funkcyjnych z Kosmopedii."""
    kosmopedia = pd.read_csv(kosmopedia_path)
    require_columns(
        kosmopedia,
        set(kosmopedia_match_columns) | {kosmopedia_function_code_column},
        "Kosmopedia ingredients",
    )

    lookup: dict[str, set[str]] = {}
    for _, row in kosmopedia.iterrows():
        codes = normalize_function_codes(row[kosmopedia_function_code_column])
        if not codes:
            continue

        for column in kosmopedia_match_columns:
            for key in kosmopedia_name_keys(row[column]):
                lookup.setdefault(key, set()).update(codes)

    return {
        ingredient_name: frozenset(codes)
        for ingredient_name, codes in lookup.items()
    }


def build_manual_function_lookup(
    function_groups_path: Path = ingredient_functional_groups_csv,
) -> dict[str, frozenset[str]]:
    """Buduje uzupełniającą mapę funkcji z ingredient_functional_groups.csv."""
    function_groups = pd.read_csv(function_groups_path)
    require_columns(
        function_groups,
        {ingredients_column, functional_group_code_column},
        "ingredient functional groups",
    )

    lookup: dict[str, set[str]] = {}
    for _, row in function_groups.iterrows():
        ingredient_name = normalize_lookup_text(row[ingredients_column])
        codes = normalize_function_codes(row[functional_group_code_column])
        if ingredient_name and codes:
            lookup.setdefault(ingredient_name, set()).update(codes)

    return {
        ingredient_name: frozenset(codes)
        for ingredient_name, codes in lookup.items()
    }


def find_function_group_codes(
    ingredient: str,
    observed_variants: set[str],
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]] | None = None,
) -> str:
    """Dopasowuje składnik lub jego warianty do kodów funkcji z Kosmopedii."""
    codes = set()
    manual_functions = manual_functions or {}
    candidates = [ingredient, *sorted(observed_variants)]

    for candidate in candidates:
        for key in kosmopedia_name_keys(candidate):
            codes.update(kosmopedia_functions.get(key, ()))

    if not codes:
        for key in kosmopedia_name_keys(ingredient):
            codes.update(manual_functions.get(key, ()))

    return "; ".join(sorted(codes))


def has_active_function_group(function_codes: str) -> bool:
    """Sprawdza, czy w kodach funkcji występuje dokładny kod active."""
    return "active" in normalize_function_codes(function_codes)


def has_function_group(function_codes: str, expected_code: str) -> bool:
    """Sprawdza, czy w kodach funkcji występuje wskazany kod."""
    return expected_code in normalize_function_codes(function_codes)


def count_active_ingredients(
    mapped_ingredients: dict[str, set[str]],
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]],
) -> int:
    """Liczy unikalne składniki produktu oznaczone kodem active."""
    active_count = 0
    for ingredient, observed_variants in mapped_ingredients.items():
        function_codes = find_function_group_codes(
            ingredient,
            observed_variants,
            kosmopedia_functions,
            manual_functions,
        )
        if has_active_function_group(function_codes):
            active_count += 1
    return active_count


def count_ingredients_by_function_group(
    mapped_ingredients: dict[str, set[str]],
    expected_code: str,
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]],
) -> int:
    """Liczy unikalne składniki produktu z wybranym kodem funkcji."""
    count = 0
    for ingredient, observed_variants in mapped_ingredients.items():
        function_codes = find_function_group_codes(
            ingredient,
            observed_variants,
            kosmopedia_functions,
            manual_functions,
        )
        if has_function_group(function_codes, expected_code):
            count += 1
    return count


def count_fragrance_allergens(
    mapped_ingredients: dict[str, set[str]],
) -> int:
    """Liczy unikalne alergeny zapachowe jawnie wymienione w składzie INCI.

    Reguła bazuje na nazwach z przepisów UE, a nie na ogólnym kodzie
    alergiczności z Kosmopedii. Nie odtwarza substancji ukrytych pod nazwą
    ``Parfum`` i nie pozwala ocenić progów stężenia wymagających etykietowania.
    """
    return len(set(mapped_ingredients).intersection(eu_fragrance_allergens))


def count_named_ingredients(
    mapped_ingredients: dict[str, set[str]],
    ingredient_names: frozenset[str],
) -> int:
    """Liczy unikalne składniki należące do jawnie zdefiniowanego zbioru."""
    return len(set(mapped_ingredients).intersection(ingredient_names))


def has_colorant(
    mapped_ingredients: dict[str, set[str]],
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]],
) -> int:
    """Sprawdza obecność potencjalnego barwnika na podstawie funkcji lub kodu CI."""
    for ingredient, observed_variants in mapped_ingredients.items():
        function_codes = find_function_group_codes(
            ingredient,
            observed_variants,
            kosmopedia_functions,
            manual_functions,
        )
        if has_function_group(function_codes, "dyes"):
            return 1
        if re.fullmatch(r"CI\s*\d{5}", ingredient):
            return 1
    return 0


def active_position_weight(position: int) -> int:
    """Nadaje wagę pozycji składnika aktywnego w składzie."""
    if position <= 5:
        return 3
    if position <= 10:
        return 2
    return 1


def calculate_active_position_score(
    ingredients_text: object,
    ingredient_mapping: InciDictionary,
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]],
) -> int:
    """Liczy punktację pozycji składników aktywnych w składzie produktu."""
    score = 0
    scored_ingredients = set()

    for position, candidate in enumerate(
        split_ingredient_candidates(ingredients_text),
        start=1,
    ):
        candidate_matches = map_ingredient_candidate(candidate, ingredient_mapping)
        for ingredient, observed_variants in candidate_matches.items():
            if ingredient in scored_ingredients:
                continue

            function_codes = find_function_group_codes(
                ingredient,
                observed_variants,
                kosmopedia_functions,
                manual_functions,
            )
            if has_active_function_group(function_codes):
                score += active_position_weight(position)
                scored_ingredients.add(ingredient)

    return score


def ingredient_matches_family(
    ingredient: str,
    column: str,
    patterns: tuple[str, ...],
) -> bool:
    """Sprawdza, czy oczyszczona nazwa składnika należy do danej rodziny."""
    exact_matches = ingredient_family_exact_matches.get(column)
    if exact_matches is not None:
        return ingredient in exact_matches

    return any(pattern in ingredient for pattern in patterns)


def build_ingredient_family_flags(
    mapped_ingredients: dict[str, set[str]],
) -> dict[str, int]:
    """Tworzy binarne flagi rodzin składników na podstawie nazw z INCI mapping."""
    ingredient_names = set(mapped_ingredients)
    return {
        column: int(
            any(
                ingredient_matches_family(ingredient, column, patterns)
                for ingredient in ingredient_names
            )
        )
        for column, patterns in ingredient_family_columns.items()
    }


def has_mapped_ingredient(
    mapped_ingredients: dict[str, set[str]],
    ingredient_name: str,
) -> int:
    """Sprawdza, czy produkt zawiera wskazany oczyszczony składnik."""
    return int(ingredient_name in mapped_ingredients)


def find_ingredient_family_position(
    ingredients_text: object,
    ingredient_mapping: InciDictionary,
    flag_column: str,
) -> int | None:
    """Znajduje pierwszą pozycję składnika z danej rodziny w składzie produktu."""
    patterns = ingredient_family_columns[flag_column]
    for position, candidate in enumerate(
        split_ingredient_candidates(ingredients_text),
        start=1,
    ):
        candidate_matches = map_ingredient_candidate(candidate, ingredient_mapping)
        if any(
            ingredient_matches_family(ingredient, flag_column, patterns)
            for ingredient in candidate_matches
        ):
            return position

    full_targets = ingredient_mapping.lookup.get(normalize_lookup_text(ingredients_text))
    if full_targets and any(
        ingredient_matches_family(ingredient, flag_column, patterns)
        for ingredient in non_ignored_names(full_targets)
    ):
        return 1

    return None


def match_slash_alias(
    candidate: str,
    ingredient_mapping: InciDictionary,
) -> set[str]:
    """Obsługuje nazwy z ukośnikiem, np. AQUA/WATER albo PARFUM/FRAGRANCE.

    Najpierw akceptujemy przypadek, gdy wszystkie części prowadzą do tej samej
    nazwy oczyszczonej. Jeśli nie, używamy tylko najdłuższej części, ale wyłącznie
    wtedy, gdy stanowi większość całej nazwy.
    """
    parts = [
        normalize_lookup_text(part)
        for part in candidate.split("/")
        if normalize_lookup_text(part)
    ]
    if len(parts) < 2:
        return set()

    part_matches = [ingredient_mapping.lookup.get(part) for part in parts]
    if all(part_matches):
        targets = {
            target
            for matches in part_matches
            for target in non_ignored_names(matches)
        }
        if len(targets) == 1:
            return targets

    comparable_length = len(re.sub(r"[^A-Z0-9]", "", candidate))
    partial_matches = []
    for position, (part, matches) in enumerate(zip(parts, part_matches)):
        if not matches:
            continue
        part_length = len(re.sub(r"[^A-Z0-9]", "", part))
        if part_length >= 5:
            partial_matches.append(
                (
                    part_length,
                    -position,
                    non_ignored_names(matches),
                )
            )

    if partial_matches and comparable_length:
        part_length, _, targets = max(partial_matches, key=lambda item: item[:2])
        if part_length / comparable_length >= 0.5:
            return targets

    return set()


def find_dictionary_phrases(
    candidate: str,
    ingredient_mapping: InciDictionary,
) -> dict[str, set[str]]:
    """Odzyskuje frazy słownikowe z tekstu, w którym brakuje separatorów."""
    words = re.findall(r"[A-Z0-9À-Ž]+(?:[,'-][A-Z0-9À-Ž]+)*", candidate)
    if len(words) < 2:
        return {}

    mapped: dict[str, set[str]] = {}
    position = 0

    while position < len(words):
        best_match = None
        for end in range(len(words), position, -1):
            # Szukamy najdłuższej możliwej frazy od bieżącej pozycji. Dzięki temu
            # "SODIUM HYALURONATE" wygra z krótszym "SODIUM".
            phrase = " ".join(words[position:end])
            targets = ingredient_mapping.lookup.get(phrase)
            if targets:
                clean_names = non_ignored_names(targets)
                if clean_names:
                    best_match = (end, phrase, clean_names)
                    break

        if not best_match:
            position += 1
            continue

        end, phrase, clean_names = best_match
        for clean_name in clean_names:
            mapped.setdefault(clean_name, set()).add(phrase)
        position = end

    return mapped


def find_concatenated_canonical_names(
    candidate: str,
    ingredient_mapping: InciDictionary,
) -> dict[str, set[str]]:
    """Segmentuje sklejony tekst nazwami kanonicznymi ze słownika.

    To obsługuje składy zapisane bez przecinków, np. OilHelianthus... Nie ma tu
    ręcznej listy wyjątków: źródłem nazw pozostaje inci_mapping.csv.
    """
    mapped: dict[str, set[str]] = {}
    covered_length = 0
    position = 0

    while position < len(candidate):
        initial = candidate[position]
        match = next(
            (
                name
                for name in ingredient_mapping.canonical_by_initial.get(initial, ())
                if candidate.startswith(name, position)
            ),
            None,
        )
        if not match:
            position += 1
            continue

        mapped.setdefault(match, set()).add(match)
        covered_length += len(re.sub(r"[^A-Z0-9À-Ž]", "", match))
        position += len(match)

    total_length = len(re.sub(r"[^A-Z0-9À-Ž]", "", candidate))
    # Segmentację akceptujemy tylko, gdy pokrywa większość tekstu. To ogranicza
    # ryzyko fałszywego dopasowania pojedynczego krótkiego fragmentu.
    if len(mapped) >= 2 and total_length and covered_length / total_length >= 0.6:
        return mapped
    return {}


def map_ingredient_candidate(
    candidate: str,
    ingredient_mapping: InciDictionary,
) -> dict[str, set[str]]:
    """Mapuje jeden fragment składu: najpierw dokładnie, potem awaryjnie."""
    exact_targets = ingredient_mapping.lookup.get(candidate)
    if exact_targets:
        return {
            clean_name: {candidate}
            for clean_name in non_ignored_names(exact_targets)
        }

    slash_targets = match_slash_alias(candidate, ingredient_mapping)
    if slash_targets:
        return {clean_name: {candidate} for clean_name in slash_targets}

    concatenated_matches = find_concatenated_canonical_names(
        candidate,
        ingredient_mapping,
    )
    if concatenated_matches:
        return concatenated_matches

    return find_dictionary_phrases(candidate, ingredient_mapping)


def map_product_ingredients(
    ingredients_text: object,
    ingredient_mapping: InciDictionary,
) -> tuple[dict[str, set[str]], set[str]]:
    """Mapuje cały skład produktu na nazwy oczyszczone i warianty źródłowe."""
    mapped: dict[str, set[str]] = {}
    unmapped = set()

    # Niektóre wpisy w mapie opisują cały wadliwie zeskrobany skład. Uwzględniamy
    # takie dopasowanie, ale nadal próbujemy rozbić skład na mniejsze elementy.
    full_source = normalize_lookup_text(ingredients_text)
    full_targets = ingredient_mapping.lookup.get(full_source)
    if full_targets:
        for clean_name in non_ignored_names(full_targets):
            mapped.setdefault(clean_name, set()).add(full_source)

    for candidate in split_ingredient_candidates(ingredients_text):
        candidate_matches = map_ingredient_candidate(candidate, ingredient_mapping)
        if not candidate_matches:
            unmapped.add(candidate)
            continue

        for clean_name, variants in candidate_matches.items():
            mapped.setdefault(clean_name, set()).update(variants)

    return mapped, unmapped


# Cechy produktu wyliczane ze składników


def add_ingredient_count(
    products: pd.DataFrame,
    mapped_values: pd.Series,
    ingredient_mapping: InciDictionary,
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]],
) -> pd.DataFrame:
    """Dodaje liczniki składników i binarne cechy wynikające z INCI."""
    mapped_ingredients = mapped_values.apply(lambda value: value[0])
    has_mapped_ingredients = mapped_ingredients.map(bool)
    products = products.loc[has_mapped_ingredients].copy()
    mapped_ingredients = mapped_ingredients.loc[has_mapped_ingredients]
    products[ingredients_count_column] = mapped_ingredients.map(len)
    products[active_count_column] = mapped_ingredients.apply(
        count_active_ingredients,
        kosmopedia_functions=kosmopedia_functions,
        manual_functions=manual_functions,
    )
    products[fragrance_count_column] = mapped_ingredients.apply(
        count_ingredients_by_function_group,
        expected_code="fragrances",
        kosmopedia_functions=kosmopedia_functions,
        manual_functions=manual_functions,
    )
    products[fragrance_allergen_count_column] = mapped_ingredients.apply(
        count_fragrance_allergens
    )
    products[preservative_count_column] = mapped_ingredients.apply(
        count_ingredients_by_function_group,
        expected_code="preservative",
        kosmopedia_functions=kosmopedia_functions,
        manual_functions=manual_functions,
    )
    products[is_alcohol_denat_column] = mapped_ingredients.apply(
        has_mapped_ingredient,
        ingredient_name="ALCOHOL DENAT",
    )
    products[drying_alcohol_count_column] = mapped_ingredients.apply(
        count_named_ingredients,
        ingredient_names=potentially_drying_alcohols,
    )
    products[is_drying_alcohol_column] = (
        products[drying_alcohol_count_column].gt(0).astype(int)
    )
    products[is_parfum_column] = mapped_ingredients.apply(
        has_mapped_ingredient,
        ingredient_name="FRAGRANCE",
    )
    products[is_colorant_column] = mapped_ingredients.apply(
        has_colorant,
        kosmopedia_functions=kosmopedia_functions,
        manual_functions=manual_functions,
    )
    products[active_share_column] = (
        products[active_count_column]
        .div(products[ingredients_count_column])
        .mul(100)
        .round(2)
    )
    products[active_position_score_column] = products[ingredients_column].apply(
        calculate_active_position_score,
        ingredient_mapping=ingredient_mapping,
        kosmopedia_functions=kosmopedia_functions,
        manual_functions=manual_functions,
    )
    ingredient_family_flags = pd.DataFrame(
        mapped_ingredients.apply(build_ingredient_family_flags).tolist(),
        index=products.index,
    )
    products = pd.concat([products, ingredient_family_flags], axis=1)
    return products


# Podsumowanie częstości składników


def build_ingredient_frequency(
    products: pd.DataFrame,
    ingredient_mapping: InciDictionary,
    kosmopedia_functions: dict[str, frozenset[str]] | None = None,
    manual_functions: dict[str, frozenset[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Liczy częstości składników na poziomie produktu.

    Składnik liczony jest maksymalnie raz w jednym produkcie, bo mapped jest
    słownikiem unikalnych nazw oczyszczonych. variants pokazuje formy, w jakich
    dana nazwa została znaleziona w oryginalnych składach.
    """
    require_columns(products, {ingredients_column}, "products")
    kosmopedia_functions = kosmopedia_functions or {}
    manual_functions = manual_functions or {}
    frequencies: dict[str, int] = {}
    variants: dict[str, set[str]] = {}
    unmapped_frequencies: dict[str, int] = {}

    for ingredients_text in products[ingredients_column]:
        mapped, unmapped = map_product_ingredients(
            ingredients_text,
            ingredient_mapping,
        )
        for clean_name, observed_variants in mapped.items():
            frequencies[clean_name] = frequencies.get(clean_name, 0) + 1
            variants.setdefault(clean_name, set()).update(observed_variants)
        for candidate in unmapped:
            unmapped_frequencies[candidate] = (
                unmapped_frequencies.get(candidate, 0) + 1
            )

    frequency_rows = [
        {
            ingredients_column: clean_name,
            functional_group_code_column: find_function_group_codes(
                clean_name,
                variants[clean_name],
                kosmopedia_functions,
                manual_functions,
            ),
            ingredient_frequency_column: frequency,
            ingredient_variants_column: "; ".join(sorted(variants[clean_name])),
        }
        for clean_name, frequency in frequencies.items()
    ]
    unmapped_rows = [
        {
            ingredients_column: candidate,
            ingredient_frequency_column: frequency,
        }
        for candidate, frequency in unmapped_frequencies.items()
    ]

    ingredient_frequency = pd.DataFrame(
        frequency_rows,
        columns=[
            ingredients_column,
            functional_group_code_column,
            ingredient_frequency_column,
            ingredient_variants_column,
        ],
    ).sort_values(
        [ingredient_frequency_column, ingredients_column],
        ascending=[False, True],
        ignore_index=True,
    )
    unmapped_frequency = pd.DataFrame(
        unmapped_rows,
        columns=[ingredients_column, ingredient_frequency_column],
    ).sort_values(
        [ingredient_frequency_column, ingredients_column],
        ascending=[False, True],
        ignore_index=True,
    )
    return ingredient_frequency, unmapped_frequency


# Podsumowania opisowe


def aggregate_category_metrics(
    products: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Liczy podstawowe metryki produktu dla wskazanego poziomu kategorii."""
    return (
        products.groupby(group_columns, dropna=False)
        .agg(
            product_count=("id", "count"),
            median_price=(price_column, "median"),
            median_standardized_price_per_unit=(
                standardized_price_per_unit_column,
                "median",
            ),
            avg_weighted_rating=(weighted_rating_column, "mean"),
            median_opinions=(opinions_column, "median"),
            median_ingredients_count=(ingredients_count_column, "median"),
            median_active_count=(active_count_column, "median"),
            avg_active_share=(active_share_column, "mean"),
            median_active_position_score=(active_position_score_column, "median"),
        )
        .reset_index()
        .round(
            {
                "median_price": 2,
                "median_standardized_price_per_unit": 2,
                "avg_weighted_rating": 2,
                "median_opinions": 2,
                "median_ingredients_count": 2,
                "median_active_count": 2,
                "avg_active_share": 2,
                "median_active_position_score": 2,
            }
        )
    )


def build_category_summary(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy tabelę category1 jako sumy i category2 jako rozbicie."""
    require_columns(
        products,
        {
            category_1_column,
            category_2_column,
            price_column,
            standardized_price_per_unit_column,
            weighted_rating_column,
            opinions_column,
            ingredients_count_column,
            active_count_column,
            active_share_column,
            active_position_score_column,
        },
        "processed products",
    )

    category1_totals = aggregate_category_metrics(products, [category_1_column])
    category1_totals[category_2_column] = "SUMA"
    category1_totals["summary_level"] = "category1_total"

    category2_rows = aggregate_category_metrics(
        products,
        [category_1_column, category_2_column],
    )
    category2_rows["summary_level"] = "category2"

    category1_order = {
        row[category_1_column]: position
        for position, row in category1_totals.sort_values(
            ["product_count", category_1_column],
            ascending=[False, True],
        ).reset_index(drop=True).iterrows()
    }
    summary = pd.concat(
        [category1_totals, category2_rows],
        ignore_index=True,
    )
    summary["_category1_order"] = summary[category_1_column].map(category1_order)
    summary["_level_order"] = summary["summary_level"].map(
        {"category1_total": 0, "category2": 1}
    )
    return (
        summary.sort_values(
            [
                "_category1_order",
                "_level_order",
                "product_count",
                category_2_column,
            ],
            ascending=[True, True, False, True],
            ignore_index=True,
        )
        .drop(columns=["_category1_order", "_level_order"])
        [
            [
                category_1_column,
                category_2_column,
                "summary_level",
                "product_count",
                "median_price",
                "median_standardized_price_per_unit",
                "avg_weighted_rating",
                "median_opinions",
                "median_ingredients_count",
                "median_active_count",
                "avg_active_share",
                "median_active_position_score",
            ]
        ]
    )


def save_category_summary_outputs(products: pd.DataFrame) -> pd.DataFrame:
    """Zapisuje połączoną tabelę podsumowania kategorii."""
    category_summary = build_category_summary(products)
    category_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    category_summary.to_csv(
        category_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return category_summary


def build_brand_summary(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy tabelę metryk produktowych dla każdej marki."""
    require_columns(
        products,
        {
            brand_column,
            price_column,
            standardized_price_per_unit_column,
            weighted_rating_column,
            opinions_column,
            ingredients_count_column,
            active_count_column,
            active_share_column,
            active_position_score_column,
        },
        "processed products",
    )
    return (
        products.groupby(brand_column, dropna=False)
        .agg(
            product_count=("id", "count"),
            median_price=(price_column, "median"),
            median_standardized_price_per_unit=(
                standardized_price_per_unit_column,
                "median",
            ),
            avg_weighted_rating=(weighted_rating_column, "mean"),
            median_opinions=(opinions_column, "median"),
            median_ingredients_count=(ingredients_count_column, "median"),
            median_active_count=(active_count_column, "median"),
            avg_active_share=(active_share_column, "mean"),
            median_active_position_score=(active_position_score_column, "median"),
        )
        .reset_index()
        .round(
            {
                "median_price": 2,
                "median_standardized_price_per_unit": 2,
                "avg_weighted_rating": 2,
                "median_opinions": 2,
                "median_ingredients_count": 2,
                "median_active_count": 2,
                "avg_active_share": 2,
                "median_active_position_score": 2,
            }
        )
        .sort_values(
            ["product_count", brand_column],
            ascending=[False, True],
            ignore_index=True,
        )
    )


def save_brand_summary_outputs(products: pd.DataFrame) -> pd.DataFrame:
    """Zapisuje tabelę podsumowania marek."""
    brand_summary = build_brand_summary(products)
    brand_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    brand_summary.to_csv(
        brand_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return brand_summary


def build_ingredient_flag_summary(
    products: pd.DataFrame,
    ingredient_mapping: InciDictionary,
) -> pd.DataFrame:
    """Tworzy podsumowanie produktów dla każdej binarnej flagi składnikowej."""
    flag_columns = list(ingredient_family_columns)
    require_columns(
        products,
        set(flag_columns)
        | {
            price_column,
            standardized_price_per_unit_column,
            weighted_rating_column,
            opinions_column,
            ingredients_count_column,
            active_count_column,
            active_share_column,
            active_position_score_column,
        },
        "processed products",
    )

    product_total = len(products)
    rows = []
    for flag_column in flag_columns:
        matched_products = products.loc[products[flag_column].eq(1)]
        product_count = len(matched_products)
        positions = matched_products[ingredients_column].apply(
            find_ingredient_family_position,
            ingredient_mapping=ingredient_mapping,
            flag_column=flag_column,
        )
        rows.append(
            {
                "ingredient_flag": flag_column,
                "ingredient": flag_column.removeprefix("is_"),
                "product_count": product_count,
                "product_share": round(product_count / product_total * 100, 2),
                "avg_position": round(positions.mean(), 2),
                "median_position": round(positions.median(), 2),
                "median_price": round(matched_products[price_column].median(), 2),
                "median_standardized_price_per_unit": round(
                    matched_products[standardized_price_per_unit_column].median(),
                    2,
                ),
                "avg_weighted_rating": round(
                    matched_products[weighted_rating_column].mean(),
                    2,
                ),
                "median_opinions": round(
                    matched_products[opinions_column].median(),
                    2,
                ),
                "median_ingredients_count": round(
                    matched_products[ingredients_count_column].median(),
                    2,
                ),
                "median_active_count": round(
                    matched_products[active_count_column].median(),
                    2,
                ),
                "avg_active_share": round(
                    matched_products[active_share_column].mean(),
                    2,
                ),
                "median_active_position_score": round(
                    matched_products[active_position_score_column].median(),
                    2,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["product_count", "ingredient"],
        ascending=[False, True],
        ignore_index=True,
    ).drop(columns=["ingredient_flag"])


def save_ingredient_flag_summary_outputs(
    products: pd.DataFrame,
    ingredient_mapping: InciDictionary,
) -> pd.DataFrame:
    """Zapisuje tabelę podsumowania flag składnikowych."""
    ingredient_flag_summary = build_ingredient_flag_summary(
        products,
        ingredient_mapping,
    )
    ingredient_flag_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    ingredient_flag_summary.to_csv(
        ingredient_flag_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return ingredient_flag_summary


def median_or_missing(values: pd.Series) -> float | pd.NA:
    """Zwraca medianę serii albo brak, gdy porównywana grupa jest pusta."""
    if values.empty:
        return pd.NA
    return round(values.median(), 4)


def build_ingredient_family_group_comparison(products: pd.DataFrame) -> pd.DataFrame:
    """Porównuje produkty z daną rodziną składników aktywnych i bez niej."""
    flag_columns = list(ingredient_family_columns)
    comparison_metrics = {
        weighted_rating_column: "weighted_rating",
        log_opinions_column: "log_opinions",
        log_standardized_price_column: "log_standardized_price",
        ingredients_count_column: "ingredients_count",
        active_count_column: "active_count",
        active_share_column: "active_share",
        active_position_score_column: "active_position_score",
    }
    require_columns(
        products,
        set(flag_columns) | set(comparison_metrics),
        "processed products",
    )

    product_total = len(products)
    rows = []
    for flag_column in flag_columns:
        with_ingredient = products.loc[products[flag_column].eq(1)]
        without_ingredient = products.loc[products[flag_column].eq(0)]
        row = {
            "ingredient": flag_column.removeprefix("is_"),
            "with_count": len(with_ingredient),
            "without_count": len(without_ingredient),
            "with_share": round(len(with_ingredient) / product_total * 100, 2),
        }

        for source_column, metric_name in comparison_metrics.items():
            with_median = median_or_missing(with_ingredient[source_column])
            without_median = median_or_missing(without_ingredient[source_column])
            row[f"with_median_{metric_name}"] = with_median
            row[f"without_median_{metric_name}"] = without_median
            row[f"median_diff_{metric_name}"] = (
                pd.NA
                if pd.isna(with_median) or pd.isna(without_median)
                else round(with_median - without_median, 4)
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["with_count", "ingredient"],
        ascending=[False, True],
        ignore_index=True,
    )


def save_ingredient_family_group_comparison_outputs(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Zapisuje tabelę porównania produktów z rodziną składników i bez niej."""
    ingredient_family_group_comparison = build_ingredient_family_group_comparison(
        products
    )
    ingredient_family_group_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    ingredient_family_group_comparison.to_csv(
        ingredient_family_group_comparison_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return ingredient_family_group_comparison


def get_ingredient_family_test_metrics() -> dict[str, str]:
    """Zwraca metryki produktu używane w testach różnic między grupami."""
    return {
        weighted_rating_column: "weighted_rating",
        log_opinions_column: "log_opinions",
        log_standardized_price_column: "log_standardized_price",
        ingredients_count_column: "ingredients_count",
        active_count_column: "active_count",
        active_share_column: "active_share",
        active_position_score_column: "active_position_score",
        fragrance_count_column: "fragrance_count",
        fragrance_allergen_count_column: "fragrance_allergen_count",
        preservative_count_column: "preservative_count",
    }


def adjust_p_values_bh(p_values: pd.Series) -> pd.Series:
    """Koryguje p-value metodą Benjamini-Hochberg dla kontroli FDR."""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype="float64")
    valid = p_values.dropna().astype(float).sort_values()
    if valid.empty:
        return adjusted

    ranks = np.arange(1, len(valid) + 1)
    ranked_adjusted = valid * len(valid) / ranks
    ranked_adjusted = ranked_adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    adjusted.loc[ranked_adjusted.index] = ranked_adjusted
    return adjusted


def mannwhitneyu_two_sided(
    with_values: pd.Series,
    without_values: pd.Series,
) -> tuple[float, float]:
    """Wykonuje dwustronny test U Manna-Whitneya z obsługą starszych SciPy."""
    from scipy.stats import mannwhitneyu

    try:
        result = mannwhitneyu(
            with_values,
            without_values,
            alternative="two-sided",
            method="asymptotic",
        )
    except TypeError:
        result = mannwhitneyu(
            with_values,
            without_values,
            alternative="two-sided",
        )
    return float(result.statistic), float(result.pvalue)


def build_ingredient_family_mannwhitney_tests(
    products: pd.DataFrame,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """Testuje różnice metryk między produktami z daną rodziną składników i bez niej."""
    flag_columns = list(ingredient_family_columns)
    test_metrics = get_ingredient_family_test_metrics()
    require_columns(
        products,
        set(flag_columns) | set(test_metrics),
        "processed products",
    )

    rows = []
    for flag_column in flag_columns:
        flag_values = products[flag_column].fillna(0).astype(int)
        for source_column, metric_name in test_metrics.items():
            metric_values = pd.to_numeric(products[source_column], errors="coerce")
            with_values = metric_values.loc[flag_values.eq(1)].dropna()
            without_values = metric_values.loc[flag_values.eq(0)].dropna()

            with_median = median_or_missing(with_values)
            without_median = median_or_missing(without_values)
            median_difference = (
                np.nan
                if pd.isna(with_median) or pd.isna(without_median)
                else round(with_median - without_median, 4)
            )

            row = {
                "ingredient": flag_column.removeprefix("is_"),
                "metric": metric_name,
                "with_count": len(with_values),
                "without_count": len(without_values),
                "with_median": with_median,
                "without_median": without_median,
                "median_difference": median_difference,
                "u_statistic": np.nan,
                "p_value": np.nan,
                "rank_biserial_correlation": np.nan,
                "test_used": "mann_whitney_u",
                "test_status": "ok",
            }

            if len(with_values) < min_group_size or len(without_values) < min_group_size:
                row["test_status"] = "skipped_small_group"
                rows.append(row)
                continue

            u_statistic, p_value = mannwhitneyu_two_sided(
                with_values,
                without_values,
            )
            row["u_statistic"] = round(u_statistic, 4)
            row["p_value"] = p_value
            row["rank_biserial_correlation"] = round(
                2 * u_statistic / (len(with_values) * len(without_values)) - 1,
                4,
            )
            rows.append(row)

    tests = pd.DataFrame(rows)
    tests["p_value_fdr_bh"] = adjust_p_values_bh(tests["p_value"])
    tests["significant_fdr_0_05"] = tests["p_value_fdr_bh"].le(0.05)
    median_difference = pd.to_numeric(tests["median_difference"], errors="coerce")
    tests["direction"] = np.select(
        [
            median_difference.gt(0),
            median_difference.lt(0),
        ],
        [
            "higher_with_ingredient",
            "lower_with_ingredient",
        ],
        default="no_median_difference",
    )
    return tests.round(
        {
            "p_value": 6,
            "p_value_fdr_bh": 6,
            "rank_biserial_correlation": 4,
        }
    )


def save_ingredient_family_mannwhitney_test_outputs(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Zapisuje wyniki testów U Manna-Whitneya dla rodzin składników."""
    mannwhitney_tests = build_ingredient_family_mannwhitney_tests(products)
    ingredient_family_mannwhitney_tests_csv.parent.mkdir(parents=True, exist_ok=True)
    mannwhitney_tests.to_csv(
        ingredient_family_mannwhitney_tests_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return mannwhitney_tests


def get_regression_outcomes() -> dict[str, str]:
    """Zwraca zmienne objaśniane w modelach regresyjnych."""
    return {
        weighted_rating_column: "weighted_rating",
        log_opinions_column: "log_opinions",
        log_standardized_price_column: "log_standardized_price",
    }


def get_regression_continuous_controls(outcome_column: str) -> list[str]:
    """Dobiera zmienne kontrolne liczbowe odpowiednie dla danego wyniku."""
    controls = [
        ingredients_count_column,
        active_share_column,
        fragrance_allergen_count_column,
        preservative_count_column,
        drying_alcohol_count_column,
        is_parfum_column,
        is_colorant_column,
    ]
    if outcome_column != log_standardized_price_column:
        controls.insert(0, log_standardized_price_column)
    return controls


def collapse_rare_categories(
    values: pd.Series,
    min_count: int = 5,
    rare_label: str = "__rare__",
) -> pd.Series:
    """Łączy rzadkie poziomy kategorii, żeby ograniczyć przeuczenie modelu."""
    normalized = values.fillna("__missing__").astype(str)
    counts = normalized.value_counts()
    return normalized.where(normalized.map(counts).ge(min_count), rare_label)


def prepare_regression_design(
    products: pd.DataFrame,
    outcome_column: str,
) -> tuple[pd.Series, pd.DataFrame, list[str], list[str]]:
    """Buduje macierz modelu regresyjnego z flagami składników i kontrolami."""
    ingredient_flags = list(ingredient_family_columns)
    continuous_controls = get_regression_continuous_controls(outcome_column)
    categorical_controls = [category_2_column, brand_column]
    required_columns = (
        {outcome_column}
        | set(ingredient_flags)
        | set(continuous_controls)
        | set(categorical_controls)
    )
    require_columns(products, required_columns, "processed products")

    model_data = products[
        [outcome_column]
        + ingredient_flags
        + continuous_controls
        + categorical_controls
    ].copy()
    numeric_columns = [outcome_column] + ingredient_flags + continuous_controls
    for column in numeric_columns:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=numeric_columns
    )

    active_flags = [
        flag_column
        for flag_column in ingredient_flags
        if model_data[flag_column].nunique(dropna=True) > 1
    ]
    skipped_flags = [
        flag_column for flag_column in ingredient_flags if flag_column not in active_flags
    ]

    category_data = model_data[categorical_controls].copy()
    category_data[category_2_column] = category_data[category_2_column].fillna(
        "__missing__"
    ).astype(str)
    category_data[brand_column] = collapse_rare_categories(category_data[brand_column])

    numeric_design = model_data[active_flags + continuous_controls].astype(float)
    categorical_design = pd.get_dummies(
        category_data,
        columns=categorical_controls,
        drop_first=True,
        dtype=float,
    )
    intercept = pd.Series(1.0, index=model_data.index, name="intercept")
    design = pd.concat([intercept, numeric_design, categorical_design], axis=1)
    return model_data[outcome_column].astype(float), design, active_flags, skipped_flags


def fit_ols_hc3(
    outcome: pd.Series,
    design: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Dopasowuje OLS i liczy odporne błędy standardowe HC3."""
    from scipy.stats import t

    x = design.to_numpy(dtype=float)
    y = outcome.to_numpy(dtype=float)
    n_obs, n_parameters = x.shape
    rank = int(np.linalg.matrix_rank(x))
    residual_df = max(n_obs - rank, 1)

    xtx_inv = np.linalg.pinv(x.T @ x)
    coefficients = xtx_inv @ x.T @ y
    fitted = x @ coefficients
    residuals = y - fitted

    total_sum_squares = float(((y - y.mean()) ** 2).sum())
    residual_sum_squares = float((residuals**2).sum())
    residual_variance_mle = max(residual_sum_squares / n_obs, np.finfo(float).eps)
    log_likelihood = float(
        -0.5
        * n_obs
        * (np.log(2 * np.pi) + np.log(residual_variance_mle) + 1)
    )
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    adjusted_r_squared = (
        1 - (1 - r_squared) * (n_obs - 1) / residual_df
        if not pd.isna(r_squared)
        else np.nan
    )

    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    leverage_denominator = np.clip(1 - leverage, 1e-8, None)
    hc3_scale = (residuals / leverage_denominator) ** 2
    robust_covariance = xtx_inv @ (x.T @ (x * hc3_scale[:, None])) @ xtx_inv
    robust_variance = np.clip(np.diag(robust_covariance), 0, None)
    robust_standard_errors = np.sqrt(robust_variance)

    t_values = np.divide(
        coefficients,
        robust_standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=robust_standard_errors > 0,
    )
    p_values = 2 * t.sf(np.abs(t_values), residual_df)
    critical_value = t.ppf(0.975, residual_df)

    coefficient_table = pd.DataFrame(
        {
            "coefficient": coefficients,
            "std_error_hc3": robust_standard_errors,
            "t_value": t_values,
            "p_value": p_values,
            "ci_95_low": coefficients - critical_value * robust_standard_errors,
            "ci_95_high": coefficients + critical_value * robust_standard_errors,
        },
        index=design.columns,
    )
    model_info = {
        "n_obs": n_obs,
        "n_parameters": n_parameters,
        "residual_df": residual_df,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "aic": 2 * rank - 2 * log_likelihood,
        "bic": np.log(n_obs) * rank - 2 * log_likelihood,
        "outcome_std": float(outcome.std(ddof=0)),
    }
    return coefficient_table, model_info


def build_ingredient_family_regression_models(products: pd.DataFrame) -> pd.DataFrame:
    """Buduje kontrolowane modele OLS dla rodzin składników aktywnych."""
    rows = []
    outcome_labels = get_regression_outcomes()
    ingredient_flags = list(ingredient_family_columns)
    for outcome_column, outcome_name in outcome_labels.items():
        outcome, design, active_flags, skipped_flags = prepare_regression_design(
            products,
            outcome_column,
        )
        coefficient_table, model_info = fit_ols_hc3(outcome, design)
        for flag_column in ingredient_flags:
            ingredient = flag_column.removeprefix("is_")
            if flag_column in skipped_flags:
                rows.append(
                    {
                        "outcome": outcome_name,
                        "ingredient": ingredient,
                        "coefficient": np.nan,
                        "standardized_coefficient": np.nan,
                        "std_error_hc3": np.nan,
                        "t_value": np.nan,
                        "p_value": np.nan,
                        "ci_95_low": np.nan,
                        "ci_95_high": np.nan,
                        "approx_percent_change": np.nan,
                        "model_status": "skipped_no_variation",
                        **model_info,
                    }
                )
                continue

            stats = coefficient_table.loc[flag_column]
            coefficient = float(stats["coefficient"])
            outcome_std = model_info["outcome_std"]
            rows.append(
                {
                    "outcome": outcome_name,
                    "ingredient": ingredient,
                    "coefficient": coefficient,
                    "standardized_coefficient": (
                        coefficient / outcome_std if outcome_std > 0 else np.nan
                    ),
                    "std_error_hc3": float(stats["std_error_hc3"]),
                    "t_value": float(stats["t_value"]),
                    "p_value": float(stats["p_value"]),
                    "ci_95_low": float(stats["ci_95_low"]),
                    "ci_95_high": float(stats["ci_95_high"]),
                    "approx_percent_change": (
                        (np.exp(coefficient) - 1) * 100
                        if outcome_name
                        in {"log_opinions", "log_standardized_price"}
                        else np.nan
                    ),
                    "model_status": "ok",
                    **model_info,
                }
            )

    regression_results = pd.DataFrame(rows)
    regression_results["p_value_fdr_bh"] = np.nan
    for outcome_name in regression_results["outcome"].dropna().unique():
        mask = (
            regression_results["outcome"].eq(outcome_name)
            & regression_results["model_status"].eq("ok")
        )
        regression_results.loc[mask, "p_value_fdr_bh"] = adjust_p_values_bh(
            regression_results.loc[mask, "p_value"]
        )
    regression_results["significant_fdr_0_05"] = regression_results[
        "p_value_fdr_bh"
    ].le(0.05)
    regression_results["direction"] = np.select(
        [
            regression_results["coefficient"].gt(0),
            regression_results["coefficient"].lt(0),
        ],
        [
            "positive",
            "negative",
        ],
        default="no_effect",
    )
    return regression_results.round(
        {
            "coefficient": 6,
            "standardized_coefficient": 4,
            "std_error_hc3": 6,
            "t_value": 4,
            "p_value": 6,
            "p_value_fdr_bh": 6,
            "ci_95_low": 6,
            "ci_95_high": 6,
            "approx_percent_change": 2,
            "r_squared": 4,
            "adjusted_r_squared": 4,
            "outcome_std": 6,
        }
    )


def save_ingredient_family_regression_outputs(products: pd.DataFrame) -> pd.DataFrame:
    """Zapisuje kontrolowane modele regresyjne dla rodzin składników."""
    regression_results = build_ingredient_family_regression_models(products)
    ingredient_family_regression_models_csv.parent.mkdir(parents=True, exist_ok=True)
    regression_results.to_csv(
        ingredient_family_regression_models_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return regression_results


def get_weighted_rating_formula_numeric_columns() -> list[str]:
    """Zwraca cechy formuły używane w głównym modelu oceny ważonej."""
    return [
        log_standardized_price_column,
        ingredients_count_column,
        active_share_column,
        fragrance_allergen_count_column,
        preservative_count_column,
        drying_alcohol_count_column,
        is_parfum_column,
        is_colorant_column,
    ]


def get_weighted_rating_regression_specs() -> dict[str, dict[str, list[str]]]:
    """Definiuje trzy główne, zagnieżdżone modele dla weighted_rating."""
    base_numeric = [log_standardized_price_column]
    formula_numeric = get_weighted_rating_formula_numeric_columns()
    return {
        "base": {
            "numeric": base_numeric,
            "categorical": [category_2_column, brand_column],
            "flags": [],
        },
        "formula": {
            "numeric": formula_numeric,
            "categorical": [category_2_column, brand_column],
            "flags": [],
        },
        "ingredient": {
            "numeric": formula_numeric,
            "categorical": [category_2_column, brand_column],
            "flags": list(ingredient_family_columns),
        },
    }


def build_weighted_rating_active_sensitivity_outputs(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Testuje osobno liczbę i pozycję aktywnych przy stałych kontrolach.

    ``active_count``, ``active_share`` i ``active_position_score`` są ze sobą
    konstrukcyjnie powiązane. Każdą z dwóch dodatkowych miar testujemy więc w
    osobnym modelu, zastępując nią ``active_share`` z głównego modelu formuły.
    Dzięki temu współczynnik opisuje związek danej miary z oceną ważoną przy
    tym samym zestawie kontroli, bez niestabilności spowodowanej kolinearnością.
    """
    formula_controls = [
        column
        for column in get_weighted_rating_formula_numeric_columns()
        if column != active_share_column
    ]
    sensitivity_specs = {
        "active_count_sensitivity": active_count_column,
        "active_position_score_sensitivity": active_position_score_column,
    }
    rows = []

    for model_name, focal_predictor in sensitivity_specs.items():
        model_spec = {
            "numeric": formula_controls + [focal_predictor],
            "categorical": [category_2_column, brand_column],
            "flags": [],
        }
        outcome, design, _ = prepare_weighted_rating_regression_design(
            products,
            model_name,
            model_spec,
        )
        coefficient_table, model_info = fit_ols_hc3(outcome, design)
        coefficients = summarize_ols_coefficients(
            coefficient_table,
            model_info,
            outcome,
            design,
            model_name,
        )
        focal_row = coefficients.loc[coefficients["term"].eq(focal_predictor)]
        if focal_row.empty:
            raise ValueError(f"Missing focal predictor in sensitivity model: {focal_predictor}")

        result = focal_row.iloc[0].to_dict()
        result["focal_predictor"] = focal_predictor
        result["control_set"] = "formula_controls_replacing_active_share"
        rows.append(result)

    sensitivity = pd.DataFrame(rows)
    sensitivity["p_value_fdr_bh"] = adjust_p_values_bh(sensitivity["p_value"])
    sensitivity["significant_fdr_0_05"] = sensitivity["p_value_fdr_bh"].le(0.05)
    return sensitivity.round(
        {
            "coefficient": 6,
            "std_error_hc3": 6,
            "t_value": 4,
            "p_value": 6,
            "p_value_fdr_bh": 6,
            "ci_95_low": 6,
            "ci_95_high": 6,
            "predictor_std": 6,
            "standardized_coefficient": 4,
            "r_squared": 4,
            "adjusted_r_squared": 4,
            "outcome_std": 6,
        }
    )


def save_weighted_rating_active_sensitivity_outputs(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Zapisuje modele wrażliwości dla miar aktywności formuły."""
    sensitivity = build_weighted_rating_active_sensitivity_outputs(products)
    weighted_rating_active_sensitivity_csv.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(
        weighted_rating_active_sensitivity_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return sensitivity


def prepare_weighted_rating_regression_design(
    products: pd.DataFrame,
    model_name: str,
    model_spec: dict[str, list[str]],
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Buduje macierz wybranego modelu końcowego dla weighted_rating."""
    numeric_columns = model_spec["numeric"] + model_spec["flags"]
    categorical_columns = model_spec["categorical"]
    required_columns = (
        {weighted_rating_column}
        | set(numeric_columns)
        | set(categorical_columns)
    )
    require_columns(products, required_columns, "processed products")

    model_data = products[
        [weighted_rating_column] + numeric_columns + categorical_columns
    ].copy()
    for column in [weighted_rating_column] + numeric_columns:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[weighted_rating_column] + numeric_columns
    )

    active_numeric_columns = [
        column
        for column in numeric_columns
        if model_data[column].nunique(dropna=True) > 1
    ]
    category_data = model_data[categorical_columns].copy()
    category_data[category_2_column] = category_data[category_2_column].fillna(
        "__missing__"
    ).astype(str)
    category_data[brand_column] = collapse_rare_categories(category_data[brand_column])

    numeric_design = model_data[active_numeric_columns].astype(float)
    categorical_design = pd.get_dummies(
        category_data,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    intercept = pd.Series(1.0, index=model_data.index, name="intercept")
    design = pd.concat([intercept, numeric_design, categorical_design], axis=1)
    design.columns = [str(column) for column in design.columns]
    if model_name == "ingredient":
        missing_flags = sorted(set(model_spec["flags"]) - set(active_numeric_columns))
        if missing_flags:
            print(f"Skipped no-variation ingredient flags: {missing_flags}")
    return model_data[weighted_rating_column].astype(float), design, active_numeric_columns


def summarize_ols_coefficients(
    coefficient_table: pd.DataFrame,
    model_info: dict[str, float],
    outcome: pd.Series,
    design: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Dodaje metadane modelu i standaryzowane współczynniki do tabeli OLS."""
    outcome_std = float(outcome.std(ddof=0))
    predictor_std = design.std(ddof=0).replace(0, np.nan)
    result = coefficient_table.reset_index(names="term")
    result.insert(0, "model", model_name)
    result.insert(1, "outcome", weighted_rating_column)
    result["predictor_std"] = result["term"].map(predictor_std)
    result["standardized_coefficient"] = (
        result["coefficient"] * result["predictor_std"] / outcome_std
    )
    result.loc[result["term"].eq("intercept"), "standardized_coefficient"] = np.nan
    for key, value in model_info.items():
        result[key] = value
    return result


def build_weighted_rating_regression_outputs(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Buduje trzy końcowe modele regresyjne dla weighted_rating."""
    coefficient_frames = []
    summary_rows = []
    specs = get_weighted_rating_regression_specs()
    for model_name, model_spec in specs.items():
        outcome, design, active_predictors = prepare_weighted_rating_regression_design(
            products,
            model_name,
            model_spec,
        )
        coefficient_table, model_info = fit_ols_hc3(outcome, design)
        coefficients = summarize_ols_coefficients(
            coefficient_table,
            model_info,
            outcome,
            design,
            model_name,
        )
        coefficients["term_type"] = np.select(
            [
                coefficients["term"].eq("intercept"),
                coefficients["term"].isin(ingredient_family_columns),
                coefficients["term"].isin(model_spec["numeric"]),
                coefficients["term"].str.startswith(f"{category_2_column}_"),
                coefficients["term"].str.startswith(f"{brand_column}_"),
            ],
            [
                "intercept",
                "ingredient_flag",
                "numeric_control",
                "category2_control",
                "brand_control",
            ],
            default="other_control",
        )
        coefficient_frames.append(coefficients)
        summary_rows.append(
            {
                "model": model_name,
                "outcome": weighted_rating_column,
                "n_obs": model_info["n_obs"],
                "n_parameters": model_info["n_parameters"],
                "residual_df": model_info["residual_df"],
                "r_squared": model_info["r_squared"],
                "adjusted_r_squared": model_info["adjusted_r_squared"],
                "aic": model_info.get("aic", np.nan),
                "bic": model_info.get("bic", np.nan),
                "predictor_count": len(active_predictors),
                "ingredient_flag_count": len(
                    set(active_predictors) & set(ingredient_family_columns)
                ),
            }
        )

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    ingredient_mask = (
        coefficients["model"].eq("ingredient")
        & coefficients["term"].isin(ingredient_family_columns)
    )
    coefficients["p_value_fdr_bh"] = np.nan
    coefficients.loc[ingredient_mask, "p_value_fdr_bh"] = adjust_p_values_bh(
        coefficients.loc[ingredient_mask, "p_value"]
    )
    coefficients["significant_fdr_0_05"] = coefficients["p_value_fdr_bh"].le(0.05)
    coefficients["direction"] = np.select(
        [
            coefficients["coefficient"].gt(0),
            coefficients["coefficient"].lt(0),
        ],
        ["positive", "negative"],
        default="no_effect",
    )

    model_summary = pd.DataFrame(summary_rows)
    model_summary["delta_adjusted_r_squared"] = (
        model_summary["adjusted_r_squared"]
        - model_summary["adjusted_r_squared"].shift(1)
    )
    ingredient_effects = coefficients.loc[ingredient_mask].copy()
    ingredient_effects["ingredient"] = ingredient_effects["term"].str.removeprefix(
        "is_"
    )
    return (
        coefficients.round(
            {
                "coefficient": 6,
                "std_error_hc3": 6,
                "t_value": 4,
                "p_value": 6,
                "p_value_fdr_bh": 6,
                "ci_95_low": 6,
                "ci_95_high": 6,
                "predictor_std": 6,
                "standardized_coefficient": 4,
                "r_squared": 4,
                "adjusted_r_squared": 4,
                "outcome_std": 6,
            }
        ),
        model_summary.round(
            {
                "r_squared": 4,
                "adjusted_r_squared": 4,
                "delta_adjusted_r_squared": 4,
                "aic": 2,
                "bic": 2,
            }
        ),
        ingredient_effects.round(
            {
                "coefficient": 6,
                "std_error_hc3": 6,
                "t_value": 4,
                "p_value": 6,
                "p_value_fdr_bh": 6,
                "ci_95_low": 6,
                "ci_95_high": 6,
                "predictor_std": 6,
                "standardized_coefficient": 4,
                "r_squared": 4,
                "adjusted_r_squared": 4,
                "outcome_std": 6,
            }
        ),
    )


def save_weighted_rating_regression_outputs(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Zapisuje końcową analizę regresji dla weighted_rating."""
    coefficients, model_summary, ingredient_effects = (
        build_weighted_rating_regression_outputs(products)
    )
    weighted_rating_regression_coefficients_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    coefficients.to_csv(
        weighted_rating_regression_coefficients_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    model_summary.to_csv(
        weighted_rating_regression_model_summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    ingredient_effects.to_csv(
        weighted_rating_regression_ingredient_effects_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return coefficients, model_summary, ingredient_effects


def build_spearman_correlation_matrix(
    products: pd.DataFrame,
    correlation_columns: dict[str, str],
) -> pd.DataFrame:
    """Tworzy macierz korelacji rang Spearmana dla wskazanych zmiennych."""
    require_columns(
        products,
        set(correlation_columns),
        "processed products",
    )
    return (
        products[list(correlation_columns)]
        .rename(columns=correlation_columns)
        .corr(method="spearman")
        .round(4)
    )


def build_product_characteristics_correlation(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy ogólną macierz korelacji charakterystyki produktu."""
    return build_spearman_correlation_matrix(
        products,
        {
            weighted_rating_column: "weighted_rating",
            log_opinions_column: "log_opinions",
            log_standardized_price_column: "log_standardized_price",
            ingredients_count_column: "ingredients_count",
            active_count_column: "active_count",
            active_position_score_column: "active_position_score",
        },
    )


def build_product_formula_correlation(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy macierz korelacji produktu rozszerzoną o zapachy i konserwanty."""
    return build_spearman_correlation_matrix(
        products,
        {
            weighted_rating_column: "weighted_rating",
            log_opinions_column: "log_opinions",
            log_standardized_price_column: "log_standardized_price",
            ingredients_count_column: "ingredients_count",
            active_count_column: "active_count",
            active_position_score_column: "active_position_score",
            fragrance_count_column: "fragrance_count",
            fragrance_allergen_count_column: "fragrance_allergen_count",
            preservative_count_column: "preservative_count",
            is_alcohol_denat_column: "is_alcohol_denat",
            drying_alcohol_count_column: "drying_alcohol_count",
            is_drying_alcohol_column: "is_drying_alcohol",
            is_parfum_column: "is_parfum",
            is_colorant_column: "is_colorant",
        },
    )


def build_active_intensity_correlation(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy macierz korelacji intensywności składników aktywnych."""
    return build_spearman_correlation_matrix(
        products,
        {
            weighted_rating_column: "weighted_rating",
            log_opinions_column: "log_opinions",
            log_standardized_price_column: "log_standardized_price",
            active_count_column: "active_count",
            active_share_column: "active_share",
            active_position_score_column: "active_position_score",
        },
    )


def build_ingredient_family_correlation(products: pd.DataFrame) -> pd.DataFrame:
    """Tworzy eksploracyjną macierz korelacji rodzin składników aktywnych."""
    correlation_columns = {
        weighted_rating_column: "weighted_rating",
        log_opinions_column: "log_opinions",
        log_standardized_price_column: "log_standardized_price",
        ingredients_count_column: "ingredients_count",
        active_count_column: "active_count",
        active_position_score_column: "active_position_score",
    }
    correlation_columns.update(
        {flag_column: flag_column for flag_column in ingredient_family_columns}
    )
    return build_spearman_correlation_matrix(products, correlation_columns)


def save_correlation_matrix_outputs(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Zapisuje macierze korelacji Spearmana."""
    product_characteristics = build_product_characteristics_correlation(products)
    product_formula = build_product_formula_correlation(products)
    active_intensity = build_active_intensity_correlation(products)
    ingredient_family = build_ingredient_family_correlation(products)

    product_characteristics_correlation_csv.parent.mkdir(parents=True, exist_ok=True)
    product_characteristics.to_csv(
        product_characteristics_correlation_csv,
        index=True,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    product_formula.to_csv(
        product_formula_correlation_csv,
        index=True,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    active_intensity.to_csv(
        active_intensity_correlation_csv,
        index=True,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    ingredient_family.to_csv(
        ingredient_family_correlation_csv,
        index=True,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return product_characteristics, product_formula, active_intensity, ingredient_family



def slugify_filename(value: str) -> str:
    """Zamienia nazwę kategorii na stabilny fragment nazwy pliku."""
    value = value.replace("ł", "l").replace("Ł", "L")
    slug = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


def save_category_specific_analysis_outputs(
    products: pd.DataFrame,
    categories: tuple[str, ...],
    output_dir: Path = category_specific_summary_dir,
) -> dict[str, dict[str, Path]]:
    """Zapisuje korelacje i porównanie rodzin składników dla wybranych kategorii."""
    require_columns(products, {category_1_column}, "processed products")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, dict[str, Path]] = {}

    for category in categories:
        category_products = products.loc[products[category_1_column].eq(category)]
        if category_products.empty:
            continue

        slug = slugify_filename(category)
        product_characteristics = build_product_characteristics_correlation(
            category_products
        )
        ingredient_family = build_ingredient_family_correlation(category_products)
        ingredient_family_group_comparison = (
            build_ingredient_family_group_comparison(category_products)
        )
        ingredient_family_mannwhitney_tests = (
            build_ingredient_family_mannwhitney_tests(category_products)
        )

        paths = {
            "product_characteristics_correlation": (
                output_dir / f"{slug}_product_characteristics_correlation.csv"
            ),
            "ingredient_family_correlation": (
                output_dir / f"{slug}_ingredient_family_correlation.csv"
            ),
            "ingredient_family_group_comparison": (
                output_dir / f"{slug}_ingredient_family_group_comparison.csv"
            ),
            "ingredient_family_mannwhitney_tests": (
                output_dir / f"{slug}_ingredient_family_mannwhitney_tests.csv"
            ),
        }
        product_characteristics.to_csv(
            paths["product_characteristics_correlation"],
            index=True,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        ingredient_family.to_csv(
            paths["ingredient_family_correlation"],
            index=True,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        ingredient_family_group_comparison.to_csv(
            paths["ingredient_family_group_comparison"],
            index=False,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        ingredient_family_mannwhitney_tests.to_csv(
            paths["ingredient_family_mannwhitney_tests"],
            index=False,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        saved_paths[category] = paths

    return saved_paths


def save_subcategory_specific_analysis_outputs(
    products: pd.DataFrame,
    subcategories: tuple[str, ...],
    output_dir: Path = subcategory_specific_summary_dir,
) -> dict[str, dict[str, Path]]:
    """Zapisuje korelacje i porównanie rodzin składników dla wybranych podkategorii."""
    require_columns(products, {category_2_column}, "processed products")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, dict[str, Path]] = {}

    for subcategory in subcategories:
        subcategory_products = products.loc[
            products[category_2_column].eq(subcategory)
        ]
        if subcategory_products.empty:
            continue

        slug = slugify_filename(subcategory)
        product_characteristics = build_product_characteristics_correlation(
            subcategory_products
        )
        ingredient_family = build_ingredient_family_correlation(subcategory_products)
        ingredient_family_group_comparison = (
            build_ingredient_family_group_comparison(subcategory_products)
        )

        paths = {
            "product_characteristics_correlation": (
                output_dir / f"{slug}_product_characteristics_correlation.csv"
            ),
            "ingredient_family_correlation": (
                output_dir / f"{slug}_ingredient_family_correlation.csv"
            ),
            "ingredient_family_group_comparison": (
                output_dir / f"{slug}_ingredient_family_group_comparison.csv"
            ),
        }
        product_characteristics.to_csv(
            paths["product_characteristics_correlation"],
            index=True,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        ingredient_family.to_csv(
            paths["ingredient_family_correlation"],
            index=True,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        ingredient_family_group_comparison.to_csv(
            paths["ingredient_family_group_comparison"],
            index=False,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        saved_paths[subcategory] = paths

    return saved_paths


# Segmentacja globalna


def prepare_segmentation_dataset(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Przygotowuje cechy do globalnej segmentacji produktów."""
    from sklearn.preprocessing import StandardScaler

    feature_columns = list(segmentation_continuous_columns) + list(
        segmentation_binary_columns
    )
    require_columns(products, set(feature_columns) | {"id"}, "processed products")

    raw_features = products[feature_columns].apply(pd.to_numeric, errors="coerce")
    if raw_features.isna().any().any():
        missing = raw_features.columns[raw_features.isna().any()].tolist()
        raise ValueError(f"Segmentation features contain missing values: {missing}")

    scaler = StandardScaler()
    model_features = pd.DataFrame(
        scaler.fit_transform(raw_features),
        columns=feature_columns,
        index=products.index,
    )
    model_features[list(segmentation_binary_columns)] = (
        model_features[list(segmentation_binary_columns)] * binary_feature_weight
    )

    segmentation_dataset = pd.concat(
        [
            products[["id", brand_column, "name", category_1_column, category_2_column]],
            raw_features.add_prefix("raw_"),
            model_features.add_prefix("model_"),
        ],
        axis=1,
    )
    return segmentation_dataset, model_features, feature_columns


def evaluate_kmeans_models(model_features: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Testuje kilka wartości k i wybiera wariant do interpretacji."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = []
    labels_by_k = {}
    for k in segmentation_k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(model_features)
        labels_by_k[k] = labels
        cluster_sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "k": k,
                "inertia": round(model.inertia_, 4),
                "silhouette_score": round(
                    silhouette_score(model_features, labels),
                    4,
                ),
                "min_cluster_size": int(cluster_sizes.min()),
                "min_cluster_share": round(cluster_sizes.min() / len(labels), 4),
                "max_cluster_size": int(cluster_sizes.max()),
            }
        )

    metrics = pd.DataFrame(rows)
    interpretable = metrics.loc[metrics["min_cluster_share"].ge(0.05)]
    selection_pool = interpretable if not interpretable.empty else metrics
    selected_k = int(
        selection_pool.sort_values(
            ["silhouette_score", "min_cluster_share"],
            ascending=[False, False],
        ).iloc[0]["k"]
    )
    metrics["selected"] = metrics["k"].eq(selected_k)
    return metrics, selected_k


def assign_kmeans_segments(
    model_features: pd.DataFrame,
    selected_k: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Dopasowuje finalny KMeans i zwraca segmenty oraz współrzędne PCA."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    model = KMeans(n_clusters=selected_k, random_state=42, n_init=50)
    labels = model.fit_predict(model_features)
    pca = PCA(n_components=2, random_state=42)
    pca_values = pca.fit_transform(model_features)
    pca_frame = pd.DataFrame(
        pca_values,
        columns=["pca_1", "pca_2"],
        index=model_features.index,
    )
    pca_frame["pca_explained_variance_1"] = round(
        pca.explained_variance_ratio_[0],
        4,
    )
    pca_frame["pca_explained_variance_2"] = round(
        pca.explained_variance_ratio_[1],
        4,
    )
    return pd.Series(labels, index=model_features.index, name=segment_column), pca_frame


def most_common_value(values: pd.Series) -> object:
    """Zwraca najczęstszą wartość w segmencie."""
    if values.empty:
        return pd.NA
    return values.value_counts(dropna=False).index[0]


def build_segment_summary(segmentation_results: pd.DataFrame) -> pd.DataFrame:
    """Buduje opisowy profil segmentów."""
    product_total = len(segmentation_results)
    rows = []
    for segment, segment_products in segmentation_results.groupby(segment_column):
        row = {
            segment_column: segment,
            "product_count": len(segment_products),
            "product_share": round(len(segment_products) / product_total * 100, 2),
            "top_category1": most_common_value(segment_products[category_1_column]),
            "top_category2": most_common_value(segment_products[category_2_column]),
            "median_price": round(segment_products[price_column].median(), 2),
            "median_standardized_price_per_unit": round(
                segment_products[standardized_price_per_unit_column].median(),
                2,
            ),
            "median_log_standardized_price": round(
                segment_products[log_standardized_price_column].median(),
                4,
            ),
            "avg_weighted_rating": round(
                segment_products[weighted_rating_column].mean(),
                4,
            ),
            "median_opinions": round(segment_products[opinions_column].median(), 2),
            "median_log_opinions": round(
                segment_products[log_opinions_column].median(),
                4,
            ),
            "median_ingredients_count": round(
                segment_products[ingredients_count_column].median(),
                2,
            ),
            "avg_active_share": round(segment_products[active_share_column].mean(), 2),
            "median_active_position_score": round(
                segment_products[active_position_score_column].median(),
                2,
            ),
            "median_fragrance_count": round(
                segment_products[fragrance_count_column].median(),
                2,
            ),
            "median_fragrance_allergen_count": round(
                segment_products[fragrance_allergen_count_column].median(),
                2,
            ),
            "median_preservative_count": round(
                segment_products[preservative_count_column].median(),
                2,
            ),
            "median_drying_alcohol_count": round(
                segment_products[drying_alcohol_count_column].median(),
                2,
            ),
        }
        for column in segmentation_binary_columns:
            row[f"share_{column}"] = round(segment_products[column].mean() * 100, 2)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(segment_column, ignore_index=True)
    label_map = build_segment_labels(summary)
    summary.insert(
        summary.columns.get_loc(segment_column) + 1,
        segment_label_column,
        summary[segment_column].map(label_map),
    )
    return summary


def build_segment_labels(segment_summary: pd.DataFrame) -> dict[int, str]:
    """Nadaje segmentom krótkie etykiety interpretacyjne na podstawie profilu."""
    if segment_summary.empty:
        return {}

    labels: dict[int, str] = {}
    remaining = set(segment_summary[segment_column].tolist())

    # Segment bazowy/popularny to najtańsza grupa w przeliczeniu na jednostkę.
    basic_segment = int(
        segment_summary.sort_values(
            ["median_log_standardized_price", "median_active_position_score"],
            ascending=[True, True],
        ).iloc[0][segment_column]
    )
    labels[basic_segment] = "basic_popular"
    remaining.discard(basic_segment)

    if remaining:
        # Segment złożony ma najdłuższe formuły i zwykle więcej zapachów/konserwantów.
        complex_candidates = segment_summary.loc[
            segment_summary[segment_column].isin(remaining)
        ]
        complex_segment = int(
            complex_candidates.sort_values(
                [
                    "median_ingredients_count",
                    "median_fragrance_count",
                    "median_fragrance_allergen_count",
                    "median_preservative_count",
                ],
                ascending=[False, False, False, False],
            ).iloc[0][segment_column]
        )
        labels[complex_segment] = "complex_sensory"
        remaining.discard(complex_segment)

    if remaining:
        # Pozostały segment traktujemy jako bardziej aktywny/premium.
        active_candidates = segment_summary.loc[
            segment_summary[segment_column].isin(remaining)
        ].sort_values(
            ["avg_active_share", "median_active_position_score"],
            ascending=[False, False],
        )
        active_segment = int(active_candidates.iloc[0][segment_column])
        labels[active_segment] = "active_premium"
        remaining.discard(active_segment)

    if remaining:
        fragrance_candidates = segment_summary.loc[
            segment_summary[segment_column].isin(remaining)
        ].sort_values(
            [
                "median_fragrance_count",
                "median_fragrance_allergen_count",
                "median_preservative_count",
            ],
            ascending=[False, False, False],
        )
        for index, segment in enumerate(fragrance_candidates[segment_column].tolist(), 1):
            label = "fragrance_preservative" if index == 1 else f"segment_{int(segment)}"
            labels[int(segment)] = label

    return labels



def save_segmentation_outputs(
    products: pd.DataFrame,
    output_dir: Path,
    dataset_csv: Path,
    results_csv: Path,
    summary_csv: Path,
    metrics_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wykonuje i zapisuje segmentację KMeans dla przekazanego zbioru produktów."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dataset, model_features, feature_columns = (
        prepare_segmentation_dataset(products)
    )
    metrics, selected_k = evaluate_kmeans_models(model_features)
    segments, pca_frame = assign_kmeans_segments(model_features, selected_k)

    segmentation_results = pd.concat(
        [products.copy(), segments, pca_frame],
        axis=1,
    )
    segment_summary = build_segment_summary(segmentation_results)
    segmentation_dataset[segment_column] = segments
    label_map = segment_summary.set_index(segment_column)[segment_label_column]
    segmentation_results[segment_label_column] = segments.map(label_map)
    segmentation_dataset[segment_label_column] = segments.map(label_map)

    segmentation_dataset.to_csv(
        dataset_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    segmentation_results.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    segment_summary.to_csv(
        summary_csv,
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
    return segmentation_dataset, segmentation_results, segment_summary, metrics


def save_global_segmentation_outputs(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wykonuje i zapisuje globalną segmentację KMeans produktów."""
    return save_segmentation_outputs(
        products,
        segmentation_dir,
        segmentation_dataset_csv,
        segmentation_results_csv,
        segment_summary_csv,
        segmentation_model_metrics_csv,
    )


def save_face_care_segmentation_outputs(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wykonuje i zapisuje segmentację tylko dla kategorii Pielęgnacja twarzy."""
    face_care_products = products.loc[
        products[category_1_column].eq(face_care_category_1)
    ].copy()
    return save_segmentation_outputs(
        face_care_products,
        face_care_segmentation_dir,
        face_care_segmentation_dataset_csv,
        face_care_segmentation_results_csv,
        face_care_segment_summary_csv,
        face_care_segmentation_model_metrics_csv,
    )


def save_cleansing_segmentation_outputs(
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wykonuje i zapisuje segmentację tylko dla oczyszczania i demakijażu twarzy."""
    cleansing_products = products.loc[
        products[category_1_column].eq(cleansing_category_1)
    ].copy()
    return save_segmentation_outputs(
        cleansing_products,
        cleansing_segmentation_dir,
        cleansing_segmentation_dataset_csv,
        cleansing_segmentation_results_csv,
        cleansing_segment_summary_csv,
        cleansing_segmentation_model_metrics_csv,
    )


# Uruchomienie całej analizy


def build_processed_products(
    ingredient_mapping: InciDictionary,
    kosmopedia_functions: dict[str, frozenset[str]],
    manual_functions: dict[str, frozenset[str]],
    source_csv: Path = input_csv,
) -> pd.DataFrame:
    """Buduje products_analysis_processed z jednego wspólnego słownika INCI."""
    products = pd.read_csv(source_csv)
    require_columns(products, {ingredients_column}, "products")

    # Najpierw zawężamy dane i porządkujemy metadane produktu. Oryginalny tekst
    # ingredients pozostaje bez zmian, bo jest podstawą audytu mapowania.
    products = apply_product_scope(products)
    products = normalize_product_columns(products)

    # Ten sam wynik mapowania zasila licznik składników.
    mapped_values = products[ingredients_column].apply(
        map_product_ingredients,
        ingredient_mapping=ingredient_mapping,
    )
    products = add_ingredient_count(
        products,
        mapped_values,
        ingredient_mapping,
        kosmopedia_functions,
        manual_functions,
    )
    products = add_weighted_rating(products)
    return add_log_opinions(products)


def main() -> None:
    """Uruchamia analizę i zapisuje pliki wynikowe."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Słownik INCI budujemy raz i przekazujemy do wszystkich etapów, żeby każdy
    # raport korzystał z identycznej logiki dopasowania.
    ingredient_mapping = build_inci_mapping_lookup()
    kosmopedia_functions = build_kosmopedia_function_lookup()
    manual_functions = build_manual_function_lookup()
    processed = build_processed_products(
        ingredient_mapping,
        kosmopedia_functions,
        manual_functions,
    )
    ingredient_frequency, unmapped_frequency = build_ingredient_frequency(
        processed,
        ingredient_mapping,
        kosmopedia_functions,
        manual_functions,
    )
    category_summary = save_category_summary_outputs(processed)
    brand_summary = save_brand_summary_outputs(processed)
    ingredient_flag_summary = save_ingredient_flag_summary_outputs(
        processed,
        ingredient_mapping,
    )
    ingredient_family_group_comparison = (
        save_ingredient_family_group_comparison_outputs(processed)
    )
    ingredient_family_mannwhitney_tests = (
        save_ingredient_family_mannwhitney_test_outputs(processed)
    )
    ingredient_family_regression_results = (
        save_ingredient_family_regression_outputs(processed)
    )
    (
        weighted_rating_regression_coefficients,
        weighted_rating_regression_model_summary,
        weighted_rating_regression_ingredient_effects,
    ) = save_weighted_rating_regression_outputs(processed)
    weighted_rating_active_sensitivity = (
        save_weighted_rating_active_sensitivity_outputs(processed)
    )
    (
        product_characteristics_correlation,
        product_formula_correlation,
        active_intensity_correlation,
        ingredient_family_correlation,
    ) = save_correlation_matrix_outputs(processed)
    category_specific_outputs = save_category_specific_analysis_outputs(
        processed,
        (
            "Pielęgnacja twarzy",
            "Oczyszczanie i demakijaż twarzy",
        ),
    )
    subcategory_specific_outputs = save_subcategory_specific_analysis_outputs(
        processed,
        (
            "Kremy do twarzy",
            "Maseczki",
            "Serum, boostery i esencje",
            "Kremy pod oczy",
            "\u017bele i pianki do mycia twarzy",
            "P\u0142yny do demakija\u017cu",
        ),
    )
    (
        segmentation_dataset,
        segmentation_results,
        segment_summary,
        segmentation_metrics,
    ) = save_global_segmentation_outputs(processed)
    (
        face_care_segmentation_dataset,
        face_care_segmentation_results,
        face_care_segment_summary,
        face_care_segmentation_metrics,
    ) = save_face_care_segmentation_outputs(processed)
    (
        cleansing_segmentation_dataset,
        cleansing_segmentation_results,
        cleansing_segment_summary,
        cleansing_segmentation_metrics,
    ) = save_cleansing_segmentation_outputs(processed)

    processed.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format=CSV_FLOAT_FORMAT)
    ingredient_frequency.to_csv(
        ingredient_frequency_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    unmapped_frequency.to_csv(
        ingredient_frequency_unmapped_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )

    print(f"Saved {len(processed)} rows to {output_csv}")
    print(f"Saved {len(ingredient_frequency)} rows to {ingredient_frequency_csv}")
    print(
        f"Saved {len(unmapped_frequency)} rows to "
        f"{ingredient_frequency_unmapped_csv}"
    )
    print(f"Saved {len(category_summary)} rows to {category_summary_csv}")
    print(f"Saved {len(brand_summary)} rows to {brand_summary_csv}")
    print(
        f"Saved {len(ingredient_flag_summary)} rows to "
        f"{ingredient_flag_summary_csv}"
    )
    print(
        f"Saved {len(ingredient_family_group_comparison)} rows to "
        f"{ingredient_family_group_comparison_csv}"
    )
    print(
        f"Saved {len(ingredient_family_mannwhitney_tests)} rows to "
        f"{ingredient_family_mannwhitney_tests_csv}"
    )
    print(
        f"Saved {len(ingredient_family_regression_results)} rows to "
        f"{ingredient_family_regression_models_csv}"
    )
    print(
        f"Saved {len(weighted_rating_regression_coefficients)} rows to "
        f"{weighted_rating_regression_coefficients_csv}"
    )
    print(
        f"Saved {len(weighted_rating_regression_model_summary)} rows to "
        f"{weighted_rating_regression_model_summary_csv}"
    )
    print(
        f"Saved {len(weighted_rating_regression_ingredient_effects)} rows to "
        f"{weighted_rating_regression_ingredient_effects_csv}"
    )
    print(
        f"Saved {len(weighted_rating_active_sensitivity)} rows to "
        f"{weighted_rating_active_sensitivity_csv}"
    )
    print(
        f"Saved product characteristics correlation matrix to "
        f"{product_characteristics_correlation_csv}"
    )
    print(
        f"Saved product formula correlation matrix to "
        f"{product_formula_correlation_csv}"
    )
    print(
        f"Saved active intensity correlation matrix to "
        f"{active_intensity_correlation_csv}"
    )
    print(
        f"Saved ingredient family correlation matrix to "
        f"{ingredient_family_correlation_csv}"
    )
    for category, paths in category_specific_outputs.items():
        print(f"Saved category-specific analysis for {category}:")
        for path in paths.values():
            print(f"  {path}")
    for subcategory, paths in subcategory_specific_outputs.items():
        print(f"Saved subcategory-specific analysis for {subcategory}:")
        for path in paths.values():
            print(f"  {path}")
    print(f"Saved {len(segmentation_dataset)} rows to {segmentation_dataset_csv}")
    print(f"Saved {len(segmentation_results)} rows to {segmentation_results_csv}")
    print(f"Saved {len(segment_summary)} rows to {segment_summary_csv}")
    print(f"Saved {len(segmentation_metrics)} rows to {segmentation_model_metrics_csv}")
    print(
        f"Saved {len(face_care_segmentation_dataset)} rows to "
        f"{face_care_segmentation_dataset_csv}"
    )
    print(
        f"Saved {len(face_care_segmentation_results)} rows to "
        f"{face_care_segmentation_results_csv}"
    )
    print(
        f"Saved {len(face_care_segment_summary)} rows to "
        f"{face_care_segment_summary_csv}"
    )
    print(
        f"Saved {len(face_care_segmentation_metrics)} rows to "
        f"{face_care_segmentation_model_metrics_csv}"
    )
    print(
        f"Saved {len(cleansing_segmentation_dataset)} rows to "
        f"{cleansing_segmentation_dataset_csv}"
    )
    print(
        f"Saved {len(cleansing_segmentation_results)} rows to "
        f"{cleansing_segmentation_results_csv}"
    )
    print(
        f"Saved {len(cleansing_segment_summary)} rows to "
        f"{cleansing_segment_summary_csv}"
    )
    print(
        f"Saved {len(cleansing_segmentation_metrics)} rows to "
        f"{cleansing_segmentation_model_metrics_csv}"
    )


if __name__ == "__main__":
    main()
