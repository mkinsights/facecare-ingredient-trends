import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests


CATEGORY_URL = "https://www.rossmann.pl/kategoria/pielegnacja-i-higiena/twarz,13047"
PRODUCTS_API_URL = "https://www.rossmann.pl/products/v4/api/Products"
PAGE_SIZE = 100
DETAIL_WORKERS = 10
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "raw" / "products_analysis.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )
}


def build_product_name(product: dict) -> str:
    parts = [
        product.get("brand", "").strip(),
        product.get("caption", "").strip(),
        product.get("unit", "").strip(),
    ]
    return " ".join(part for part in parts if part)


def extract_category_id(category_url: str) -> int:
    match = re.search(r",(\d+)(?:\?.*)?$", category_url)
    if not match:
        raise ValueError("Nie udalo sie odczytac categoryId z URL-a kategorii.")
    return int(match.group(1))


def strip_html(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(no_tags).split())


def extract_ingredients_from_html(html: str) -> str | None:
    section_patterns = [
        r"<h2[^>]*>\s*Sk(?:ł|l)adniki\s*</h2>.*?<(?:div|section)[^>]*productDescriptionContent[^>]*>(.*?)</(?:div|section)>",
        r"<h2[^>]*>\s*Ingredients\s*</h2>.*?<(?:div|section)[^>]*productDescriptionContent[^>]*>(.*?)</(?:div|section)>",
        r"<p[^>]*>\s*Ingredients:\s*(.*?)</p>",
        r"<p[^>]*>\s*Sk(?:ł|l)adniki:\s*(.*?)</p>",
    ]

    for pattern in section_patterns:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            ingredients = strip_html(match.group(1))
            ingredients = re.sub(
                r"^(Ingredients|Skladniki|Składniki)\s*:\s*",
                "",
                ingredients,
                flags=re.IGNORECASE,
            )
            ingredients = re.sub(r"^:\s*", "", ingredients)
            return ingredients or None

    return None


def fetch_products_page(category_id: int, page: int) -> dict:
    response = requests.get(
        PRODUCTS_API_URL,
        params={
            "page": page,
            "pageSize": PAGE_SIZE,
            "categoryId": category_id,
            "order": "default",
        },
        headers=HEADERS | {"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"]


def fetch_product_page_details(product_url: str) -> dict:
    response = requests.get(product_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    html = response.text
    return {
        "ingredients": extract_ingredients_from_html(html),
    }


def fetch_products(category_url: str) -> list[dict]:
    category_id = extract_category_id(category_url)
    first_page_data = fetch_products_page(category_id, page=1)
    total_pages = first_page_data["totalPages"]

    products_by_id = {}

    for page in range(1, total_pages + 1):
        page_data = first_page_data if page == 1 else fetch_products_page(category_id, page)

        for item in page_data["items"]:
            products_by_id[item["id"]] = {
                "id": item["id"],
                "brand": item.get("brand"),
                "name": build_product_name(item),
                "url": urljoin("https://www.rossmann.pl", item["navigateUrl"]),
                "price": item.get("price"),
                "capacity": item.get("unit"),
                "stars": item.get("averageRating"),
                "opinions": item.get("totalReviews"),
                "availability": item.get("availability"),
                "price_per_unit": item.get("pricePerUnit"),
                "category": item.get("category"),
            }

    return list(products_by_id.values())


def filter_products(products: list[dict]) -> list[dict]:
    return [
        product
        for product in products
        if product.get("availability") == "available" and (product.get("opinions") or 0) >= 1
    ]


def enrich_products(products: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as executor:
        future_to_product = {
            executor.submit(fetch_product_page_details, product["url"]): product
            for product in products
        }

        completed = 0
        total = len(future_to_product)

        for future in as_completed(future_to_product):
            product = future_to_product[future]
            product.update(future.result())
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Pobrano szczegoly dla {completed}/{total} produktow...")

    return products


def filter_products_with_ingredients(products: list[dict]) -> list[dict]:
    return [product for product in products if product.get("ingredients")]


def scrape_products(category_url: str) -> list[dict]:
    products = filter_products(fetch_products(category_url))
    print(f"Po filtracji zostalo {len(products)} produktow.")
    enriched_products = enrich_products(products)
    products_with_ingredients = filter_products_with_ingredients(enriched_products)
    print(f"Po odfiltrowaniu produktow bez skladu zostalo {len(products_with_ingredients)} produktow.")
    return products_with_ingredients


def save_products_to_csv(
    products: list[dict],
    output_path: Path = OUTPUT_CSV,
) -> None:
    fieldnames = [
        "id",
        "brand",
        "name",
        "url",
        "category",
        "price",
        "price_per_unit",
        "capacity",
        "stars",
        "opinions",
        "ingredients",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for product in products:
            writer.writerow({field: product.get(field) for field in fieldnames})


if __name__ == "__main__":
    products = scrape_products(CATEGORY_URL)
    save_products_to_csv(products)
    print(f"Zapisano {len(products)} produktow do pliku {OUTPUT_CSV}.")
