from __future__ import annotations

import csv
import json
import re
import shutil
from html import unescape
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
TARGET_FILES = (RAW_DIR / "products_analysis.csv",)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )
}


def is_missing_ingredients(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    return normalized in {"", ".", "brak danych"}


def strip_html(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(no_tags).split())


def is_probable_ingredients_text(value: str) -> bool:
    uppercase_value = value.upper()
    blocked_phrases = (
        "DODATKOWE INFORMACJE",
        "OSTRZEŻENIA",
        "PODMIOT ODPOWIEDZIALNY",
        "KOD EAN",
        "OPINIE",
        "WSZYSTKIE OPINIE",
    )
    if any(phrase in uppercase_value for phrase in blocked_phrases):
        return False

    parts = [part.strip() for part in value.split(",") if part.strip()]
    return len(parts) >= 6


def clean_ingredients_candidate(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = strip_html(value)
    cleaned = re.sub(
        r"^(Ingredients|Skladniki|Składniki)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^:\s*", "", cleaned)
    cleaned = cleaned.strip(" .")

    if is_missing_ingredients(cleaned):
        return None

    if not is_probable_ingredients_text(cleaned):
        return None

    return cleaned


def extract_next_data_html_blocks(html: str) -> list[str]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return []

    data = json.loads(match.group(1))
    details = (
        data.get("props", {})
        .get("pageProps", {})
        .get("initialState", {})
        .get("product", {})
        .get("productDetails", {})
        .get("data", [])
    )
    return [
        item.get("html", "")
        for item in details
        if isinstance(item, dict) and isinstance(item.get("html"), str)
    ]


def extract_ingredients_from_next_data(html: str) -> str | None:
    for block in extract_next_data_html_blocks(html):
        if not re.search(r"(Ingredients|Skladniki|Składniki)\s*:", block, flags=re.IGNORECASE):
            continue

        candidate = clean_ingredients_candidate(block)
        if candidate:
            return candidate

    return None


def extract_ingredients_from_html(html: str) -> str | None:
    patterns = [
        r"<p[^>]*>\s*(?:Ingredients|Skladniki|Składniki)\s*:\s*(.*?)</p>",
        r"<h2[^>]*>\s*(?:Ingredients|Skladniki|Składniki)\s*</h2>.*?<p[^>]*>(.*?)</p>",
    ]

    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        candidate = clean_ingredients_candidate(match.group(1))
        if candidate:
            return candidate

    match = re.search(
        r"([A-Z][A-Z0-9\-/() ]{20,}(?:,\s*[A-Z0-9\-/() ]{2,}){5,})",
        html,
    )
    if not match:
        return None

    candidate = clean_ingredients_candidate(match.group(1))
    if candidate and any(token in candidate for token in ("AQUA", "GLYCERIN", "PARFUM", "CI ")):
        return candidate

    return None


def fetch_ingredients(url: str) -> str | None:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    html = response.text
    return extract_ingredients_from_next_data(html) or extract_ingredients_from_html(html)


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), list(reader.fieldnames or [])


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)

    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refill_file(path: Path, ingredients_by_id: dict[str, str]) -> int:
    rows, fieldnames = load_csv(path)
    updated = 0

    for row in rows:
        ingredient_value = ingredients_by_id.get(str(row.get("id", "")))
        if not ingredient_value or not is_missing_ingredients(row.get("ingredients")):
            continue

        row["ingredients"] = ingredient_value
        updated += 1

    if updated:
        save_csv(path, rows, fieldnames)

    return updated


def main() -> None:
    source_rows, _ = load_csv(TARGET_FILES[0])
    missing_rows = [row for row in source_rows if is_missing_ingredients(row.get("ingredients"))]

    recovered: dict[str, str] = {}
    for index, row in enumerate(missing_rows, start=1):
        product_id = str(row["id"])
        ingredients = fetch_ingredients(row["url"])
        if ingredients:
            recovered[product_id] = ingredients
            print(f"[{index}/{len(missing_rows)}] {product_id} -> znaleziono sklad")
        else:
            print(f"[{index}/{len(missing_rows)}] {product_id} -> nadal brak skladu")

    for path in TARGET_FILES:
        updated = refill_file(path, recovered)
        print(f"{path.name}: zaktualizowano {updated} rekordow.")

    print(f"Odzyskano sklad dla {len(recovered)} produktow.")


if __name__ == "__main__":
    main()
