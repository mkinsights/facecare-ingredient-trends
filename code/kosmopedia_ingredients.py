from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


BASE_URL = "https://www.kosmopedia.org/encyklopedia/"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "raw" / "kosmopedia_ingredients.csv"
DEFAULT_CACHE_DIR = ROOT / "raw" / "kosmopedia_html_cache"

GROUP_LABELS = {
    "base": "Składniki bazowe",
    "active": "Składniki aktywne",
    "preservative": "Konserwanty",
    "preservatives": "Konserwanty",
    "filters": "Filtry UV",
    "uv filters": "Filtry UV",
    "dyes": "Barwniki",
    "fragrance": "Substancje zapachowe",
    "fragrances": "Substancje zapachowe",
}

ALLERGY_LABELS = {
    "nodata": "brak danych",
    "brak danych": "brak danych",
    "none": "brak alergiczności",
    "potencial": "potencjalny",
    "potential": "potencjalny",
    "alergen": "alergen",
}


@dataclass
class Ingredient:
    inci_name: str
    nazwa_zwyczajowa: str
    identyfikacja_chemiczna: str
    cas: str
    aktywna_grupa_funkcyjna: str
    aktywna_grupa_funkcyjna_kod: str
    funkcja_w_produkcie: str
    dzialanie_kosmetyczne: str
    bezpieczenstwo: str
    alergicznosc: str
    alergicznosc_kod: str
    wplyw_na_srodowisko: str
    pochodzenie: str
    url: str


def clean_text(value: str) -> str:
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", value)
    value = re.sub(r"(?is)<[^>]+>", " ", value)
    value = unescape(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s*", "\n", value)
    return value.strip(" \n")


def first_match(pattern: str, html: str, flags: int = re.I | re.S) -> str:
    match = re.search(pattern, html, flags)
    return clean_text(match.group(1)) if match else ""


def slug_cache_path(cache_dir: Path, url: str) -> Path:
    slug = urlparse(url).path.strip("/").replace("/", "__") or "index"
    return cache_dir / f"{slug}.html"


def fetch(session: requests.Session, url: str, cache_dir: Path, delay: float) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = slug_cache_path(cache_dir, url)
    if path.exists():
        return path.read_text(encoding="utf-8")

    response = session.get(url, timeout=30)
    response.raise_for_status()
    html = response.text
    path.write_text(html, encoding="utf-8")
    if delay:
        time.sleep(delay)
    return html


def extract_index_links(index_html: str) -> list[tuple[str, str]]:
    content_match = re.search(
        r'<div class="page-content wrapper-narrow">(.*?)<footer', index_html, re.I | re.S
    )
    content = content_match.group(1) if content_match else index_html
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'<a\s+[^>]*href="(https://www\.kosmopedia\.org/encyklopedia/[^"#?]+/)"[^>]*>(.*?)</a>',
        re.I | re.S,
    )
    for url, label_html in pattern.findall(content):
        label = clean_text(label_html)
        if url.rstrip("/") == BASE_URL.rstrip("/") or not label:
            continue
        if url in seen:
            continue
        seen.add(url)
        links.append((label, url))
    return links


def extract_header_field(label: str, header_html: str) -> str:
    pattern = rf"<strong>\s*{re.escape(label)}\s*:\s*</strong>\s*(.*?)(?:<br\s*/?>|</p>|$)"
    return first_match(pattern, header_html)


def extract_cas(header_html: str) -> str:
    value = extract_header_field("Numer CAS", header_html)
    return value


def extract_selected_list_value(html: str, list_class: str, labels: dict[str, str]) -> tuple[str, str]:
    pattern = rf'<ul[^>]*class="[^"]*{re.escape(list_class)}[^"]*"[^>]*data-group="([^"]*)"[^>]*>(.*?)</ul>'
    match = re.search(pattern, html, re.I | re.S)
    if not match:
        return "", ""

    selected_code = unescape(match.group(1)).strip()
    selected_parts = [part.strip().lower() for part in selected_code.split(",") if part.strip()]
    selected_labels = [labels[part] for part in selected_parts if part in labels]
    if selected_labels:
        return ", ".join(dict.fromkeys(selected_labels)), selected_code

    for code, label_html in re.findall(
        r'<li[^>]*data-group="([^"]*)"[^>]*>(.*?)</li>', match.group(2), re.I | re.S
    ):
        if code == selected_code:
            return clean_text(label_html), selected_code
    return "", selected_code


def extract_section_by_id(html: str, section_id: str) -> str:
    pattern = rf'<h2[^>]*id="{re.escape(section_id)}"[^>]*>.*?</h2>(.*?)(?=<h2\b|<section\b|</article>)'
    match = re.search(pattern, html, re.I | re.S)
    if not match:
        return ""
    section_html = re.sub(
        r'<div[^>]*class="[^"]*link-list-wrapper[^"]*"[^>]*>.*?</div>',
        " ",
        match.group(1),
        flags=re.I | re.S,
    )
    return clean_text(section_html)


def parse_ingredient(index_name: str, url: str, html: str) -> Ingredient:
    title = first_match(r'<h1[^>]*class="[^"]*page-title[^"]*"[^>]*>(.*?)</h1>', html)
    raw_header = re.search(
        r'<p[^>]*class="[^"]*text-below-title[^"]*"[^>]*>(.*?)</p>', html, re.I | re.S
    )
    header_html = raw_header.group(1) if raw_header else ""
    functional_label, functional_code = extract_selected_list_value(
        html, "functional-group-list", GROUP_LABELS
    )
    allergy_label, allergy_code = extract_selected_list_value(
        html, "allergy-group-list", ALLERGY_LABELS
    )

    return Ingredient(
        inci_name=title or index_name,
        nazwa_zwyczajowa=extract_header_field("Nazwa zwyczajowa", header_html),
        identyfikacja_chemiczna=extract_header_field("Identyfikacja chemiczna", header_html),
        cas=extract_cas(header_html),
        aktywna_grupa_funkcyjna=functional_label,
        aktywna_grupa_funkcyjna_kod=functional_code,
        funkcja_w_produkcie=extract_section_by_id(html, "funkcja-w-produkcie"),
        dzialanie_kosmetyczne=extract_section_by_id(html, "dziaanie-kosmetyczne"),
        bezpieczenstwo=extract_section_by_id(html, "bezpieczestwo"),
        alergicznosc=allergy_label,
        alergicznosc_kod=allergy_code,
        wplyw_na_srodowisko=extract_section_by_id(html, "wpyw-na-rodowisko"),
        pochodzenie=extract_section_by_id(html, "pochodzenie"),
        url=url,
    )


def write_csv(rows: Iterable[Ingredient], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def scrape(args: argparse.Namespace) -> list[Ingredient]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; KosmopediaIngredientsScraper/1.0; "
                "+https://www.kosmopedia.org/encyklopedia/)"
            )
        }
    )

    index_html = fetch(session, BASE_URL, args.cache_dir, args.delay)
    links = extract_index_links(index_html)
    if args.limit:
        links = links[: args.limit]
    if not links:
        raise RuntimeError("Nie znaleziono linków do składników w indeksie Kosmopedii.")

    rows: list[Ingredient] = []
    total = len(links)
    for idx, (name, url) in enumerate(links, start=1):
        try:
            html = fetch(session, url, args.cache_dir, args.delay)
            rows.append(parse_ingredient(name, url, html))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {idx}/{total} Nie udało się pobrać {url}: {exc}", file=sys.stderr)
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"[INFO] Przetworzono {idx}/{total}", file=sys.stderr)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape ingredient cards from Kosmopedia encyclopedia."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="Use a small number for tests.")
    args = parser.parse_args()

    rows = scrape(args)
    if not rows:
        raise RuntimeError("Nie zapisano żadnego składnika.")
    write_csv(rows, args.output)
    print(f"Zapisano {len(rows)} składników do {args.output}")


if __name__ == "__main__":
    main()
