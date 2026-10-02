import argparse
import csv
import shutil
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import (
    InvalidSessionIdException,
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
INPUT_PRODUCTS_CSV = ROOT / "raw" / "products_analysis.csv"
OUTPUT_PRODUCTS_CSV = ROOT / ".cache" / "products_review_collection_state.csv"
OUTPUT_REVIEWS_CSV = ROOT / "raw" / "product_reviews.csv"
MAX_TEXT_REVIEWS = 10
MAX_COLLECTION_ROUNDS = 30
WAIT_SECONDS = 0.6


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,3200")
    return webdriver.Chrome(options=options)


def load_products(input_path: str | Path) -> list[dict]:
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_existing_reviews(output_path: str | Path) -> list[dict]:
    path = Path(output_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def backup_file(path_str: str | Path) -> None:
    path = Path(path_str)
    if not path.exists():
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)


def safe_text(parent, css_selector: str) -> str:
    try:
        return parent.find_element(By.CSS_SELECTOR, css_selector).text.strip()
    except NoSuchElementException:
        return ""


def collect_review_cards(driver: webdriver.Chrome) -> list:
    comment_elements = driver.find_elements(By.CSS_SELECTOR, 'p[class*="comment__"]')
    cards = []
    for comment_element in comment_elements:
        try:
            card = comment_element.find_element(
                By.XPATH,
                './ancestor::div[contains(@class, "opinion__")][1]',
            )
            cards.append(card)
        except NoSuchElementException:
            continue
    return cards


def open_opinions_section(driver: webdriver.Chrome) -> None:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "opinions")))
    toggle = driver.find_element(By.CSS_SELECTOR, '[data-testid="toggle-opinions-button"]')
    driver.execute_script("arguments[0].click();", toggle)
    time.sleep(WAIT_SECONDS)


def click_load_more_reviews(driver: webdriver.Chrome, current_count: int) -> bool:
    opinions_section = driver.find_element(By.ID, "opinions")
    driver.execute_script("arguments[0].scrollIntoView({block: 'end'});", opinions_section)
    time.sleep(WAIT_SECONDS)

    xpath = (
        ".//*[self::button or self::a]"
        "[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄ„Ä†ÄĹĹĂ“ĹšĹąĹ»', "
        "'abcdefghijklmnopqrstuvwxyzÄ…Ä‡Ä™Ĺ‚Ĺ„ĂłĹ›ĹşĹĽ'), 'kolejne opinie')]"
        " | .//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄ„Ä†ÄĹĹĂ“ĹšĹąĹ»', "
        "'abcdefghijklmnopqrstuvwxyzÄ…Ä‡Ä™Ĺ‚Ĺ„ĂłĹ›ĹşĹĽ'), 'kolejne opinie')]/ancestor::*[self::button or self::a][1]"
    )
    candidates = opinions_section.find_elements(By.XPATH, xpath)

    for candidate in candidates:
        try:
            driver.execute_script("arguments[0].click();", candidate)
            for _ in range(5):
                time.sleep(WAIT_SECONDS)
                new_count = len(collect_review_cards(driver))
                if new_count > current_count:
                    return True
        except Exception:
            continue

    return False


def collect_text_reviews(driver: webdriver.Chrome, max_reviews: int) -> list[dict]:
    reviews = []
    seen_keys = set()
    stalled_rounds = 0
    last_card_count = -1

    for _ in range(MAX_COLLECTION_ROUNDS):
        cards = collect_review_cards(driver)
        card_count = len(cards)

        for card in cards:
            review_text = safe_text(card, 'p[class*="comment__"]')
            if not review_text:
                continue

            review = {
                "review_rating": safe_text(card, 'div[class*="review__"]'),
                "review_text": review_text,
            }
            review_key = (review["review_rating"], review["review_text"])
            if review_key in seen_keys:
                continue

            seen_keys.add(review_key)
            reviews.append(review)
            if len(reviews) >= max_reviews:
                return reviews

        if card_count > last_card_count:
            last_card_count = card_count
            stalled_rounds = 0
        else:
            stalled_rounds += 1

        driver.execute_script("window.scrollBy(0, 1200);")
        time.sleep(WAIT_SECONDS)

        if click_load_more_reviews(driver, card_count):
            stalled_rounds = 0
            continue

        if stalled_rounds >= 2:
            break

    return reviews


def scrape_reviews_for_product(driver: webdriver.Chrome, product: dict, max_reviews: int) -> list[dict]:
    driver.get(product["url"])
    open_opinions_section(driver)
    reviews = collect_text_reviews(driver, max_reviews=max_reviews)

    for index, review in enumerate(reviews, start=1):
        review["product_id"] = product["id"]
        review["product_name"] = product["name"]
        review["product_url"] = product["url"]
        review["review_index"] = index

    return reviews


def save_reviews(reviews: list[dict], output_path: str | Path) -> bool:
    fieldnames = [
        "product_id",
        "product_name",
        "product_url",
        "review_index",
        "review_rating",
        "review_text",
    ]
    existing_count = len(load_existing_reviews(output_path))
    if existing_count > len(reviews):
        return False

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: review.get(field, "") for field in fieldnames})
    return True


def save_products_with_review_counts(
    products: list[dict],
    output_path: str | Path,
) -> None:
    if not products:
        return

    fieldnames = list(products[0].keys())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)


def deduplicate_reviews(reviews: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    seen_keys: set[tuple[str, str, str]] = set()

    for review in reviews:
        product_id = review.get("product_id", "")
        rating = review.get("review_rating", "")
        text = review.get("review_text", "")
        review_key = (product_id, rating, text)
        if not product_id or not text or review_key in seen_keys:
            continue

        seen_keys.add(review_key)
        grouped.setdefault(product_id, []).append(
            {
                "product_id": product_id,
                "product_name": review.get("product_name", ""),
                "product_url": review.get("product_url", ""),
                "review_rating": rating,
                "review_text": text,
            }
        )

    deduplicated = []
    for product_id in sorted(grouped.keys()):
        for index, review in enumerate(grouped[product_id], start=1):
            review["review_index"] = index
            deduplicated.append(review)

    return deduplicated


def merge_product_reviews(
    all_reviews: list[dict], product: dict, new_reviews: list[dict], max_reviews: int
) -> list[dict]:
    kept_reviews = []
    product_reviews = []

    for review in all_reviews:
        if review.get("product_id") == product["id"]:
            product_reviews.append(review)
        else:
            kept_reviews.append(review)

    merged_reviews = deduplicate_reviews(product_reviews + new_reviews)[:max_reviews]
    for index, review in enumerate(merged_reviews, start=1):
        review["product_name"] = product["name"]
        review["product_url"] = product["url"]
        review["review_index"] = index

    return deduplicate_reviews(kept_reviews + merged_reviews)


def enrich_products_with_review_counts(products: list[dict], all_reviews: list[dict], target: int) -> list[dict]:
    counts = {}
    for review in all_reviews:
        counts[review["product_id"]] = counts.get(review["product_id"], 0) + 1

    for product in products:
        count = counts.get(product["id"], 0)
        product["selenium_text_reviews_count"] = count
        product["selenium_text_reviews_target"] = target
        product["selenium_reviews_complete"] = count >= target
        product["selenium_reviews_processed"] = count >= target

    return products


def sort_products_for_retry(products: list[dict], target: int) -> list[dict]:
    def review_count(product: dict) -> int:
        return int(product.get("selenium_text_reviews_count", 0) or 0)

    incomplete = [product for product in products if review_count(product) < target]
    complete = [product for product in products if review_count(product) >= target]

    def incomplete_priority(product: dict) -> tuple[int, int, str]:
        count = review_count(product)
        priority_group = 0 if count <= 5 else 1
        return (priority_group, -count, product.get("id", ""))

    incomplete.sort(key=incomplete_priority)
    complete.sort(key=lambda product: (review_count(product), product.get("id", "")))
    return incomplete + complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PRODUCTS_CSV)
    parser.add_argument("--products-output", type=Path, default=OUTPUT_PRODUCTS_CSV)
    parser.add_argument("--reviews-output", type=Path, default=OUTPUT_REVIEWS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-reviews", type=int, default=MAX_TEXT_REVIEWS)
    args = parser.parse_args()

    products = load_products(args.input)
    if args.limit is not None:
        products = products[: args.limit]

    all_reviews = deduplicate_reviews(load_existing_reviews(args.reviews_output))
    backup_file(args.reviews_output)
    backup_file(args.products_output)
    products = enrich_products_with_review_counts(products, all_reviews, args.max_reviews)
    products = sort_products_for_retry(products, args.max_reviews)
    save_products_with_review_counts(products, args.products_output)

    driver = build_driver()

    try:
        total = len(products)
        for index, product in enumerate(products, start=1):
            existing_count = int(product.get("selenium_text_reviews_count", 0) or 0)
            if existing_count >= args.max_reviews:
                product["selenium_reviews_processed"] = True
                print(f"[{index}/{total}] {product['id']} -> pomijam, ma juz {existing_count} opinii")
                continue

            try:
                reviews = scrape_reviews_for_product(driver, product, args.max_reviews)
                all_reviews = merge_product_reviews(all_reviews, product, reviews, args.max_reviews)
                products = enrich_products_with_review_counts(products, all_reviews, args.max_reviews)
                products = sort_products_for_retry(products, args.max_reviews)
                product["selenium_reviews_processed"] = product["selenium_reviews_complete"]
                if not save_reviews(all_reviews, args.reviews_output):
                    all_reviews = deduplicate_reviews(load_existing_reviews(args.reviews_output))
                    products = enrich_products_with_review_counts(products, all_reviews, args.max_reviews)
                    products = sort_products_for_retry(products, args.max_reviews)
                    print(
                        f"[{index}/{total}] {product['id']} -> pomijam zapis, bo na dysku jest lepszy stan opinii"
                    )
                save_products_with_review_counts(products, args.products_output)
                print(
                    f"[{index}/{total}] {product['id']} -> teraz ma {product['selenium_text_reviews_count']} opinii tekstowych"
                )
            except (InvalidSessionIdException, TimeoutException, WebDriverException):
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = build_driver()
                print(f"[{index}/{total}] {product['id']} -> restart przegladarki po bledzie sesji")
            except TimeoutException:
                print(f"[{index}/{total}] {product['id']} -> timeout przy pobieraniu opinii")
            except Exception as error:
                print(f"[{index}/{total}] {product['id']} -> blad: {type(error).__name__}")
    finally:
        driver.quit()

    print(
        f"Zapisano {len(all_reviews)} opinii do {args.reviews_output} "
        f"oraz {len(products)} produktow do {args.products_output}."
    )


if __name__ == "__main__":
    main()
