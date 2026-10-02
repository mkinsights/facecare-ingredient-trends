# Skład kosmetyków i opinie

Repozytorium zawiera kod i dane użyte w pracy magisterskiej. Analiza łączy składy INCI kosmetyków do pielęgnacji twarzy z ocenami i tekstami opinii.

Dane są migawką wykorzystaną w pracy. Ponowne pobranie stron może dać inne wyniki, ponieważ oferta, opinie i budowa stron zmieniają się w czasie.

## Środowisko

Analizę wykonano w Pythonie 3.11.9. Dokładne wersje bibliotek są zapisane w `requirements.txt`.

```powershell
py -3.11 -m venv "$env:USERPROFILE\venvs\magisterka"
& "$env:USERPROFILE\venvs\magisterka\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

Środowisko znajduje się poza repozytorium, dlatego nie dodaje lokalnych plików do projektu.

Chrome lub Chromium jest potrzebny tylko do ponownego pobierania opinii.

## Foldery

- `code` – skrypty analizy i pobierania danych;
- `raw` – dane źródłowe użyte w pracy;
- `processed` – dane po czyszczeniu i wyniki;
- `processed/summary` – wyniki dotyczące produktów i składników;
- `processed/sentiment/model` – wyniki modelu wydźwięku;
- `processed/sentiment/validation` – ręczne etykiety i walidacja;
- `processed/sentiment/benchmark` – porównanie z modelem nadzorowanym;
- `processed/sentiment/product` – wyniki na poziomie produktu;
- `processed/sentiment/aspects` – wyniki dla relacji składnik–aspekt;
- `processed/sentiment/multilevel` – modele wielopoziomowe.

Folder `validation` ma trzy części: `initial`, `tuned` i `final`. Odpowiadają one pierwszej walidacji, walidacji po poprawkach i końcowej walidacji warstwowej.

Skrypty zapisują wyniki tabelaryczne i nie tworzą wykresów. W repozytorium nie ma też zapisanych modeli ani plików tymczasowych.

## Uruchomienie

Całą analizę uruchamia się z głównego folderu projektu:

```powershell
python run_analysis.py
```

Skrypt kolejno:

1. czyści dane produktów i analizuje składy INCI;
2. przygotowuje opinie;
3. buduje model wydźwięku i sprawdza go na ręcznych etykietach;
4. łączy wyniki z produktami;
5. liczy modele dla relacji składnik–aspekt i modele wielopoziomowe;
6. sprawdza, czy najważniejsze liczebności i wyniki są zgodne z pracą.

Samo sprawdzenie zapisanych wyników:

```powershell
python run_analysis.py --check
```

Skrypty `create_sentiment_*_sample.py` pokazują sposób losowania próbek do ręcznego oznaczenia. Nie są częścią automatycznego przebiegu, ponieważ utworzyłyby nowe puste próbki.

## Dane

Najważniejsze pliki wejściowe:

- `raw/products_analysis.csv` – 1 999 produktów i ich składy INCI;
- `raw/product_reviews.csv` – 12 839 opinii przed czyszczeniem;
- `raw/kosmopedia_ingredients.csv` – 1 200 kart składników;
- `processed/inci_mapping.csv` – słownik nazw INCI;
- `processed/ingredient_functional_groups.csv` – uzupełnienia grup funkcjonalnych.

Ponowne pobieranie danych nie jest potrzebne do odtworzenia wyników. Wymaga internetu i uwzględnienia aktualnych zasad serwisów źródłowych:

```powershell
python code/rossmann_products.py
python code/refill_missing_ingredients.py
python code/rossmann_reviews.py
python code/kosmopedia_ingredients.py
```

Wyniki opisują zależności obserwowane w danych i nie powinny być interpretowane jako związki przyczynowe. Model wydźwięku uczy się głównie na ocenach całych opinii, dlatego jego wyniki są przybliżeniem.
