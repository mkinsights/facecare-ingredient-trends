# Instrukcja ręcznej anotacji opinii kosmetycznych

Plik do uzupełnienia: `manual_stratified_validation_sample_240.csv`.

Każdy wiersz zawiera jeden fragment opinii. Oceniaj wyłącznie treść tego
fragmentu, bez domyślania się informacji z nazwy produktu, oceny gwiazdkowej
ani pozostałej części recenzji.

## Kolumny do uzupełnienia

### `manual_aspect`

Wpisz jeden **główny** aspekt opisany w fragmencie. Używaj dokładnie jednego z
poniższych kodów albo `no_aspect`.

| Kod | Co oznacza | Przykłady |
| --- | --- | --- |
| `effectiveness` | Ogólna skuteczność i efekty działania na skórę, poza konkretnymi aspektami niżej. | „wyrównuje koloryt”, „działa na przebarwienia”, „nic nie robi”, „świetnie sprawdza się w rutynie” |
| `hydration_nourishment` | Nawilżenie, odżywienie oraz odczucie suchości, ściągnięcia lub przesuszenia. | „dobrze nawilża”, „wysusza”, „skóra jest ściągnięta”, „odżywia” |
| `skin_tolerance` | Tolerancja skóry i ocena podrażnień. | „nie szczypie”, „koi”, „piecze”, „uczula”, „zaczerwienienie”, „ulga” |
| `pore_clogging` | Zapychanie porów, komedogenność, zaskórniki wynikające z zapychania. | „nie zatyka porów”, „zapchał mnie”, „mam zaskórniki” |
| `cleansing_makeup_removal` | Mycie, oczyszczanie albo demakijaż. | „dobrze domywa”, „nie zmywa makijażu”, „pieni się”, „oczyszcza” |
| `texture_consistency` | Właściwości samej konsystencji lub tekstury przed/w trakcie użycia. | „lekka formuła”, „rzadki”, „kremowy”, „żelowy”, „lepki”, „tłusty” |
| `absorption_finish` | Wchłanianie i efekt pozostawiony na skórze po aplikacji. | „szybko się wchłania”, „roluje się”, „zostawia film”, „błyszczy”, „matowi”, „bieli” |
| `scent` | Zapach, aromat, perfumowość albo brak zapachu. | „ładnie pachnie”, „za intensywny zapach”, „bezzapachowy” |
| `application` | Łatwość i wygoda nakładania, rozprowadzania lub dozowania. | „łatwo się rozprowadza”, „wygodna aplikacja”, „trudno wydobyć”, „bezproblemowa aplikacja” |
| `packaging` | Fizyczne opakowanie i jego działanie. | „pompka się psuje”, „przecieka”, „wygodna tubka”, „atomizer” |
| `price_value` | Cena, opłacalność i relacja jakości do ceny. | „za drogi”, „wart swojej ceny”, „kupiłam w promocji” |
| `efficiency` | Wydajność i tempo zużycia. | „wystarcza na długo”, „mała ilość wystarcza”, „szybko się kończy” |
| `formula_composition` | Skład lub obecność/nieobecność składników jako taka. | „dobry skład”, „ma retinol”, „bez alkoholu”, „dużo zapachu w składzie” |
| `no_aspect` | Brak oceny konkretnej właściwości produktu. | „polecam”, „super”, „kupiłam pierwszy raz”, „ładny produkt” |

## Reguły rozstrzygania przypadków granicznych

- „Lekka formuła” to `texture_consistency`, chyba że wypowiedź ocenia wyłącznie
  skład, np. „lekka formuła bez silikonów” - wtedy `formula_composition`.
- „Szybko się wchłania, nie zostawia filmu” to `absorption_finish`.
- „Nie zatyka porów, koi cerę” ma jako główny aspekt `pore_clogging`, a
  `skin_tolerance` wpisz dodatkowo w `manual_notes`.
- „Nie mam prawie zaskórników, koloryt jest równomierny” - główny aspekt wybierz
  zgodnie z głównym sensem zdania; drugi wpisz w `manual_notes`. Zaskórniki to
  `pore_clogging`, koloryt to `effectiveness`.
- „Doskonały dla cery tłustej z rozszerzonymi porami” bez informacji o działaniu
  to `no_aspect`. Samo wskazanie typu cery nie jest oceną efektu.
- „Bardzo fajna mgiełka”, „straszny płyn”, „just ok” bez doprecyzowania to
  `no_aspect`.
- „Dobrze przemyślany skład” to `formula_composition`, nawet gdy opinia ogólna
  jest pozytywna.
- Nie przypisuj automatycznie `pore_clogging` wypowiedzi o wysypce. Wysypka bez
  informacji o porach lub zapychaniu należy zwykle do `skin_tolerance`.

## `manual_sentiment`

Oceń sentyment wobec wybranego **głównego aspektu**, nie ogólny ton całej
recenzji.

- `positive`: autor chwali tę właściwość lub opisuje korzyść.
- `negative`: autor zgłasza problem, brak efektu lub niepożądaną właściwość.
- `neutral`: ocena jest mieszana, czysto opisowa albo nie daje jasnej oceny.

Przykłady:

- „Nie szczypie w oczy” -> `skin_tolerance`, `positive`.
- „Strasznie wysusza i ściąga” -> `hydration_nourishment`, `negative`.
- „Konsystencja jest żelowa” -> `texture_consistency`, `neutral`.
- „Nawilża, ale zostawia bardzo lepki film” -> wybierz aspekt dominujący;
  dla `hydration_nourishment` sentyment jest `positive`, a
  `absorption_finish` wpisz w `manual_notes`.

## `has_negation`

Wpisz `yes`, gdy negacja zmienia sens oceny konkretnej właściwości.
W przeciwnym razie wpisz `no`.

- `yes`: „nie zatyka”, „nie podrażnia”, „nie szczypie”, „nie roluje się”,
  „nie świecę się”, „nie nawilża”.
- `no`: „nie spodziewałam się takich rezultatów”, „nie mogłam uwierzyć” - tu
  „nie” jest elementem ogólnej konstrukcji językowej, a nie prostą negacją
  właściwości produktu.

## `manual_notes`

Pole opcjonalne. Wpisuj w nim:

- dodatkowe aspekty obecne w tym samym fragmencie, używając ich kodów
  rozdzielonych średnikiem, np. `skin_tolerance; absorption_finish`;
- krótką uwagę tylko wtedy, gdy przypadek jest niejednoznaczny.

Nie wpisuj tu przewidywanej klasy modelu ani oceny produktu z pamięci.

## Kolejność decyzji

1. Czy fragment opisuje konkretną właściwość? Jeśli nie, wybierz `no_aspect`.
2. Jaki jest główny temat wypowiedzi? Wpisz go w `manual_aspect`.
3. Czy jest drugi, niezależny aspekt? Dopisz jego kod w `manual_notes`.
4. Jaki jest sentyment wobec głównego aspektu?
5. Czy negacja zmienia jego sens?

