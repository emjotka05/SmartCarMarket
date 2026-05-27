"""Moduł AI – dopasowanie samochodów algorytmem genetycznym - pygad"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pygad

from cli import UserPreferences

# Wagi składowych funkcji celu (suma komponentów ≈ skala 0–100 przy idealnym aucie).
WEIGHT_BUDGET = 35.0
WEIGHT_MILEAGE = 30.0
WEIGHT_YEAR = 25.0
WEIGHT_TRANSMISSION = 10.0
WEIGHT_BRAND = 8.0
WEIGHT_MODEL = 12.0

# Współczynniki kar za naruszenie twardych ograniczeń.
PENALTY_BUDGET_PER_PLN_RATIO = 80.0
PENALTY_MILEAGE_PER_KM_RATIO = 60.0
PENALTY_YEAR_PER_YEAR = 15.0
PENALTY_TRANSMISSION_MISMATCH = 8.0
PENALTY_BRAND_MISMATCH = 6.0
PENALTY_MODEL_MISMATCH = 10.0
PENALTY_STRICT_BRAND_MISMATCH = 1e6

CURRENT_YEAR = 2026


@dataclass(frozen=True)
class CarMatchResult:
    """Wynik dopasowania pojedynczego samochodu."""

    row_index: int
    fitness: float
    brand: str
    model: str
    price_pln: float
    mileage_km: int
    year: int
    transmission: str


def _normalize_transmission(value: object) -> str:
    """Znormalizuj nazwę skrzyni do porównań (małe litery, bez spacji)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _normalize_text(value: object) -> str:
    """Znormalizuj pole tekstowe do porównań (małe litery)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _is_match(candidate: str, preference: str) -> bool:
    """Sprawdź dopasowanie tekstowe (substring w obie strony)."""
    if not candidate or not preference:
        return False
    return preference in candidate or candidate in preference


def compute_car_fitness(
    price_pln: float,
    mileage_km: float,
    year: float,
    transmission: object,
    brand: object,
    model: object,
    prefs: UserPreferences,
    year_ceiling: int = CURRENT_YEAR,
) -> float:
    """
    Oblicz wartość dopasowania (fitness) pojedynczego samochodu do preferencji.

    Funkcja jest **maksymizowana** przez pygad (wyższy wynik = lepsze dopasowanie).

    Matematyka (dla samochodu o cenie ``P``, przebiegu ``M``, roczniku ``Y``):

    **1. Składowa budżetowa** (waga ``w_B = WEIGHT_BUDGET``):

    - Jeśli ``P <= B_max`` (samochód mieści się w budżecie):

      ``S_B = w_B * (B_max - P) / B_max``

      Im tańszy względem limitu, tym wyższa nagroda (liniowo od 0 do ``w_B``).

    - Jeśli ``P > B_max``:

      ``S_B = - PENALTY_BUDGET * (P - B_max) / B_max``

      Kara rośnie liniowo z nadwyżką ceny ponad budżet.

    **2. Składowa przebiegu** (waga ``w_M = WEIGHT_MILEAGE``):

    - Jeśli ``M <= M_max``:

      ``S_M = w_M * (M_max - M) / M_max``

      Niższy przebieg → wyższa nagroda.

    - Jeśli ``M > M_max``:

      ``S_M = - PENALTY_MILEAGE * (M - M_max) / M_max``

    **3. Składowa rocznika** (waga ``w_Y = WEIGHT_YEAR``):

    - Jeśli ``Y >= Y_min``:

      ``S_Y = w_Y * (Y - Y_min) / (Y_ceil - Y_min)``

      gdzie ``Y_ceil`` to górny rok skali (domyślnie bieżący rok). Nowszy rocznik
      w dopuszczalnym przedziale daje wyższą nagrodę.

    - Jeśli ``Y < Y_min``:

      ``S_Y = - PENALTY_YEAR * (Y_min - Y)``

      Kara za każdy rok poniżej minimum.

    **4. Składowa skrzyni** (waga ``w_T = WEIGHT_TRANSMISSION``):

    - Gdy użytkownik **nie** podał preferencji: ``S_T = w_T / 2`` (neutralna stała).

    - Gdy podał i skrzynia pasuje (porównanie bez rozróżniania wielkości liter):
      ``S_T = w_T``.

    - Gdy podał i nie pasuje: ``S_T = - PENALTY_TRANSMISSION_MISMATCH``.

    **5. Składowa marki** (waga ``w_BR = WEIGHT_BRAND``):

    - Brak preferencji marki: ``S_BR = w_BR / 2``.
    - Dopasowanie marki (substring, case-insensitive): ``S_BR = w_BR``.
    - Brak dopasowania: ``S_BR = - PENALTY_BRAND_MISMATCH``.

    **6. Składowa modelu** (waga ``w_MO = WEIGHT_MODEL``):

    - Brak preferencji modelu: ``S_MO = w_MO / 2``.
    - Dopasowanie modelu (substring, case-insensitive): ``S_MO = w_MO``.
    - Brak dopasowania: ``S_MO = - PENALTY_MODEL_MISMATCH``.

    **Fitness końcowy:**

    ``F = S_B + S_M + S_Y + S_T + S_BR + S_MO``

    Args:
        price_pln: Cena oferty w PLN (kolumna ASKPRICE po czyszczeniu).
        mileage_km: Przebieg w km (KMDRIVEN).
        year: Rocznik (YEAR).
        transmission: Typ skrzyni (TRANSMISSION).
        brand: Marka (BRAND).
        model: Model (MODEL).
        prefs: Preferencje użytkownika.
        year_ceiling: Górna granica skali rocznika w nagrodzie.

    Returns:
        Liczba rzeczywista – fitness do maksymalizacji.
    """
    budget_max = prefs.max_budget_pln
    mileage_max = prefs.max_mileage_km
    year_min = prefs.min_year

    # --- 1. Budżet ---
    if price_pln <= budget_max:
        budget_score = WEIGHT_BUDGET * (budget_max - price_pln) / budget_max
    else:
        over_ratio = (price_pln - budget_max) / budget_max
        budget_score = -PENALTY_BUDGET_PER_PLN_RATIO * over_ratio

    # --- 2. Przebieg ---
    if mileage_km <= mileage_max:
        mileage_score = WEIGHT_MILEAGE * (mileage_max - mileage_km) / mileage_max
    else:
        over_ratio = (mileage_km - mileage_max) / mileage_max
        mileage_score = -PENALTY_MILEAGE_PER_KM_RATIO * over_ratio

    # --- 3. Rocznik ---
    year_span = max(year_ceiling - year_min, 1)
    if year >= year_min:
        year_score = WEIGHT_YEAR * (year - year_min) / year_span
    else:
        year_score = -PENALTY_YEAR_PER_YEAR * (year_min - year)

    # --- 4. Skrzynia biegów ---
    if prefs.preferred_transmission is None:
        transmission_score = WEIGHT_TRANSMISSION / 2.0
    else:
        car_tx = _normalize_transmission(transmission)
        pref_tx = _normalize_transmission(prefs.preferred_transmission)
        if pref_tx in car_tx or car_tx in pref_tx:
            transmission_score = WEIGHT_TRANSMISSION
        else:
            transmission_score = -PENALTY_TRANSMISSION_MISMATCH

    # --- 5. Marka ---
    car_brand = _normalize_text(brand)
    pref_brand = _normalize_text(prefs.preferred_brand)
    if not pref_brand:
        brand_score = WEIGHT_BRAND / 2.0
    elif _is_match(car_brand, pref_brand):
        brand_score = WEIGHT_BRAND
    else:
        if prefs.strict_brand_focus:
            # Twarde wymuszenie marki: rozwiązania spoza marki stają się praktycznie
            # niemożliwe do wygrania w selekcji i rankingu.
            brand_score = -PENALTY_STRICT_BRAND_MISMATCH
        else:
            brand_score = -PENALTY_BRAND_MISMATCH

    # --- 6. Model ---
    car_model = _normalize_text(model)
    pref_model = _normalize_text(prefs.preferred_model)
    if not pref_model:
        model_score = WEIGHT_MODEL / 2.0
    elif _is_match(car_model, pref_model):
        model_score = WEIGHT_MODEL
    else:
        model_score = -PENALTY_MODEL_MISMATCH

    return float(
        budget_score
        + mileage_score
        + year_score
        + transmission_score
        + brand_score
        + model_score
    )


def _build_fitness_callable(
    df: pd.DataFrame,
    prefs: UserPreferences,
) -> Callable:
    """
    Zbuduj funkcję fitness zgodną z API pygad (maksymalizacja).

    Chromosom: jeden gen = indeks wiersza w ``df`` (całkowity).
    """
    prices = df["ASKPRICE"].to_numpy(dtype=float)
    mileages = df["KMDRIVEN"].to_numpy(dtype=float)
    years = df["YEAR"].to_numpy(dtype=float)
    transmissions = df["TRANSMISSION"].tolist()
    brands = df["BRAND"].tolist()
    models = df["MODEL"].tolist()
    num_rows = len(df)

    def fitness_func(ga_instance, solution, solution_idx) -> float:
        row_index = int(solution[0])
        if row_index < 0 or row_index >= num_rows:
            return -1e6
        return compute_car_fitness(
            price_pln=prices[row_index],
            mileage_km=mileages[row_index],
            year=years[row_index],
            transmission=transmissions[row_index],
            brand=brands[row_index],
            model=models[row_index],
            prefs=prefs,
        )

    return fitness_func


def run_genetic_search(
    df: pd.DataFrame,
    prefs: UserPreferences,
    num_generations: int = 40,
    sol_per_pop: int = 50,
    num_parents_mating: int = 10,
    mutation_percent_genes: float = 10.0,
    random_seed: Optional[int] = 42,
) -> pygad.GA:
    """
    Uruchom algorytm genetyczny wybierający indeksy wierszy z populacji aut.

    Args:
        df: Oczyszczona ramka danych z ``database.clean_car_data``.
        prefs: Preferencje użytkownika.
        num_generations: Liczba pokoleń ewolucji.
        sol_per_pop: Wielkość populacji.
        num_parents_mating: Liczba rodziców krzyżowanych w każdym pokoleniu.
        mutation_percent_genes: Procent genów poddawanych mutacji.
        random_seed: Ziarno RNG dla powtarzalności testów (``None`` = losowo).

    Returns:
        Wytrenowana instancja ``pygad.GA`` po ``run()``.
    """
    if df.empty:
        raise ValueError("DataFrame samochodów jest pusty – brak populacji do ewolucji.")

    num_rows = len(df)
    gene_space = list(range(num_rows))
    normalized_brands = df["BRAND"].map(_normalize_text).tolist()
    pref_brand = _normalize_text(prefs.preferred_brand)

    if prefs.strict_brand_focus and pref_brand:
        strict_gene_space = [
            idx for idx, brand in enumerate(normalized_brands) if _is_match(brand, pref_brand)
        ]
        if strict_gene_space:
            gene_space = strict_gene_space

    fitness_func = _build_fitness_callable(df, prefs)

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=min(num_parents_mating, sol_per_pop // 2),
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=1,
        gene_type=int,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_num_genes=1,
        mutation_percent_genes=mutation_percent_genes,
        random_seed=random_seed,
        suppress_warnings=True,
        save_best_solutions=True,
    )

    ga.run()
    return ga


def get_top_matches(
    df: pd.DataFrame,
    prefs: UserPreferences,
    ga_instance: pygad.GA,
    top_n: int = 3,
) -> List[CarMatchResult]:
    """
    Wybierz ``top_n`` najlepiej dopasowanych, unikalnych samochodów.

    Końcowy ranking liczony jest przez pełne przeliczenie fitness dla całej
    populacji. Dzięki temu prezentowane TOP N jest globalnie najlepsze względem
    funkcji celu, niezależnie od jakości zbieżności GA.

    Args:
        df: Ramka danych samochodów.
        prefs: Preferencje (do przeliczenia fitness przy sortowaniu).
        ga_instance: Zakończona instancja pygad po ``run_genetic_search``.
        top_n: Liczba wyników do zwrócenia.

    Returns:
        Lista wyników posortowana malejąco według fitness.
    """
    # Gwarancja poprawności rankingu:
    # finalne TOP N liczymy po pełnym przeliczeniu fitness dla wszystkich aut.
    # GA nadal pełni rolę przeszukiwania ewolucyjnego, ale prezentowany wynik
    # jest globalnie najlepszy względem funkcji celu.
    pref_brand = _normalize_text(prefs.preferred_brand)
    all_scored: List[Tuple[int, float]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        if prefs.strict_brand_focus and pref_brand:
            row_brand = _normalize_text(row["BRAND"])
            if not _is_match(row_brand, pref_brand):
                continue
        fit = compute_car_fitness(
            price_pln=float(row["ASKPRICE"]),
            mileage_km=float(row["KMDRIVEN"]),
            year=float(row["YEAR"]),
            transmission=row["TRANSMISSION"],
            brand=row["BRAND"],
            model=row["MODEL"],
            prefs=prefs,
        )
        all_scored.append((idx, fit))

    sorted_indices = sorted(all_scored, key=lambda item: item[1], reverse=True)[:top_n]

    results: List[CarMatchResult] = []
    for row_index, fitness in sorted_indices:
        row = df.iloc[row_index]
        results.append(
            CarMatchResult(
                row_index=row_index,
                fitness=fitness,
                brand=str(row["BRAND"]),
                model=str(row["MODEL"]),
                price_pln=float(row["ASKPRICE"]),
                mileage_km=int(row["KMDRIVEN"]),
                year=int(row["YEAR"]),
                transmission=str(row["TRANSMISSION"]),
            )
        )
    return results


def find_best_cars(
    df: pd.DataFrame,
    prefs: UserPreferences,
    top_n: int = 3,
    **ga_kwargs,
) -> List[CarMatchResult]:
    """
    Uruchom ewolucję i zwróć ``top_n`` najlepszych dopasowań.

    Args:
        df: Oczyszczona populacja aut.
        prefs: Preferencje użytkownika.
        top_n: Liczba rekomendacji.
        **ga_kwargs: Opcjonalne parametry przekazywane do ``run_genetic_search``.

    Returns:
        Lista ``CarMatchResult``.
    """
    ga = run_genetic_search(df, prefs, **ga_kwargs)
    return get_top_matches(df, prefs, ga, top_n=top_n)
