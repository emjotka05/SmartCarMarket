"""Moduł CLI"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UserPreferences:
    """Preferencje wyszukiwania samochodu podane przez użytkownika."""

    max_budget_pln: float
    max_mileage_km: int
    min_year: int
    preferred_transmission: Optional[str] = None
    preferred_brand: Optional[str] = None
    preferred_model: Optional[str] = None
    strict_brand_focus: bool = False


def _read_positive_float(prompt: str) -> float:
    """
    Wczytaj dodatnią liczbę float
    """
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            print("  Error: enter a number.")
            continue
        if value <= 0:
            print("  Error: value must be greater than zero.")
            continue
        return value


def _read_positive_int(prompt: str, min_value: int = 1) -> int:
    """
    Wczytaj dodatnią liczbę całkowitą 
    """
    while True:
        raw = input(prompt).strip().replace(" ", "")
        if not raw.isdigit():
            print("  Error: enter only digits (no letters or special characters).")
            continue
        value = int(raw)
        if value < min_value:
            print(f"  Error: value must be at least {min_value}.")
            continue
        return value


def _read_optional_transmission() -> Optional[str]:
    """
    Wczytaj opcjonalną preferencję skrzyni biegów.
    """
    print(
        "\nPreferred transmission - optional."
        "\n  Manual, Automatic."
        "\n  Press Enter to skip."
    )
    raw = input("Transmission: ").strip()
    if not raw:
        return None
    return raw.lower()


def _read_optional_text(prompt: str) -> Optional[str]:
    """
    Wczytaj opcjonalny tekst.
    """
    raw = input(prompt).strip()
    if not raw:
        return None
    return raw.lower()


def _read_yes_no(prompt: str) -> bool:
    """
    Wczytaj odpowiedź tak/nie.

    """
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Error: enter 'yes' or 'no'.")


def collect_user_preferences() -> UserPreferences:
    """
    Interaktywnie zbierz wymagania użytkownika przez input().
    """
    print("  SmartCarMarket – Your search preferences")
    print("_" * 30)

    max_budget = _read_positive_float("\nMaximum budget pln: ")
    max_mileage = _read_positive_int("\nMaximum acceptable mileage km: ")
    min_year = _read_positive_int(
        "\nMinimum year: ",
        min_value=1990,
    )
    transmission = _read_optional_transmission()
    preferred_brand = _read_optional_text(
        "\nPreferred brand - optional, Enter to skip: "
    )
    preferred_model = _read_optional_text(
        "Preferred model - optional, Enter to skip: "
    )
    strict_brand_focus = False
    if preferred_brand:
        strict_brand_focus = _read_yes_no(
            "Brand mandatory? yes/no: "
        )

    prefs = UserPreferences(
        max_budget_pln=max_budget,
        max_mileage_km=max_mileage,
        min_year=min_year,
        preferred_transmission=transmission,
        preferred_brand=preferred_brand,
        preferred_model=preferred_model,
        strict_brand_focus=strict_brand_focus,
    )

    print("\nPreferences summary")
    print(f"  Budget:       up to {prefs.max_budget_pln:,.0f} pln")
    print(f"  Mileage:      up to {prefs.max_mileage_km:,} km")
    print(f"  Year:         from {prefs.min_year}")
    if prefs.preferred_transmission:
        print(f"  Transmission:  {prefs.preferred_transmission}")
    else:
        print("  Transmission:  no preference")
    if prefs.preferred_brand:
        print(f"  Brand:         {prefs.preferred_brand}")
    else:
        print("  Brand:         no preference")
    if prefs.preferred_brand:
        print(
            "  Brand focus:   "
            + ("strict - required" if prefs.strict_brand_focus else "soft - preferred")
        )
    if prefs.preferred_model:
        print(f"  Model:         {prefs.preferred_model}")
    else:
        print("  Model:         no preference")
    print("=" * 60 + "\n")

    return prefs
