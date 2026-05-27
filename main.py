""" SmartCarMarket"""

from database import clean_car_data, get_base_population
from cli import collect_user_preferences
from ai import find_best_cars


def _print_recommendations(matches) -> None:
    """najlepsze dopasowania."""
    print("\n" + "=" * 60)
    print("  TOP 3:  ")
 

    if not matches:
        print("  No results to display.")
        return

    for rank, car in enumerate(matches, start=1):
        print(f"\n#{rank} with fitness: {car.fitness:.2f}")
        print(f"  Brand:       {car.brand}")
        print(f"  Model:       {car.model}")
        print(f"  Price:       {car.price_pln:,.0f} PLN")
        print(f"  Mileage:     {car.mileage_km:,} km")
        print(f"  Year:        {car.year}")
        print(f"  Transmission: {car.transmission}")

def main() -> None:
    """kolejnosc: baza czyszczenie preferencje ewolucja  wyniki."""
    print("SmartCarMarket --- optimized searching with genetic algorithm\n")

    raw_data = get_base_population()
    if raw_data is None:
        print("Failed to retrieve data from the database. Check .env file and connection.")
        return

    cars_df = clean_car_data(raw_data)
    print(f"Retrieved and cleaned {len(cars_df)} car offers.")

    if cars_df.empty:
        print("Car population is empty!")
        return

    preferences = collect_user_preferences()

    top_matches = find_best_cars(cars_df, preferences, top_n=3)

    _print_recommendations(top_matches)


if __name__ == "__main__":
    main()
