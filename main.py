from database import get_base_population, clean_car_data

def main():

    car_parameters = get_base_population()

    if car_parameters is not None:
        car_parameters = clean_car_data(car_parameters)
           
        print(f"\nCollected {len(car_parameters)} cars.")
        print("Base Population Data:")
        print(car_parameters.head(10))

if __name__ == "__main__":
    main()