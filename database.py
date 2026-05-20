import os
import oracledb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

db_queries = {
    'query_base_population' : """
            SELECT Brand, model, Year, kmDriven, Transmission, FuelType, AskPrice 
            FROM used_cars 
            WHERE ROWNUM <= 500"""
}

def clean_car_data(df):
    df.columns = df.columns.str.upper()
    
    df['KMDRIVEN'] = df['KMDRIVEN'].astype(str).str.replace(' km', '', regex=False)
    df['KMDRIVEN'] = df['KMDRIVEN'].str.replace(',', '', regex=False)
    df['KMDRIVEN'] = df['KMDRIVEN'].str.split('.').str[0]
    df['KMDRIVEN'] = pd.to_numeric(df['KMDRIVEN'], errors='coerce').fillna(0).astype(int)

    df['ASKPRICE'] = df['ASKPRICE'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['ASKPRICE'] = pd.to_numeric(df['ASKPRICE'], errors='coerce').fillna(0)

    return df

def get_base_population():
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    dsn = os.getenv("DB_DSN")

    if not all([username, password, dsn]):
        print("Missing data in .env.")
        return None

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connected to db!")

        query = db_queries['query_base_population']
        
        df = pd.read_sql(query, con=connection)
        return df

    except Exception as e:
        print(f"Error while connecting or getting data. {e}")
        return None
        
    finally:
        if 'connection' in locals():
            connection.close()