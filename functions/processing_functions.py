import numpy as np
import pandas as pd

# Implémentation de la fonction qui récuperera les différentes variables temporelles à partir de DELIVERY_START
def extract_time_features(df, datetime_col = 'DELIVERY_START'):
    df = df.copy()
    
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day_of_month'] = df[datetime_col].dt.day
    df['day_of_week'] = df[datetime_col].dt.weekday # lundi = 0 ; dimanche = 6
    df['hour'] = df[datetime_col].dt.hour

    df['is_weekend'] = df['day_of_week'].isin([5,6])
    df['is_peak_hour'] = df['hour'].isin([6,7,8,9,10,17,18,19,20,21,22]) # on vise large sur les heures de pointes

    # Convertir les booléens en int (0/1), juste pour être explicite
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_peak_hour'] = df['is_peak_hour'].astype(int)

    def map_season(month):
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Autumn
    
    df['season'] = df['month'].apply(map_season)

    return df

def add_energy_features(df):
    df = df.copy()

    # Capacité totale pilotable hors renouvelables
    df['dispatchable_capacity'] = df['coal_power_available'] + df['gas_power_available'] + df['nucelear_power_available']
    
    # Capacité totale estimée (renouvelable + pilotable)
    df['total_capacity_estimate'] = df['solar_power_forecasts_average'] + df['wind_power_forecasts_average'] + df['dispatchable_capacity']
    
    # Part des renouvelables dans la capacité totale
    df['renewable_share'] = (df['solar_power_forecasts_average'] + df['wind_power_forecasts_average']) / df['total_capacity_estimate']
    
    # Ratio charge prévue / capacité pilotable (pression sur l'offre pilotable)
    df['load_vs_dispatchable'] = df['load_forecast'] / df['dispatchable_capacity']

    return df
