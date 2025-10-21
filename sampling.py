import pandas as pd
import argparse
import os

from src.aircraft import clean_flight_data
from src.sampling import generateDfSamples

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates pickle sample dataframe from flight data')
    parser.add_argument("--n_samples", required=True, type=int)
    parser.add_argument("--n_flights", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    arg = parser.parse_args()

    n_samples = arg.n_samples
    n_flights = arg.n_flights
    month = arg.month

    df = pd.read_csv('flight_data/flightlist_20190101_20190131.csv.gz', 
                    dtype={
                        'callsign':str,
                        'number':str,
                        'icao24':str,
                        'registration':str,
                        'typecode':str,
                        'origin':str,
                        'destination':str,
                        'firstseen':object,
                        'lastseen':object,
                        'day':object,
                        'latitude_1':float,
                        'longitude_1':float,
                        'altitude_1':float,
                        'latitude_2':float,
                        'longitude_2':float,
                        'altitude_2':float
                        })

    df.drop('number', axis=1, inplace=True)
    df.drop('registration', axis=1, inplace=True)
    df.drop('icao24', axis=1, inplace=True)

    df_aircraft = pd.read_csv('flight_data/aircraft.csv', index_col = 0)

    df = clean_flight_data(df, df_aircraft)
    df = df.sample(frac=1)
    
    if n_flights == 0:
        n_flights = df.shape[0]

    df_samples = generateDfSamples(df, n_samples, n_flights)
    df_samples = df_samples.sort_values("time")

    try: os.makedirs("results")
    except: pass
    try: os.makedirs(f"results/month_{month}")
    except: pass
    
    df_samples.to_pickle(f"results/month_{month}/samples.pkl")
