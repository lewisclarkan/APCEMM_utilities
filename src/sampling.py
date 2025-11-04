import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import random
import time

from alive_progress import alive_bar
from pycontrails import Flight

def calcTotalDistance(flights) -> float:
    """Calculates total distance flown"""
    
    total_distance = 0 

    for flight in flights:
        total_distance += flight.length

    return total_distance

def preprocessFlights(flights, n_flights):
    with alive_bar(n_flights) as bar:
        for i in range(len(flights)):
            flights[i] = flights[i].resample_and_fill('1min')
            bar()
    return flights

def samplePoint(flights, total_distance):
    """Generate a random index i for flight index, and j 
    for segment index. Inputs are a list of pycontrails flight objects"""

    total_distance_int = int(round(total_distance))
    sample_distance = random.randint(0, total_distance_int - 1)

    cumulative_distance = 0
    for i, flight in enumerate(flights):
        if cumulative_distance + flight.length > sample_distance:
            # Found the flight
            remaining_dist = sample_distance - cumulative_distance
            lengths = flight.segment_length()
            segment_cum = 0
            for j, seg_len in enumerate(lengths):
                if segment_cum + seg_len > remaining_dist:
                    return [i, j]
                segment_cum += seg_len
            # Fallback in case of rounding issues
            return [i, len(lengths) - 1]
        cumulative_distance += flight.length

    # Fallback: return last flight and last segment
    return [len(flights) - 1, len(flights[-1].segment_length()) - 1]

def generateFlight(flight):

    flight_attrs = {
        "flight_id":        flight["callsign"],
        "aircraft_type":    flight["typecode"]
    }

    df=pd.DataFrame()

    # TODO: see if there is a better way to do this, especially altitude
    df["latitude"] = np.array([flight["latitude_1"], flight["latitude_2"]])
    df["longitude"] = np.array([flight["longitude_1"], flight["longitude_2"]])
    df["time"] = np.array([flight["firstseen"], flight["lastseen"]])
    df["altitude_ft"] = np.array([35_000.0, 35_000.0])

    return Flight(df, attrs=flight_attrs)

def generateDfSamples(df, n_samples, n_flights):

    samples = np.empty((n_samples,2),int)
    flights = []

    print("Converting to list of flight objects...")
    with alive_bar(n_flights) as bar:
        for i in range(0, n_flights):
            flights.append(generateFlight(df.iloc[i]))
            bar()

    total_distance = calcTotalDistance(flights)

    print(f"\nTotal distance flown in dataset was {total_distance/1000:.2f} km.")

    print("\nPreprocessing flights...")
    flights = preprocessFlights(flights, n_flights)

    print("\nTaking samples...")
    with alive_bar(n_samples) as bar:
        for i in range(0, n_samples):
            samples[i] = samplePoint(flights, total_distance)
            bar()

    longitudes   = np.empty(n_samples)
    latitudes    = np.empty(n_samples)
    altitudes    = np.empty(n_samples)
    times        = np.empty(n_samples, dtype = 'datetime64[s]')
    aircrafts    = np.empty(n_samples, dtype = object)

    print("\nDetermining sample characteristics...\n")
    with alive_bar(n_samples) as bar:
        for i in range(0, n_samples):
            longitudes[i]   = flights[samples[i][0]]['longitude'][samples[i][1]]
            latitudes[i]    = flights[samples[i][0]]['latitude'][samples[i][1]]
            altitudes[i]    = flights[samples[i][0]]['altitude'][samples[i][1]]
            times[i]        = flights[samples[i][0]]['time'][samples[i][1]]
            aircrafts[i]    = flights[samples[i][0]].attrs['aircraft_type']
            bar()

    df_samples = pd.DataFrame({
        "index": np.arange(0, n_samples, 1).astype(np.int32),
        "longitude": longitudes.astype(np.float32),
        "latitude": latitudes.astype(np.float32),
        "altitude": altitudes.astype(np.float32),
        "time": times,
        "aircraft type": pd.Categorical(aircrafts),
    })

    return df_samples

    #return
