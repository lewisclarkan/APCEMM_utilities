import numpy as np
import pandas as pd
from pycontrails import Flight

def set_flight_parameters(sample, df_aircraft, altitude, i):

    flight_id = i
    aircraft_type = sample["aircraft type"]

    flight_attrs = {
        "flight_id" : flight_id,
        "aircraft_type" : aircraft_type,
    }

    df = pd.DataFrame()
    df["longitude"]          = np.linspace(sample["longitude"], sample["longitude"], 1)
    df["latitude"]           = np.linspace(sample["latitude"], sample["latitude"], 1)
    df["altitude"]           = np.linspace(altitude, altitude, 1)
    df["time"]               = pd.date_range(sample["time"], sample["time"], periods=1)

    return Flight(df, attrs=flight_attrs)


def get_aircraft_properties(sample, df_aircraft, temperature):

    #LCV = 43.1e6        # for jet fuel
    LCV = 42.5e6
    

    aircraft_type = sample["aircraft type"]


    efficiency = df_aircraft.loc[aircraft_type, "eta"]
    design_mach = df_aircraft.loc[aircraft_type, "Design_Mach"]
    thrust = 0.2*df_aircraft.loc[aircraft_type, "Thrust"]
    nvPM_EI_m = df_aircraft.loc[aircraft_type, "Soot-g/kg"]/3.3
    mass = df_aircraft.loc[aircraft_type, "mass"]
    wingspan = df_aircraft.loc[aircraft_type, "wingspan"]

    air_speed = design_mach * np.sqrt(1.4*287*temperature)
    fuel_flow_rate = thrust * air_speed / (efficiency * LCV)

    properties_dict = {
        "efficiency": efficiency,
        "air_speed": air_speed,
        "nvPM_EI_m": nvPM_EI_m,
        "fuel_flow_rate": fuel_flow_rate,
        "mass": mass,
        "wingspan": wingspan
    }

    return properties_dict


def clean_flight_data(df_samples, df_aircraft):

    aircraft_types = df_aircraft.index.tolist()
    df_samples = df_samples[df_samples["typecode"].isin(aircraft_types)]

    return df_samples

if __name__ == '__main__':

    df_samples = pd.read_pickle("samples/samples.pkl")
    df_aircraft = pd.read_csv('flight_data/aircraft.csv', index_col = 0)

    df_samples = (clean_flight_data(df_samples,df_aircraft))
