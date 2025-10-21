import pandas as pd
import numpy as np
import yaml
from subprocess import call
import os
import argparse
import pickle

from pycontrails import Flight

from src.aircraft import set_flight_parameters, get_aircraft_properties
from src.generate_yaml import generate_yaml_d
from src.geodata import open_dataset, advect, get_albedo, get_temperature_and_clouds_met
from src.radiative_forcing import read_apcemm_data, apce_data_struct
from src.file_management import write_output_header, write_output

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Runs the main computation')
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    arg = parser.parse_args()

    start_index = arg.start
    end_index = arg.end
    month = arg.month

    print(f"Starting final.py between samples {start_index} and {end_index} for month {month}...\n")

    # Read in the flight and aircraft data
    df_samples  = pd.read_pickle(f"results/month_{month}/samples.pkl")
    df_aircraft = pd.read_csv('flight_data/aircraft.csv', index_col = 0)

    # Open the albedo dataset
    met_albedo = get_albedo('gribs/albedo.grib')

    # Prepare the output files
    try: os.makedirs(f"results/month_{month}/{start_index}_{end_index}")
    except: pass
    try: os.makedirs(f"outputs")
    except: pass
    
    file_name =  (f"outputs/{month}_{start_index}_{end_index}.txt")
    write_output_header(file_name)

    # Start the main loop
    for i in range(start_index, end_index):

        success1 = False
        success2 = False

        identifier = i
        sample = df_samples.iloc[i,:]

        # Download and open the low and high resolution datasets
        print("Downloading and opening dataset...\n")

        while not success1:
            try:
                met = open_dataset(sample)
                success1 = True
            except (OSError, BlockingIOError) as e:
                #print(f"Error opening dataset for sample {i}: {e}")
                success1 = False

        while not success2:
            try:
                met_temp = get_temperature_and_clouds_met(sample)
                success2 = True
            except (OSError, BlockingIOError) as e:
                #print(f"Error opening temperature dataset for sample {i}: {e}")
                success2 = False

        # Create the flight object and set its parameters
        altitude = 10900
        fl = set_flight_parameters(sample, df_aircraft, altitude, i)

        try: os.makedirs(f"results/month_{month}/{start_index}_{end_index}/mets")
        except: pass
        try: os.makedirs(f"results/month_{month}/{start_index}_{end_index}/yamls")
        except: pass

        # Run the DryAdvection model
        print("Running DryAdvection model...\n")
        ds, ds_temp, pressure, temperature, flag = advect(met, met_temp, fl)

        # Check if the DryAdvection model ran successfully
        if flag == False:

            try: os.remove(f"results/month_{month}/{start_index}_{end_index}/mets/input{i}.nc")
            except FileNotFoundError: pass
            try: os.remove(f"results/month_{month}/{start_index}_{end_index}/mets/input_temp{i}.nc")
            except FileNotFoundError: pass

            ds.to_netcdf(f"results/month_{month}/{start_index}_{end_index}/mets/input{i}.nc")
            ds_temp.to_netcdf(f"results/month_{month}/{start_index}_{end_index}/mets/input_temp{i}.nc")

            properties_dict = get_aircraft_properties(sample, df_aircraft, temperature)

            path = (f"../../../../results/month_{month}/{start_index}_{end_index}/APCEMM_results/APCEMM_out_{i}/")
            met_file_path = (f"../../../../results/month_{month}/{start_index}_{end_index}/mets/input{i}.nc")

            # Generate the .yaml input dictionary and output to file
            d = generate_yaml_d(identifier, sample, fl, float(pressure/100), properties_dict, met_file_path, path)
            with open(f"results/month_{month}/{start_index}_{end_index}/yamls/input{i}.yaml", "w") as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False, sort_keys=False)

            try: os.makedirs(f"results/month_{month}/{start_index}_{end_index}/APCEMM_results")
            except: pass

            # Run the APCEMM model
            apcemm_file_path = "../../build/APCEMM"
            call(["./../../build/APCEMM", f"results/month_{month}/{start_index}_{end_index}/yamls/input{i}.yaml"])
            print("APCEMM done")

            # Read the output data from APCEMM
            apce_data = read_apcemm_data(f"./results/month_{month}/{start_index}_{end_index}/APCEMM_results/APCEMM_out_{i}/")

            try:
                # Check the status of the contrail formation
                with open(f"results/month_{month}/{start_index}_{end_index}/APCEMM_results/APCEMM_out_{i}/status_case0", "r") as f:
                    status = f.readline()

                # If the contrail did not persist, set the results and write them to the output file
                if (str(status) == "NoPersistence\n"):
                    status = "No persistence"
                    print("No_persistence\n")

                # If the conrail persisted, write the output files.
                else:
                    try: os.makedirs(f"results/month_{month}/{start_index}_{end_index}/")
                    except: pass

                    apce_data_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}_apce_data.pkl")
                    sample_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}_sample.pkl")
                    ds_temp_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}_ds_temp.pkl")

                    print("Pickling APCE_DATA")
                    print(apce_data)

                    with open(apce_data_file_name, "wb") as f:
                        pickle.dump(apce_data, f)
                    with open(sample_file_name, "wb") as f:
                        pickle.dump(sample, f)
                    with open(ds_temp_file_name, "wb") as f:
                        pickle.dump(ds_temp, f)

                    status = "Contrail_formed"

            # Perform error handling
            except FileNotFoundError:
                status = "Error"
                continue

        # Perform error handling
        else:
            status = "Error"

        write_output(file_name, sample, status)
    