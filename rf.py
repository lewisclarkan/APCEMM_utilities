import pandas as pd
import numpy as np
import argparse
import pickle
import xarray as xr

from src.rf_core  import read_apcemm_data, apce_data_struct, calc_sample
from src.file_management import write_output_header_contrail, write_output_contrail

def average_particle_radius_over_time(ds_t):
    radii = []
    for ds in ds_t:
        r = ds["r"].values
        n = ds["Overall size distribution"].values
        total_particles = n.sum()
        if total_particles == 0:
            radii.append(0.0)
        else:
            radii.append((r * n).sum() / total_particles)
    return np.mean(radii)

def average_number_concentration_over_time(ds_t):
    concentrations = []
    for ds in ds_t:
        n = ds["Overall size distribution"].values
        # If n is 1D, sum; if 2D (e.g. y,x), take mean over grid
        if n.ndim == 1:
            concentrations.append(n.sum())
        else:
            concentrations.append(n.sum(axis=0).mean())
    return np.mean(concentrations)

def average_optical_depth_over_time(ds_t):
    optical_depths = []
    for ds in ds_t:
        od = ds["Vertical optical depth"].values
        # If od is 2D (e.g. y,x), take mean over grid
        optical_depths.append(np.mean(od))
    return np.mean(optical_depths)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Runs the main computation')
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    arg = parser.parse_args()

    start_index = arg.start
    end_index = arg.end
    month = arg.month

    # This flag is used to check whether the file we are trying to read is already being accessed
    flag_2 = 0

    # Access the albedo grib file that we have already downloaded
    ds = xr.load_dataset('gribs/albedo.grib', engine="cfgrib")
    met_albedo = ds.expand_dims({'level':[-1]})

    # Read in the APCEMM summary results. Add column titles
    df = pd.read_csv(f"outputs/{month}_{start_index}_{end_index}.txt", sep = ",", header = 0)
    df.columns = ["Index", "Status", "Latitude", "Longitude", "Altitude", "Time"]

    write_output_header_contrail(f"outputs/{month}_{start_index}_{end_index}_contrail.txt")

    # Iterating from 1 to the number of plumes simulated
    for i in range(0, df.shape[0]):

        temp = start_index + i
        
        apce_data_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{temp}_apce_data.pkl")
        sample_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{temp}_sample.pkl")
        ds_temp_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{temp}_ds_temp.pkl")
        path_start = (f"results/month_{month}/{start_index}_{end_index}/sample_{temp}")
        
        try: 
            with open(f"results/month_{month}/{start_index}_{end_index}/APCEMM_results/APCEMM_out_{temp}/status_case0", "r") as f:
                status = f.readline()
        except:
            status = "Complete\n"

        print(status)
            
        if str(status) == "Complete\n":
            print(sample_file_name)
            apce_data = read_apcemm_data(f"./results/month_{month}/{start_index}_{end_index}/APCEMM_results/APCEMM_out_{temp}/")
            try:
                with open(sample_file_name, "rb") as f:
                    
                    sample = pickle.load(f)
            except FileNotFoundError:
                flag_2 = 1
                
            if flag_2==0:
                

                with open(ds_temp_file_name, "rb") as f:
                    ds_temp = pickle.load(f)

                j_per_m, age = calc_sample(apce_data, sample, met_albedo, ds_temp, path_start)
            
                ds_t = apce_data.ds_t

                avg_r_time = average_particle_radius_over_time(ds_t) * 1e6  # Convert to micrometers
                avg_n_time = average_number_concentration_over_time(ds_t) * 1e6
                avg_od_time = average_optical_depth_over_time(ds_t)

                write_output_contrail(f"outputs/{month}_{start_index}_{end_index}_contrail.txt", sample, df.iloc[i, 1], age, j_per_m, avg_r_time, avg_n_time, avg_od_time)
            
            else:
            
                print("Error")
                
            
        if status == "NoWaterSaturation":
        
            write_output_contrail(f"outputs/{month}_{start_index}_{end_index}_contrail.txt", sample, status, 0,0,  0, 0, 0)
