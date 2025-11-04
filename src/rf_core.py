import subprocess

import numpy as np
import pandas as pd
from pysolar.solar import *
from pathlib import Path

from src.file_management import *

def run_libradtran(xs, b_nighttime, path_start):

    uvspec_path = "../libRadtran-2.0.6/bin/uvspec"

    input_dir   = Path(path_start) / "inps"
    input_dir.mkdir(parents=True, exist_ok=True)

    result_dir  = Path(path_start) / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(xs)):

        input_thermal_clr   = Path(input_dir) / f"run_clearThermal{i}.DAT"
        input_thermal_con   = Path(input_dir) / f"runThermal{i}.DAT"
        input_solar_con     = Path(input_dir) / f"runSolar{i}.DAT"
        input_solar_clr     = Path(input_dir) / f"run_clearSolar{i}.DAT"
        output_thermal_clr  = Path(result_dir) / f"result_clrThermal{i}.DAT"
        output_thermal_con  = Path(result_dir) / f"result_conThermal{i}.DAT"
        output_solar_con    = Path(result_dir) / f"result_conSolar{i}.DAT"
        output_solar_clr    = Path(result_dir) / f"result_clrSolar{i}.DAT"

        # Run libRadtran over the thermal spectrum for contrail and no contrail. 
        subprocess.run([uvspec_path], stdin=open(input_thermal_clr, "rb"),
                       stdout=open(output_thermal_clr, "wb"), check=True)
        subprocess.run([uvspec_path], stdin=open(input_thermal_con, "rb"),
                       stdout=open(output_thermal_con, "wb"), check=True)     

        # If it is daytime, also run over the solar spectrum for both.
        if (not b_nighttime):
            subprocess.run([uvspec_path], stdin=open(input_solar_con, "rb"), 
                           stdout=open(output_solar_con, "wb"), check=True)
            subprocess.run([uvspec_path], stdin=open(input_solar_clr, "rb"), 
                           stdout=open(output_solar_clr, "wb"), check=True)               

    dfs_thermal_clr = []
    dfs_thermal_con = []
    dfs_solar_con = []
    dfs_solar_clr = []

    # Read the results in for all libRadtran runs.
    for i in range(0, len(xs)):
        if (not b_nighttime):

            output_solar_con = Path(result_dir) / f"result_conSolar{i}.DAT"
            df = pd.read_csv(output_solar_con, header=None, sep =r'\s+')
            dfs_solar_con.append(df)

            output_solar_clr = Path(result_dir) / f"result_clrSolar{i}.DAT"
            df = pd.read_csv(output_solar_clr, header=None, sep =r'\s+')
            dfs_solar_clr.append(df)

        output_thermal_con  = Path(result_dir) / f"result_conThermal{i}.DAT"
        df = pd.read_csv(output_thermal_con, header=None, sep =r'\s+')
        dfs_thermal_con.append(df)

        output_thermal_clr  = Path(result_dir) / f"result_clrThermal{i}.DAT"
        df = pd.read_csv(output_thermal_clr, header=None, sep =r'\s+')
        dfs_thermal_clr.append(df)

    diffs_con = []
    diffs_clr = []

    for i in range(0, len(xs)):
        # Calculate the downward minus upward fluxes for both cases.
        df_thermal_con = dfs_thermal_con[i].values[0]
        df_thermal_clr = dfs_thermal_clr[i].values[0]

        # df_x_y[1] is the downward flux at TOA.
        # df_x_y[3] is the upward flux at TOA.
        # Hence, diff_x_y is equal to the downward flux minus the upward flux.
        diff_thermal_con = (df_thermal_con[1]) - (df_thermal_con[3])
        diff_thermal_clr = (df_thermal_clr[1]) - (df_thermal_clr[3])

        if (b_nighttime):
            diffs_con.append(diff_thermal_con)
            diffs_clr.append(diff_thermal_clr)

        else:
            df_solar_con = dfs_solar_contrail[i].values[0]
            df_solar_clr = dfs_solar_clear[i].values[0]
            
            diff_solar_con = (df_solar_con[1]) - (df_solar_con[3])
            diff_solar_clr = (df_solar_clr[1]) - (df_solar_clr[3])

            diffs_con.append(diff_solar_con + diff_thermal_con)
            diffs_clr.append(diff_solar_clr + diff_thermal_clr)
            
    # Calculate the difference in net flux due to contrail.
    diff = np.array(diffs_con) - np.array(diffs_clr)

    # TODO Change this to use sampling rather than the current implementation.
    width = xs[0][-1] - xs[0][0]
    w_per_m = diff * np.array(width)
    total_w_per_m = np.sum(w_per_m)

    return total_w_per_m

def calc_sample(apce_data, sample, met_albedo, ds_temp, path_start):

    ds_t = apce_data.ds_t
    t = apce_data.t
    
    timestep = t[1] - t[0]

    latitude, longitude = gen_lat_lon(sample)     
    altitude = sample["altitude"]       

    solar_source_path = "../libRadtran-2.0.6/data/solar_flux/atlas_plus_modtran"
    data_files_path = "../libRadtran-2.0.6/data"
    atmosphere_file_path = "../libRadtran-2.0.6/data/atmmod/afglus.dat"            

    samples = 5
    j=0

    total_w_per_m_s = []

    for ds_tt in ds_t:

        # Set age (e.g. 10 mins, 20 mins ...)
        age = t[j]
        print(age)

        # Set the actual time in the pandas format
        sample_time = sample["time"] + pd.Timedelta(minutes=age)

        # Set the albedo value based on the albedo dataset - this assumes albedo is 
        # not a function of time in order to reduce the number of dataset downloads
        albedo = met_albedo['fal'].sel(longitude=sample["longitude"], latitude=sample["latitude"], time ='2024-03-01T13:00:00.000000000', level=-1, method='nearest').values
       
        # Get the dataset for clouds and temperature for the whole atmosphere along
        # the advected path of the contrail
        ds_temp_tt = ds_temp.sel(time = age/60, method='nearest')

        # Get the sample time in the format required by libRadtran i.e. "YYYY MM DD HH MM SS"
        sample_time_array = [sample_time.year, sample_time.month, sample_time.day, sample_time.hour, sample_time.minute, sample_time.second]
        sample_time_format = (f"{sample_time_array[0]} {sample_time_array[1]} {sample_time_array[2]} {sample_time_array[3]} {sample_time_array[4]} {sample_time_array[5]} ")

        # Check whether it is night or not and set the boolean value
        b_nighttime = check_night(sample_time, sample["latitude"], sample["longitude"])

        # Split the APCEMM output based on the number of samples, and return lists of IWCs, Eff_rads, xs and ys
        contrail_IWCs,     xs, ys = generate_indices(ds_tt["x"], ds_tt["y"], ds_tt["IWC"], samples)
        contrail_Eff_rads, xs, ys = generate_indices(ds_tt["x"], ds_tt["y"], ds_tt["Effective radius"], samples)

        # Average the IWC and Eff_rads over the columns
        contrail_IWCs_avg     = average_columns(contrail_IWCs)
        contrail_Eff_rads_avg = average_columns(contrail_Eff_rads)

        # Adjust the altitudes based on cruise altitude
        contrail_IWCs_avg, contrail_Eff_rads_avg, ys = adjust_altitude(contrail_IWCs_avg, contrail_Eff_rads_avg, ys, altitude)

        # Get the LWC and IWC for the natural clouds
        cloud_LWC_cols = (np.repeat((ds_temp_tt["cloud_LWC"].values)[:,np.newaxis], samples-1, axis=1)).T
        cloud_IWC_cols = (np.repeat((ds_temp_tt["cloud_IWC"].values)[:,np.newaxis], samples-1, axis=1)).T
        max_cloud_cover = (ds_temp_tt["fraction_of_cloud_cover"]).max()
        cloud_ys = ds_temp_tt["altitude"].values

        # Generate random numbers that don't repeat. Then, calculate
        # the number of columns which should be set to zero (due) to
        # the cloud overlap, and then set them to zero
        list_of_ints = np.arange(0,samples-1)
        np.random.shuffle(list_of_ints)
        for i in range(0,round((1-max_cloud_cover.values) * samples)-1):
            cloud_LWC_cols[list_of_ints[i]][:] = 0
            cloud_IWC_cols[list_of_ints[i]][:] = 0

        habit = write_cloud_files(contrail_IWCs_avg, contrail_Eff_rads_avg, cloud_LWC_cols, cloud_IWC_cols, ys, cloud_ys, age, path_start)

        write_inp_files(atmosphere_file_path, data_files_path, solar_source_path, sample_time_format, latitude, longitude, albedo, habit, xs, path_start)

        total_w_per_m = run_libradtran(xs, b_nighttime, path_start)
        total_w_per_m_s.append(total_w_per_m)
        
        j = j + 1

    j_per_m = sum(total_w_per_m_s) * timestep

    return j_per_m, age
