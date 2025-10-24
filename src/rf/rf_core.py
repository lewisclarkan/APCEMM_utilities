import os
import xarray as xr
import numpy as np
import pandas as pd
from pysolar.solar import *

from src.rf.rf_io import *

# Functions used for both pre- and post-processing

def removeLow(arr, cutoff = 1e-3):
    func = lambda x: (x > cutoff) * x
    vfunc = np.vectorize(func)
    return vfunc(arr)

def generate_indices(x, y, quantity, samples):

    indices = []
    quantity_s = []
    xs = []

    for i in range(0, samples):
        index = i * int(len(quantity)/samples)
        indices.append(index)

    for i in range(0,len(indices)-1):
        quantity_s.append(quantity[:, indices[i]:indices[i+1]+1:1])
        xs.append(x[indices[i]:indices[i+1]+1:1])

    ys = y.values

    return quantity_s, xs, ys

def average_columns(quantity):

    quantity_avg = []

    for i in range(0, len(quantity)):
        temp = []
        for j in quantity[i]:
            if j.size > 0:
                temp.append((j.values).mean())
            else:
                temp.append(0)
        quantity_avg.append(temp)

    return quantity_avg

def adjust_altitude(IWCs_avg, Eff_rads_avg, ys, base_altitude):

    ys = (ys + base_altitude)/1000
    ys = np.concatenate(([0, min(ys)-1/1000],ys))

    temp = [0,0]

    Eff_rads_temp_avg = []
    IWCs_temp_avg = []

    for i in range(0, len(Eff_rads_avg)):
        Eff_rads_temp_avg.append(np.concatenate((temp, Eff_rads_avg[i])) * 10 ** 6)
        IWCs_temp_avg.append(np.concatenate((temp, IWCs_avg[i]))*1000)


    IWCs_avg = IWCs_temp_avg
    Eff_rads_avg = Eff_rads_temp_avg

    return IWCs_avg, Eff_rads_avg, ys

def run_libradtran(xs, b_nighttime, path_start):

    try: 
        os.makedirs(f"./{path_start}/result")
    except:
        pass

    for i in range(0, len(xs)):
    
        clouds = (f"./{path_start}/clouds/cloud{i}.DAT")

        inpThermal = (f"./{path_start}/inps/runThermal{i}.DAT")
        inpSolar   = (f"./{path_start}/inps/runSolar{i}.DAT")

        inp_clearThermal    = (f"./{path_start}/inps/run_clearThermal{i}.DAT")
        inp_clearSolar      = (f"./{path_start}/inps/run_clearSolar{i}.DAT")

        diector1 = (f"./{path_start}/result/result_conThermal{i}.DAT")
        diector2 = (f"./{path_start}/result/result_conSolar{i}.DAT")
        diector3 = (f"./{path_start}/result/result_clrThermal{i}.DAT")
        diector4 = (f"./{path_start}/result/result_clrSolar{i}.DAT")

        # If it is daytime, we need to run the solar cases
        if (not b_nighttime):
            os.system(f"../libRadtran-2.0.6/bin/uvspec < {inpSolar} > {diector2}")
            os.system(f"../libRadtran-2.0.6/bin/uvspec < {inp_clearSolar} > {diector4}")

    
        os.system(f"../libRadtran-2.0.6/bin/uvspec < {inp_clearThermal} > {diector3}")
        os.system(f"../libRadtran-2.0.6/bin/uvspec < {inpThermal} > {diector1}")

    conThermal_results = []
    conSolar_results = []

    clrThermal_results = []
    clrSolar_results = []

    for i in range(0, len(xs)):

        if (not b_nighttime):
            diector2 = (f"./{path_start}/result/result_conSolar{i}.DAT")
            diector4 = (f"./{path_start}/result/result_clrSolar{i}.DAT")

            df2 = pd.read_csv(diector2, header=None, sep ='\s+')
            conSolar_results.append(df2)

            df4 = pd.read_csv(diector4, header=None, sep ='\s+')
            clrSolar_results.append(df4)

        diector1 = (f"./{path_start}/result/result_conThermal{i}.DAT")
        diector3 = (f"./{path_start}/result/result_clrThermal{i}.DAT")

        df = pd.read_csv(diector1, header=None, sep ='\s+')
        conThermal_results.append(df)

        df3 = pd.read_csv(diector3, header=None, sep ='\s+')
        clrThermal_results.append(df3)

    con_down_minus_up = []
    clr_down_minus_up = []

    for i in range(0, len(xs)):

        if (b_nighttime):

            con_down_minus_up.append((conThermal_results[i].values[0][1]) - (conThermal_results[i].values[0][3]))
            clr_down_minus_up.append((clrThermal_results[i].values[0][1]) - (clrThermal_results[i].values[0][3]))

        else:

            con_down_minus_up.append((conThermal_results[i].values[0][1] + conSolar_results[i].values[0][1]) - (conThermal_results[i].values[0][3] + conSolar_results[i].values[0][3]))
            clr_down_minus_up.append((clrThermal_results[i].values[0][1] + clrSolar_results[i].values[0][1]) - (clrThermal_results[i].values[0][3] + clrSolar_results[i].values[0][3]))

    diff = np.array(con_down_minus_up) - np.array(clr_down_minus_up)
    width = xs[0][-1] - xs[0][0]
    w_per_m = diff * np.array(width)
    total_w_per_m = np.sum(w_per_m)

    return total_w_per_m

def gen_lat_lon(sample):

    def decdeg2dms(dd):
        mult = -1 if dd < 0 else 1
        mnt,sec = divmod(abs(dd)*3600, 60)
        deg,mnt = divmod(mnt, 60)
        return mult*deg, mult*mnt, mult*sec
    
    temp_lat = decdeg2dms(sample["latitude"])
    temp_lon = decdeg2dms(sample["longitude"])

    if temp_lat[0] < 0:
        N_or_S = "S"
    else:
        N_or_S = "N"

    temp_lat=np.absolute(np.array(temp_lat))
    latitude = f"{N_or_S} {np.round(temp_lat[0],0):.0f} {np.round(temp_lat[1],0):.0f} {np.round(temp_lat[2],1):.1f}"

    if temp_lon[0] < 0:
        E_or_W = "W"
    else:
        E_or_W = "E"

    temp_lon=np.absolute(np.array(temp_lon))
    longitude = f"{E_or_W} {np.round(temp_lon[0],0):.0f} {np.round(temp_lon[1],0):.0f} {np.round(temp_lon[2],1):.1f}"
    
    return latitude, longitude

def check_night(sample_time, latitude, longitude):
    """Returns TRUE if the time is night, False otherwise"""

    dobj = datetime.datetime(sample_time.year, sample_time.month, sample_time.day, sample_time.hour, sample_time.minute, sample_time.second, 0, tzinfo=datetime.timezone.utc)
    solar_elevation = get_altitude(latitude, longitude, dobj)

    if solar_elevation <= 2:
        return True
    else:
        return False

def get_ice_habit(age):

    if age<=20:
        ice_habit = "droxtal"

    else:
        ice_habit = "rough-aggregate"

    return ice_habit

def calc_sample(apce_data, sample, met_albedo, ds_temp, path_start):

    ds_t = apce_data.ds_t
    t = apce_data.t
    
    print(t)

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
        print(ds_tt["depth"])

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



"""if __name__ == "__main__":

    sample_time = pd.Timestamp("2019-01-01 04:20:00")
    latitude = 50.391472
    longitude = 12.512389

    print(check_night(sample_time, latitude, longitude))"""