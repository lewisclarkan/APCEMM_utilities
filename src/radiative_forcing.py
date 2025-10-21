import os
import xarray as xr
import numpy as np
import pandas as pd
from pysolar.solar import *


# Functions used for both pre- and post-processing

class apce_data_struct:
    def __init__(self, t, ds_t, icemass, h2omass, numparts):
        self.t = t
        self.ds_t = ds_t
        self.icemass = icemass
        self.h2omass = h2omass
        self.numparts = numparts
    
def read_apcemm_data(directory):
    t_mins = []
    ds_t = []
    ice_mass = []
    total_h2o_mass = []
    num_particles = []

    for file in sorted(os.listdir(directory)):
        if(file.startswith('ts_aerosol') and file.endswith('.nc')):
            file_path = os.path.join(directory,file)
            ds = xr.open_dataset(file_path, engine = "netcdf4", decode_times = False)
            ds_t.append(ds)
            tokens = file_path.split('.')
            mins = int(tokens[-2][-2:])
            hrs = int(tokens[-2][-4:-2])
            t_mins.append(hrs*60 + mins)

            ice_mass.append(ds["Ice Mass"])
            num_particles.append(ds["Number Ice Particles"])
            dx = abs(ds["x"][-1] - ds["x"][0])/len(ds["x"])
            dy = abs(ds["y"][-1] - ds["y"][0])/len(ds["y"])
            
            h2o_mass = np.sum(ds["H2O"]) * 1e6 / 6.022e23 * 0.018 * dx*dy + ds["Ice Mass"]
            total_h2o_mass.append(h2o_mass.values)
            
    return apce_data_struct(t_mins, ds_t, ice_mass, total_h2o_mass, num_particles)

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

def write_cloud_files(contrail_IWCs_avg, contrail_Eff_rads_avg, cloud_LWC_cols, cloud_IWC_cols, ys, cloud_ys, age, path_start):

    clouds_dir = (f"./{path_start}/clouds")

    habit = get_ice_habit(age)

    try: 
        os.makedirs(clouds_dir)
    except:
        pass

    for i in range(0,len(contrail_IWCs_avg)):

        indices = []
        file_name = (f"./{path_start}/clouds/ice_cloud{i}.DAT")
        file_name_empty = (f"./{path_start}/clouds/ice_cloud_empty{i}.DAT")
        file_name_water = (f"./{path_start}/clouds/water_cloud{i}.DAT")

        eff_rad_switch = 80 # CURRENTLY THIS IS ARBITRARY

        natural_IC_Eff_rad = 25
        natural_WC_Eff_rad = 14

        with open(file_name, "w") as f:
            f.write("#      z         IWC          R_eff\n")
            f.write("#     (km)     (g/m^3)         (um)\n")

            max_eff_rad = np.max(contrail_Eff_rads_avg)

            if max_eff_rad > eff_rad_switch:
                habit = "droxtal"
                lower_limit = 9.48
                upper_limit = 293.1

            else:
                habit = "rough-aggregate"
                lower_limit = 3.55
                upper_limit = 108.1

            for j in range(len(ys)-1, 0, -1):
                if (contrail_Eff_rads_avg[i][j] < lower_limit): 
                    contrail_IWCs_avg[i][j]=0
                elif (contrail_Eff_rads_avg[i][j] > upper_limit):
                    contrail_Eff_rads_avg[i][j]=upper_limit
                f.write(f"     {ys[j]:.3f}   {contrail_IWCs_avg[i][j]:.9f}   {contrail_Eff_rads_avg[i][j]:.9f}\n")

                minimum_y = ys[j] - 0.1

            for k in range(0,len(cloud_ys)):
                if cloud_ys[k] > minimum_y:
                    indices.append(k)

            cloud_IWC_cols_new_i = []
            cloud_ys_new_i = []
            for k in range(0,len(cloud_ys)):
                cloud_IWC_cols_new_i = (np.delete(cloud_IWC_cols[i],indices))
                cloud_ys_new_i = (np.delete(cloud_ys,indices))

            f.write(f"     {minimum_y:.3f}   {0:.3f}   {natural_IC_Eff_rad:.9f}\n")
            for k in range(len(cloud_IWC_cols_new_i)-1, 0 , -1):
                f.write(f"     {cloud_ys_new_i[k]:.3f}   {cloud_IWC_cols_new_i[k]:.9f}   {natural_IC_Eff_rad:.9f}\n")
                
        with open(file_name_empty, "w") as f:
            f.write("#      z         IWC          R_eff\n")
            f.write(f"     {minimum_y:.3f}   {0:.3f}   {natural_IC_Eff_rad:.9f}\n")
            for k in range(len(cloud_IWC_cols_new_i)-1, 0 , -1):
                f.write(f"     {cloud_ys_new_i[k]:.3f}   {cloud_IWC_cols_new_i[k]:.9f}   {natural_IC_Eff_rad:.9f}\n")

        with open(file_name_water, "w") as f:
            f.write("#      z         LWC          R_eff\n")
            f.write("#      z         IWC          R_eff\n")
            for k in range(len(cloud_ys)-1, 0 , -1):
                f.write(f"     {cloud_ys[k]:.3f}   {cloud_LWC_cols[i][k]:.9f}   {natural_WC_Eff_rad:.9f}\n")

    return habit

def write_inp_files(atmosphere_file_path, data_files_path, solar_source_path, time, latitude, longitude, albedo, ice_habit, xs, path_start):

    solver = "disort"
    correlated_k = "fu"
    ice_parameterisation = "yang"

    try: 
        os.makedirs(f"./{path_start}/inps")
    except:
        pass

    for i in range(0, len(xs)):

        ice_clouds = (f"./{path_start}/clouds/ice_cloud{i}.DAT")
        empty_ice_clouds = (f"./{path_start}/clouds/ice_cloud_empty{i}.DAT")
        water_clouds = (f"./{path_start}/clouds/water_cloud{i}.DAT")

        inpThermal = (f"./{path_start}/inps/runThermal{i}.DAT")
        inpSolar   = (f"./{path_start}/inps/runSolar{i}.DAT")

        inp_clearThermal    = (f"./{path_start}/inps/run_clearThermal{i}.DAT")
        inp_clearSolar      = (f"./{path_start}/inps/run_clearSolar{i}.DAT")

        with open(inpThermal, "w") as file:

            file.write(f"data_files_path {data_files_path}\n")
            file.write("source thermal\n")
            file.write("\n")

            file.write(f"time {time}\n")
            file.write(f"latitude {latitude}\n")
            file.write(f"longitude {longitude}\n")
            file.write(f"albedo {albedo}\n")
            file.write("\n")

            file.write(f"rte_solver {solver}\n")
            file.write("pseudospherical\n")
            file.write(f"mol_abs_param {correlated_k}\n")
            file.write("\n")

            file.write(f"ic_file 1D {ice_clouds}\n")
            file.write(f"ic_properties {ice_parameterisation}\n")
            file.write(f"ic_habit {ice_habit}\n")
            file.write("\n")

            file.write(f"wc_file 1D {water_clouds}\n")
            file.write("\n")

            file.write("zout TOA\n")
            file.write("output_process sum\n")
            file.write("quiet\n")

        with open(inpSolar, "w") as file:

            file.write(f"data_files_path {data_files_path}\n")
            file.write(f"source solar {solar_source_path}\n")
            file.write("\n")

            file.write(f"time {time}\n")
            file.write(f"latitude {latitude}\n")
            file.write(f"longitude {longitude}\n")
            file.write(f"albedo {albedo}\n")
            file.write("\n")

            file.write(f"rte_solver {solver}\n")
            file.write("pseudospherical\n")
            file.write(f"mol_abs_param {correlated_k}\n")
            file.write("\n")

            file.write(f"ic_file 1D {ice_clouds}\n")
            file.write(f"ic_properties {ice_parameterisation}\n")
            file.write(f"ic_habit {ice_habit}\n")
            file.write("\n")

            file.write(f"wc_file 1D {water_clouds}\n")
            file.write("\n")

            file.write("zout TOA\n")
            file.write("output_process sum\n")
            file.write("quiet\n")

        with open(inp_clearThermal, "w") as file:

            file.write(f"data_files_path {data_files_path}\n")
            file.write("source thermal\n")
            file.write("\n")

            file.write(f"time {time}\n")
            file.write(f"latitude {latitude}\n")
            file.write(f"longitude {longitude}\n")
            file.write(f"albedo {albedo}\n")
            file.write("\n")

            file.write(f"rte_solver {solver}\n")
            file.write("pseudospherical\n")
            file.write(f"mol_abs_param {correlated_k}\n")
            file.write("\n")

            file.write(f"ic_file 1D {empty_ice_clouds}\n")
            file.write(f"ic_properties {ice_parameterisation}\n")
            file.write(f"ic_habit {ice_habit}\n")
            file.write("\n")

            file.write(f"wc_file 1D {water_clouds}\n")
            file.write("\n")

            file.write("zout TOA\n")
            file.write("output_process sum\n")
            file.write("quiet\n")

        with open(inp_clearSolar, "w") as file:

            file.write(f"data_files_path {data_files_path}\n")
            file.write(f"source solar {solar_source_path}\n")
            file.write("\n")

            file.write(f"time {time}\n")
            file.write(f"latitude {latitude}\n")
            file.write(f"longitude {longitude}\n")
            file.write(f"albedo {albedo}\n")
            file.write("\n")

            file.write(f"rte_solver {solver}\n")
            file.write("pseudospherical\n")
            file.write(f"mol_abs_param {correlated_k}\n")
            file.write("\n")

            file.write(f"ic_file 1D {empty_ice_clouds}\n")
            file.write(f"ic_properties {ice_parameterisation}\n")
            file.write(f"ic_habit {ice_habit}\n")
            file.write("\n")

            file.write(f"wc_file 1D {water_clouds}\n")
            file.write("\n")

            file.write("zout TOA\n")
            file.write("output_process sum\n")
            file.write("quiet\n")

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