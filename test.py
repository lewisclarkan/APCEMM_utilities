import netCDF4 as nc
import xarray as xr
import os 
import numpy as np
import matplotlib.pyplot as plt

#ds = xr.open_dataset('RF_interpolation_data/yang2013_aggregates.nc', engine="netcdf4")

#print(ds.solar_zenith_angle)
#print(ds.ice_cloud_temp)
#print(ds.surface_albedo)
#print(ds.ice_water_content)
#print(ds.surface_temperature)
#print(ds.crystal_effective_radius)
#print(ds.optical_thickness_liquid_water_cloud)
#print(ds.cloud_fraction)

#temp = ds.interp(solar_zenith_angle=12.5, ice_cloud_temp=220, surface_albedo=0.2, ice_water_content = 0.01, surface_temperature=275, crystal_effective_radius=7.5, optical_thickness_liquid_water_cloud=3, cloud_fraction=0.2)
#print(temp.values)

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

ds_t = (read_apcemm_data("./results/month_1/19_20/APCEMM_results/APCEMM_out_19/").ds_t)

"""['r_e', 'Pressure', 'Altitude', 'H2O', 'Temperature', 'Ice aerosol particle number', 
'Ice aerosol surface area', 'Ice aerosol volume', 'Effective radius', 'Horizontal optical depth', 
'Vertical optical depth', 'Overall size distribution', 'Ice Mass', 'Number Ice Particles', 'Extinction', 
'IWC', 'RHi', 'width', 'depth', 'intOD'] """

#Looking at a different 2D contour, IWC over time. Notice the contrast in ice particles vs ice mass distribution.
fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi=200, figsize=[6, 6])
fig.suptitle("Ice Water Content")

ds_t_1h = ds_t[6]
X_map1h, Y_map1h = np.meshgrid(ds_t_1h["x"], ds_t_1h["y"])
map1h = ax.contourf(X_map1h, Y_map1h, ds_t_1h['IWC'], 1000, vmin=0)
ax.set_title('t = 1h')
ax.set_xlim(-2750,500)
ax.set_ylim(-200,50)
plt.colorbar(map1h, ax=ax,fraction=0.046, pad=0.1)


plt.savefig("temp.png")