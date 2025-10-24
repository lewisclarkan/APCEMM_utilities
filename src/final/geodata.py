import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import zarr
import yaml

import cdsapi

#from aircraft import set_flight_parameters

from pycontrails import DiskCacheStore
from pycontrails import Flight
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.core import met_var, GeoVectorDataset, models
from pycontrails.physics import constants, thermo, units
from pycontrails.datalib.ecmwf import ERA5ModelLevel, ERA5
from pycontrails import MetDataset
from pycontrails.models.apcemm import utils
from typing import Any
from scipy.interpolate import NearestNDInterpolator


def open_dataset(sample):

    s_index, s_longitude, s_latitude, s_altitude, s_time, s_type = sample

    max_life = 12

    time = (str(s_time), str(s_time + np.timedelta64(max_life, 'h')))

    era5ml = ERA5ModelLevel(
        time=time,
        variables=("t", "q", "u", "v", "w"),
        grid=1,  # horizontal resolution, 0.25 by default
        model_levels=range(70, 91),
        pressure_levels=np.arange(170, 400, 10),
        cachestore = DiskCacheStore(cache_dir = "../../../cache"),
        cache_download=True,
    )
    met_t = era5ml.open_metdataset(xr_kwargs={"engine":"h5netcdf"})

    geopotential = met_t.data.coords["altitude"].data

    temp1 = np.repeat(geopotential, len(met_t.data.coords["time"]))
    temp2 = np.tile(temp1, len(met_t.data.coords["longitude"])*len(met_t.data.coords["latitude"]))

    geopotential_4d = np.reshape(temp2, (len(met_t.data.coords["longitude"]),len(met_t.data.coords["latitude"]),len(met_t.data.coords["level"]),len(met_t.data.coords["time"])))

    ds = met_t.data.assign(geopotential_height=(met_t.data["air_temperature"].dims, geopotential_4d))

    met = MetDataset(ds)

    return met

def get_temperature_and_clouds_met(sample):

    s_index, s_longitude, s_latitude, s_altitude, s_time, s_type = sample

    max_life = 12

    time = (str(s_time), str(s_time + np.timedelta64(max_life, 'h')))

    era5ml = ERA5ModelLevel(
        time=time,
        variables=["t","clwc","ciwc","q","cc"],
        model_levels=np.arange(1, 138, 1),
        pressure_levels=[1000,975,950,925,900,875,850,825,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,70,50,30,20,10,5,1],
        grid=1,
        cachestore = DiskCacheStore(cache_dir="../../../cache"),
        cache_download=True,
        ) 
    
    met_t = era5ml.open_metdataset(xr_kwargs={"engine":"h5netcdf"})

    geopotential = met_t.data.coords["altitude"].data

    temp1 = np.repeat(geopotential, len(met_t.data.coords["time"]))
    temp2 = np.tile(temp1, len(met_t.data.coords["longitude"])*len(met_t.data.coords["latitude"]))

    geopotential_4d = np.reshape(temp2, (len(met_t.data.coords["longitude"]),len(met_t.data.coords["latitude"]),len(met_t.data.coords["level"]),len(met_t.data.coords["time"])))

    ds = met_t.data.assign(geopotential_height=(met_t.data["air_temperature"].dims, geopotential_4d))

    met = MetDataset(ds)

    return met

def get_albedo(path):

    #client = cdsapi.Client()

    #dataset = 'reanalysis-era5-single-levels'
    #request = {
    #    'product_type': ['reanalysis'],
    #    'variable': ['forecast_albedo'],
    #    'year': ['2024', '2024'],
    #    'month': ['03', '03'],
    #    'day': ['01', '01'],
    #    'time': ['13:00', '14:00'],
    #    'data_format': 'grib',
    #}
    target = path

    #client.retrieve(dataset, request, target)

    ds = xr.load_dataset(path, engine="cfgrib")
    ds = ds.expand_dims({'level':[-1]})

    met = MetDataset(ds)

    return met

def normal_wind_shear(
    u_hi,
    u_lo,
    v_hi,
    v_lo,
    azimuth,
    dz: float,
):
    r"""Compute segment-normal wind shear from wind speeds at lower and upper levels.

    Parameters
    ----------
    u_hi : ArrayScalarLike
        Eastward wind at upper level [:math:`m/s`]
    u_lo : ArrayScalarLike
        Eastward wind at lower level [:math:`m/s`]
    v_hi : ArrayScalarLike
        Northward wind at upper level [:math:`m/s`]
    v_lo : ArrayScalarLike
        Northward wind at lower level [:math:`m/s`]
    azimuth : ArrayScalarLike
        Segment azimuth [:math:`\deg`]
    dz : float
        Distance between upper and lower level [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        Segment-normal wind shear [:math:`1/s`]
    """
    du_dz = (u_hi - u_lo) / dz
    dv_dz = (v_hi - v_lo) / dz
    az_radians = units.degrees_to_radians(azimuth)
    sin_az = np.sin(az_radians)
    cos_az = np.cos(az_radians)
    return sin_az * dv_dz - cos_az * du_dz

def generate_temp_profile(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    azimuth: np.ndarray,
    altitude: np.ndarray,
    met: MetDataset,
    humidity_scaling,
    dz_m: float,
    interp_kwargs: dict[str, Any],
) -> xr.Dataset:

    # Ensure that altitudes are sorted ascending
    altitude = np.sort(altitude)

    # Check for required fields in met
    vars = (
        met_var.AirTemperature,
        met_var.GeopotentialHeight,
        met_var.SpecificHumidity,
    )
    met.ensure_vars(vars)
    met.standardize_variables(vars)

    # Flatten input arrays
    time = time.ravel()
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    azimuth = azimuth.ravel()
    altitude = altitude.ravel()

    # Estimate pressure levels close to target altitudes
    # (not exact because this assumes the ISA temperature profile)
    pressure = units.m_to_pl(altitude) * 1e2

    # Broadcast to required shape and create vector for initial interpolation
    # onto original pressure levels at target horizontal location.
    shape = (time.size, altitude.size)
    time = np.broadcast_to(time[:, np.newaxis], shape).ravel()
    longitude = np.broadcast_to(longitude[:, np.newaxis], shape).ravel()
    latitude = np.broadcast_to(latitude[:, np.newaxis], shape).ravel()
    azimuth = np.broadcast_to(azimuth[:, np.newaxis], shape).ravel()
    level = np.broadcast_to(pressure[np.newaxis, :] / 1e2, shape).ravel()
    vector = GeoVectorDataset(
        data={"azimuth": azimuth},
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
    )

    # Downselect met before interpolation
    met = vector.downselect_met(met)

    # Interpolate meteorology data onto vector
    scale_humidity = humidity_scaling is not None and "specific_humidity" not in vector
    for met_key in (
        "air_temperature",
        "geopotential_height",
        "specific_cloud_liquid_water_content",
        "specific_cloud_ice_water_content",
        "specific_humidity",
        "fraction_of_cloud_cover"
    ):
        models.interpolate_met(met, vector, met_key, **interp_kwargs)

    # Interpolate winds at lower level for shear calculation
    air_pressure_lower = thermo.pressure_dz(vector["air_temperature"], vector.air_pressure, dz_m)
    lower_level = air_pressure_lower / 100.0

    # Apply humidity scaling
    if scale_humidity and humidity_scaling is not None:
        humidity_scaling.eval(vector, copy_source=False)

    # Compute RHi and segment-normal shear

    # Reshape interpolated fields to (time, level).
    nlev = altitude.size
    ntime = len(vector) // nlev
    shape = (ntime, nlev)
    time = np.unique(vector["time"])
    time = (time - time[0]) / np.timedelta64(1, "h")
    temperature = vector["air_temperature"].reshape(shape)
    specific_cloud_liquid_water_content = vector["specific_cloud_liquid_water_content"].reshape(shape)
    specific_cloud_ice_water_content = vector["specific_cloud_ice_water_content"].reshape(shape)
    specific_humidity = vector["specific_humidity"].reshape(shape)
    fraction_of_cloud_cover = vector["fraction_of_cloud_cover"].reshape(shape)
    z = vector["geopotential_height"].reshape(shape)

    # Interpolate fields to target altitudes profile-by-profile
    # to obtain 2D arrays with dimensions (time, altitude).
    temperature_on_z = np.zeros(shape, dtype=temperature.dtype)
    specific_cloud_liquid_water_content_on_z = np.zeros(shape, dtype=specific_cloud_liquid_water_content.dtype)
    specific_cloud_ice_water_content_on_z = np.zeros(shape, dtype=specific_cloud_ice_water_content.dtype)
    specific_humidity_on_z = np.zeros(shape, dtype=specific_cloud_ice_water_content.dtype)
    fraction_of_cloud_cover_on_z = np.zeros(shape, dtype=fraction_of_cloud_cover.dtype)

    # Fields should already be on pressure levels close to target
    # altitudes, so this just uses linear interpolation and constant
    # extrapolation on fields expected by APCEMM.
    # NaNs are preserved at the start and end of interpolated profiles
    # but removed in interiors.
    def interp(z: np.ndarray, z0: np.ndarray, f0: np.ndarray) -> np.ndarray:
        # mask nans
        mask = np.isnan(z0) | np.isnan(f0)
        if np.all(mask):
            msg = (
                "Found all-NaN profile during APCEMM meterology input file creation. "
                "MetDataset may have insufficient spatiotemporal coverage."
            )
            raise ValueError(msg)
        z0 = z0[~mask]
        f0 = f0[~mask]

        # interpolate
        assert np.all(np.diff(z0) > 0)  # expect increasing altitudes
        fi = np.interp(z, z0, f0, left=f0[0], right=f0[-1])

        # restore nans at start and end of profile
        if mask[0]:  # nans at top of profile
            fi[z > z0.max()] = np.nan
        if mask[-1]:  # nans at end of profile
            fi[z < z0.min()] = np.nan
        return fi

    # The manual for loop is unlikely to be a bottleneck since a
    # substantial amount of work is done within each iteration.
    for i in range(ntime):
        temperature_on_z[i, :] = interp(altitude, z[i, :], temperature[i, :])
        specific_cloud_liquid_water_content_on_z[i, :] = interp(altitude, z[i, :], specific_cloud_liquid_water_content[i, :])
        specific_cloud_ice_water_content_on_z[i, :] = interp(altitude, z[i, :], specific_cloud_ice_water_content[i, :])
        specific_humidity_on_z[i, :] = interp(altitude, z[i, :], specific_humidity[i, :])
        fraction_of_cloud_cover_on_z[i, :] = interp(altitude, z[i, :], fraction_of_cloud_cover[i, :])

    # APCEMM also requires initial pressure profile
    pressure_on_z = interp(altitude, z[0, :], pressure)

    # Create APCEMM input dataset.
    # Transpose require because APCEMM expects (altitude, time) arrays.
    return xr.Dataset(
        data_vars={"pressure": (("altitude",), pressure_on_z.astype("float32") / 1e2, {"units": "hPa"}),
            "temperature": (("altitude", "time"),temperature_on_z.astype("float32").T,{"units": "K"}),
            "specific_cloud_liquid_water_content": (("altitude", "time"),specific_cloud_liquid_water_content_on_z.astype("float32").T,{"units": "kg/kg"}),  
            "specific_cloud_ice_water_content": (("altitude", "time"),specific_cloud_ice_water_content_on_z.astype("float32").T,{"units": "kg/kg"}),
            "specific_humidity": (("altitude", "time"),specific_humidity.astype("float32").T,{"units": "kg/kg"}),
            "fraction_of_cloud_cover" : (("altitude", "time"), fraction_of_cloud_cover_on_z.astype("float32").T, {"units": "kg/kg"}),
            },
        coords={
            "altitude": ("altitude", altitude.astype("float32") / 1e3, {"units": "km"}),
            "time": ("time", time, {"units": "hours"}),
        },
    )

def generate_apcemm_input_met(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    azimuth: np.ndarray,
    altitude: np.ndarray,
    met: MetDataset,
    humidity_scaling,
    dz_m: float,
    interp_kwargs: dict[str, Any],
) -> xr.Dataset:
    r"""Create xarray Dataset for APCEMM meteorology netCDF file.

    This dataset contains a sequence of atmospheric profiles along the
    Lagrangian trajectory of an advected flight segment. The along-trajectory
    dimension is parameterized by time (rather than latitude and longitude),
    so the dataset coordinates are air pressure and time.

    Parameters
    ----------
    time : np.ndarray
        Time coordinates along the Lagrangian trajectory of the advected flight segment.
        Values must be coercible to ``np.datetime64`` by :class:`GeoVectorDataset`.
        Will be flattened before use if not 1-dimensional.
    longitude : np.ndarray
        Longitude [WGS84] along the Lagrangian trajectory of the advected flight segment.
        Defines the longitude of the trajectory at each time and should have the
        same shape as :param:`time`
        Will be flattened before use if not 1-dimensional.
    latitude : np.ndarray
        Latitude [WGS84] along the Lagrangian trajectory of the advected flight segment.
        Defines the longitude of the trajectory at each time and should have the
        same shape as :param:`time`
        Will be flattened before use if not 1-dimensional.
    azimuth : np.ndarray
        Azimuth [:math:`\deg`] of the advected flight segment at each point along its
        Lagrangian trajectory. Note that the azimuth defines the orientation of the
        advected segment itself, and not the direction in which advection is transporting
        the segment. The azimuth is used to convert horizontal winds into segment-normal
        wind shear. Must have the same shape as :param:`time`.
        Will be flattened before use if not 1-dimensional.
    altitude : np.ndarray
        Defines altitudes [:math:`m`] on which atmospheric profiles are computed.
        Profiles are defined using the same set of altitudes at every point
        along the Lagrangian trajectory of the advected flight segment. Note that
        this parameter does not have to have the same shape as :param:`time`.
    met : MetDataset
        Meteorology used to generate the sequence of atmospheric profiles. Must contain:
        - air temperature [:math:`K`]
        - specific humidity [:math:`kg/kg`]
        - geopotential height [:math:`m`]
        - eastward wind [:math:`m/s`]
        - northward wind [:math:`m/s`]
        - vertical velocity [:math:`Pa/s`]
    humidity_scaling : HumidityScaling
        Humidity scaling applied to specific humidity in :param:`met` before
        generating atmospheric profiles.
    dz_m : float
        Altitude difference [:math:`m`] used to approximate vertical derivatives
        when computing wind shear.

    Returns
    -------
    xr.Dataset
        Meteorology dataset in required format for APCEMM input.
    """

    # Ensure that altitudes are sorted ascending
    altitude = np.sort(altitude)

    # Check for required fields in met
    vars = (
        met_var.AirTemperature,
        met_var.SpecificHumidity,
        met_var.GeopotentialHeight,
        met_var.EastwardWind,
        met_var.NorthwardWind,
        met_var.VerticalVelocity,
    )
    met.ensure_vars(vars)
    met.standardize_variables(vars)

    # Flatten input arrays
    time = time.ravel()
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    azimuth = azimuth.ravel()
    altitude = altitude.ravel()

    # Estimate pressure levels close to target altitudes
    # (not exact because this assumes the ISA temperature profile)
    pressure = units.m_to_pl(altitude) * 1e2

    # Broadcast to required shape and create vector for initial interpolation
    # onto original pressure levels at target horizontal location.
    shape = (time.size, altitude.size)
    time = np.broadcast_to(time[:, np.newaxis], shape).ravel()
    longitude = np.broadcast_to(longitude[:, np.newaxis], shape).ravel()
    latitude = np.broadcast_to(latitude[:, np.newaxis], shape).ravel()
    azimuth = np.broadcast_to(azimuth[:, np.newaxis], shape).ravel()
    level = np.broadcast_to(pressure[np.newaxis, :] / 1e2, shape).ravel()
    vector = GeoVectorDataset(
        data={"azimuth": azimuth},
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
    )

    # Downselect met before interpolation
    met = vector.downselect_met(met)

    # Interpolate meteorology data onto vector
    scale_humidity = humidity_scaling is not None and "specific_humidity" not in vector
    for met_key in (
        "air_temperature",
        "eastward_wind",
        "geopotential_height",
        "northward_wind",
        "specific_humidity",
        "lagrangian_tendency_of_air_pressure",
    ):
        models.interpolate_met(met, vector, met_key, **interp_kwargs)

    # Interpolate winds at lower level for shear calculation
    air_pressure_lower = thermo.pressure_dz(vector["air_temperature"], vector.air_pressure, dz_m)
    lower_level = air_pressure_lower / 100.0
    for met_key in ("eastward_wind", "northward_wind"):
        vector_key = f"{met_key}_lower"
        models.interpolate_met(met, vector, met_key, vector_key, **interp_kwargs, level=lower_level)

    # Apply humidity scaling
    if scale_humidity and humidity_scaling is not None:
        humidity_scaling.eval(vector, copy_source=False)

    # Compute RHi and segment-normal shear
    vector.setdefault(
        "rhi",
        thermo.rhi(vector["specific_humidity"], vector["air_temperature"], vector.air_pressure),
    )
    vector.setdefault(
        "normal_shear",
        normal_wind_shear(
            vector["eastward_wind"],
            vector["eastward_wind_lower"],
            vector["northward_wind"],
            vector["northward_wind_lower"],
            vector["azimuth"],
            dz_m,
        ),
    )

    # Reshape interpolated fields to (time, level).
    nlev = altitude.size
    ntime = len(vector) // nlev
    shape = (ntime, nlev)
    time = np.unique(vector["time"])
    time = (time - time[0]) / np.timedelta64(1, "h")
    temperature = vector["air_temperature"].reshape(shape)
    qv = vector["specific_humidity"].reshape(shape)
    z = vector["geopotential_height"].reshape(shape)
    rhi = vector["rhi"].reshape(shape)
    shear = vector["normal_shear"].reshape(shape)
    shear[:, -1] = shear[:, -2]  # lowest level will be nan
    omega = vector["lagrangian_tendency_of_air_pressure"].reshape(shape)
    virtual_temperature = temperature * (1 + qv / constants.epsilon) / (1 + qv)
    density = pressure[np.newaxis, :] / (constants.R_d * virtual_temperature)
    w = -omega / (density * constants.g)

    # Interpolate fields to target altitudes profile-by-profile
    # to obtain 2D arrays with dimensions (time, altitude).
    temperature_on_z = np.zeros(shape, dtype=temperature.dtype)
    rhi_on_z = np.zeros(shape, dtype=rhi.dtype)
    shear_on_z = np.zeros(shape, dtype=shear.dtype)
    w_on_z = np.zeros(shape, dtype=w.dtype)

    # Fields should already be on pressure levels close to target
    # altitudes, so this just uses linear interpolation and constant
    # extrapolation on fields expected by APCEMM.
    # NaNs are preserved at the start and end of interpolated profiles
    # but removed in interiors.
    def interp(z: np.ndarray, z0: np.ndarray, f0: np.ndarray) -> np.ndarray:
        # mask nans
        mask = np.isnan(z0) | np.isnan(f0)
        if np.all(mask):
            msg = (
                "Found all-NaN profile during APCEMM meterology input file creation. "
                "MetDataset may have insufficient spatiotemporal coverage."
            )
            raise ValueError(msg)
        z0 = z0[~mask]
        f0 = f0[~mask]

        # interpolate
        assert np.all(np.diff(z0) > 0)  # expect increasing altitudes
        fi = np.interp(z, z0, f0, left=f0[0], right=f0[-1])

        # restore nans at start and end of profile
        if mask[0]:  # nans at top of profile
            fi[z > z0.max()] = np.nan
        if mask[-1]:  # nans at end of profile
            fi[z < z0.min()] = np.nan
        return fi

    # The manual for loop is unlikely to be a bottleneck since a
    # substantial amount of work is done within each iteration.
    for i in range(ntime):
        temperature_on_z[i, :] = interp(altitude, z[i, :], temperature[i, :])
        rhi_on_z[i, :] = interp(altitude, z[i, :], rhi[i, :])
        shear_on_z[i, :] = interp(altitude, z[i, :], shear[i, :])
        w_on_z[i, :] = interp(altitude, z[i, :], w[i, :])

    # APCEMM also requires initial pressure profile
    pressure_on_z = interp(altitude, z[0, :], pressure)

    # Create APCEMM input dataset.
    # Transpose require because APCEMM expects (altitude, time) arrays.
    return xr.Dataset(
        data_vars={
            "pressure": (("altitude",), pressure_on_z.astype("float32") / 1e2, {"units": "hPa"}),
            "temperature": (
                ("altitude", "time"),
                temperature_on_z.astype("float32").T,
                {"units": "K"},
            ),
            "relative_humidity_ice": (
                ("altitude", "time"),
                1e2 * rhi_on_z.astype("float32").T,
                {"units": "percent"},
            ),
            "shear": (("altitude", "time"), shear_on_z.astype("float32").T, {"units": "s**-1"}),
            "w": (("altitude", "time"), w_on_z.astype("float32").T, {"units": "m s**-1"}),
        },
        coords={
            "altitude": ("altitude", altitude.astype("float32") / 1e3, {"units": "km"}),
            "time": ("time", time, {"units": "hours"}),
        },
    )

def fix_dataset(ds):

    indices = np.where(np.isfinite(ds["temperature"]))
    interp = NearestNDInterpolator(np.transpose(indices), ds["temperature"].values[indices])
    ds["temperature"][...] = interp(*np.indices(ds["temperature"].shape)) 

    ds["specific_cloud_liquid_water_content"] = ds["specific_cloud_liquid_water_content"].fillna(0)
    ds["specific_cloud_ice_water_content"] = ds["specific_cloud_ice_water_content"].fillna(0)
    ds["specific_humidity"] = ds["specific_humidity"].fillna(0)

    ds["pressure"][0] = 1000
    ds["pressure"][-1] = 1

    ds["fraction_of_cloud_cover"] = ds["fraction_of_cloud_cover"].fillna(0)

    ds["moist_density"] = (ds["pressure"]*1e2/286.9)/ds["temperature"] #* (1 + ds["specific_humidity"])) / (1 + 1.609 * ds["specific_humidity"])

    ds["cloud_LWC"] = ds["specific_cloud_liquid_water_content"] * ds["moist_density"] * 1e3
    ds["cloud_IWC"] = ds["specific_cloud_ice_water_content"] * ds["moist_density"] * 1e3

    return ds

def advect(met, met_temp, fl):

    dt_input_met = np.timedelta64(6, "m")
    dt_integration = np.timedelta64(2, 'm')
    max_age = np.timedelta64(24, 'h')

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 1.0,  # initial plume depth, [m]
        "width": 1.0,  # initial plume width, [m]
    }

    dry_adv = DryAdvection(met, params)

    dry_adv_df = dry_adv.eval(fl).dataframe

    # We re-set the max-age parameter to the maximum age that the DryAdvection model has calculated.
    # This ensures that for DryAdvection models that terminate early, whilst generating the APCEMM
    # input files we do not get NaN errors (that are found in the final entry of dry_adv_df) when
    # the DryAdvection model terminates early. 
    try:
        max_age = list(dry_adv_df["time"].values)[-2] - dry_adv_df["time"].min()
    except IndexError:
        return 1, 1, 1, 1, True
        

    air_pressure = dry_adv_df["air_pressure"].values
    lon = dry_adv_df["longitude"].values
    lat = dry_adv_df["latitude"].values
    azimuth = dry_adv_df["azimuth"].values
    time = dry_adv_df["time"].values

    n_profiles = int(max_age / dt_input_met) + 1
    tick = np.timedelta64(1, "s")
    target_elapsed = np.linspace(
        0, (n_profiles - 1) * dt_input_met / tick, n_profiles
    )
    target_time = time[0] + target_elapsed * tick
    elapsed = (dry_adv_df["time"] - dry_adv_df["time"][0]) / tick

    min_pos = np.min(lon[lon>0], initial = np.inf)
    max_neg = np.max(lon[lon<0], initial=-np.inf)
    if (180 - min_pos) + (180 + max_neg) < 180 and min_pos < np.inf and max_neg > -np.inf:
        lon = np.where(lon > 0, lon - 360, lon)
    interp_lon = np.interp(target_elapsed, elapsed, lon)
    interp_lon = np.where(interp_lon > 180, interp_lon - 360, interp_lon)

    interp_lat = np.interp(target_elapsed, elapsed, lat)
    interp_az = np.interp(target_elapsed, elapsed, azimuth)

    altitude = met["altitude"].values
    altitude_temp = met_temp["altitude"].values

    ds = generate_apcemm_input_met(
        time=target_time,
        longitude=interp_lon,
        latitude=interp_lat,
        azimuth=interp_az,
        altitude=altitude,
        met=met,
        humidity_scaling=None,
        dz_m=200,
        interp_kwargs={'method':'linear'})
    
    ds_temp = generate_temp_profile(
        time=target_time,
        longitude=interp_lon,
        latitude=interp_lat,
        azimuth=interp_az,
        altitude=altitude_temp,
        met=met_temp,
        humidity_scaling=None,
        dz_m=200,
        interp_kwargs={'method':'linear'})
    
    ds_temp = fix_dataset(ds_temp)

    temperature = ds["temperature"].sel(time=0,altitude=10.9,method='nearest').values

    return ds, ds_temp, air_pressure[0], temperature, False


"""if __name__ == '__main__':

    met = get_albedo('gribs/download.grib')

    longitude = 55.2
    latitude = 12.2 

    print(met['fal'].data.sel(longitude=longitude, latitude=latitude, time ='2024-03-01T13:00:00.000000000', level=-1, method='nearest').values)"""
