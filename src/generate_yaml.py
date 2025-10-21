import pandas as pd
from pycontrails import Flight

def generate_yaml_d(identifier, sample, fl, pressure, properties_dict, met_file_path, path):

    longitude = float(sample["longitude"])
    latitude = float(sample["latitude"])

    emission_day = int(sample["time"].day)
    emission_time = int(sample["time"].hour)

    output_folder = path
    input_background_condition = '../../../../../../input_data/init.txt'
    intput_engine_emissions = '../../../../../../input_data/ENG_EI.txt'
    force_seed_value = 'F'
    seed_value = 0

    plume_time = 24         # [h]
    temperature = 217       # [K], can be overwritten by met
    RHw = 63.94             # [%], can be overwritten by met

    horiz_diff_coeff = 15.0 # [m^2/s]
    verti_diff_coeff = 0.15 # [m^2/s], can be overwritten by met
    wind_shear = 0.002      # [1/s]
    brunt_vaisala = 0.013   # [1/s]

    NOx = 5100      # [ppt]
    HNO3 = 81.5     # [ppt]
    O3 = 100        # [ppb]
    CO = 40         # [ppb]
    CH4 = 1.76      # [ppm]
    SO2 = 7.25      # [ppt]

    transport_time_step = 6    # [min]

    ice_growth_time_step = 6   # [min]

    input_timestep = 0.1 # hours

    # Depending on aircraft type, fuel type and engine type, 
    # calculate the emission indices, fuel flow, aircraft mass,
    # number of engines, wingspan, core exit temp, and exit bypass area.

    NOx_fuel = 1.2
    CO_fuel = 0.8
    UHC = 0.6

    SO2_fuel = 0.00001      # [g/kg fuel]
    SO2_to_SO4 = 4      # [%]
    soot = float(properties_dict["nvPM_EI_m"])        # [g/kg fuel]
    soot_radius = 20e-9                        # [m], does not depend on aircraft, fuel or engine type
    
    total_fuel_flow = float(properties_dict["fuel_flow_rate"]) # [kg/s]
    aircraft_mass   = float(properties_dict["mass"])           # [kg], at beginning of cruise
    flight_speed    = float(properties_dict["air_speed"])      # [m/s] 
    wingspan        = float(properties_dict["wingspan"])       # [m]

    num_engines     = 2         # 2 or 4
    core_exit_temp = 553.65     # [K]
    exit_bypass_area = 0.9772   # [m^2]

    humidity_mod_scheme = 'none' # none / constant / scaling
    humidity_scaling_constant_a = 0.9779
    humidity_scaling_constant_b = 1.635
    
    d = {
    'SIMULATION MENU': {'OpenMP Num Threads (positive int)' : 8,
                        'PARAM SWEEP SUBMENU':  {'Parameter sweep (T/F)' : 'T',
                                                'Run Monte Carlo (T/F)' : 'F',
                                                'Num Monte Carlo runs (int)' : 2},
                        'OUTPUT SUBMENU':   {'Output folder (string)' : output_folder,
                                             'Overwrite if folder exists (T/F)' : 'T'},
                        'Use threaded FFT (T/F)' : 'F',
                        'FFTW WISDOM SUBMENU':  {'Use FFTW WISDOM (T/F)' : 'T',
                                                 'Dir w/ write permission (string)': './'},
                        'Input background condition (string)'   : input_background_condition,
                        'Input engine emissions (string)'       : intput_engine_emissions,
                        'SAVE FORWARD RESULTS SUBMENU': {'Save forward results (T/F)' : 'F',
                                                        'netCDF filename format (string)' : 'APCEMM_Case_*'},
                        'ADJOINT OPTIMIZATION SUBMENU': {'Turn on adjoint optim. (T/F)': 'F',
                                                        'netCDF filename format (string)': 'APCEMM_ADJ_Case_*'},
                        'BOX MODEL SUBMENU':    {'Run box model (T/F)': 'F',
                                                 'netCDF filename format (string)': 'APCEMM_BOX_Case_*'},
                        'RANDOM NUMBER GENERATION SUBMENU': {'Force seed value (T/F)': force_seed_value,
                                                             'Seed value (positive int)': seed_value}},
    'PARAMETER MENU':   {'Plume Process [hr] (double)': plume_time,
                        'METEOROLOGICAL PARAMETERS SUBMENU': {'Temperature [K] (double)': temperature,
                                                            'R.Hum. wrt water [%] (double)': RHw,
                                                            'Pressure [hPa] (double)': pressure,
                                                            'Horiz. diff. coeff. [m^2/s] (double)': horiz_diff_coeff,
                                                            'Verti. diff. [m^2/s] (double)': verti_diff_coeff,
                                                            'Wind shear [1/s] (double)': wind_shear,
                                                            'Brunt-Vaisala Frequency [s^-1] (double)': brunt_vaisala},
                        'LOCATION AND TIME SUBMENU':    {'LON [deg] (double)': longitude,
                                                         'LAT [deg] (double)': latitude,
                                                         'Emission day [1-365] (int)': emission_day,
                                                         'Emission time [hr] (double)': emission_time},
                        'BACKGROUND MIXING RATIOS SUBMENU': {'NOx [ppt] (double)': NOx,
                                                             'HNO3 [ppt] (double)': HNO3,
                                                             'O3 [ppb] (double)': O3,
                                                             'CO [ppb] (double)': CO,
                                                             'CH4 [ppm] (double)': CH4,
                                                             'SO2 [ppt] (double)': SO2},
                        'EMISSION INDICES SUBMENU': {'NOx [g(NO2)/kg_fuel] (double)': NOx_fuel,
                                                     'CO [g/kg_fuel] (double)': CO_fuel,
                                                     'UHC [g/kg_fuel] (double)': UHC,
                                                     'SO2 [g/kg_fuel] (double)': SO2_fuel,
                                                     'SO2 to SO4 conv [%] (double)': SO2_to_SO4,
                                                     'Soot [g/kg_fuel] (double)': soot},
                        'Soot Radius [m] (double)': soot_radius,
                        'Total fuel flow [kg/s] (double)': total_fuel_flow,
                        'Aircraft mass [kg] (double)': aircraft_mass,
                        'Flight speed [m/s] (double)': flight_speed,
                        'Num. of engines [2/4] (int)': num_engines,
                        'Wingspan [m] (double)': wingspan,
                        'Core exit temp. [K] (double)': core_exit_temp,
                        'Exit bypass area [m^2] (double)': exit_bypass_area},
    'TRANSPORT MENU':  {'Turn on Transport (T/F)': 'T',
                        'Fill Negative Values (T/F)': 'T',
                        'Transport Timestep [min] (double)': transport_time_step,
                        'PLUME UPDRAFT SUBMENU' : {'Turn on plume updraft (T/F)': 'F',
                                                   'Updraft timescale [s] (double)': 3600,
                                                   'Updraft veloc. [cm/s] (double)': 5}},
    'CHEMISTRY MENU':  {'Turn on Chemistry (T/F)': 'F',
                        'Perform hetero. chem. (T/F)': 'F',
                        'Chemistry Timestep [min] (double)' : 6,
                        'Photolysis rates folder (string)': '/path/to/input'},
    'AEROSOL MENU':    {'Turn on grav. settling (T/F)': 'T',
                        'Turn on solid coagulation (T/F)': 'T',
                        'Turn on liquid coagulation (T/F)': 'F',
                        'Coag. timestep [min] (double)': 60,
                        'Turn on ice growth (T/F)': 'T',
                        'Ice growth timestep [min] (double)': ice_growth_time_step},
    'METEOROLOGY MENU':  {'METEOROLOGICAL INPUT SUBMENU' : {'Use met. input (T/F)': 'T',
                                                          'Met input file path (string)': met_file_path,
                                                          'Time series data timestep [hr] (double)': input_timestep,
                                                          'Init temp. from met. (T/F)': 'T',
                                                          'Temp. time series input (T/F)': 'T',
                                                          'Interpolate temp. met. data (T/F)': 'T',
                                                          'Init RH from met. (T/F)': 'T',
                                                          'RH time series input (T/F)': 'T',
                                                          'Interpolate RH met. data (T/F)': 'T',
                                                          'Init wind shear from met. (T/F)': 'T',
                                                          'Wind shear time series input (T/F)': 'T',
                                                          'Interpolate shear met. data (T/F)': 'F',
                                                          'Init vert. veloc. from met. data (T/F)': 'T',
                                                          'Vert. veloc. time series input (T/F)': 'F',
                                                          'Interpolate vert. veloc. met. data (T/F)': 'F',
                                                          'HUMIDITY SCALING OPTIONS' : {'Humidity modification scheme (none / constant / scaling)': humidity_mod_scheme,
                                                                                        'Constant RHi [%] (double)': 110,
                                                                                        'Humidity scaling constant a (double)': humidity_scaling_constant_a,
                                                                                        'Humidity scaling constant b (double)': humidity_scaling_constant_b}},
                        'IMPOSE MOIST LAYER DEPTH SUBMENU': {'Impose moist layer depth (T/F)': 'F',
                                                             'Moist layer depth [m] (double)': 1000,
                                                             'Subsaturated air RHi [%] (double)': 80},
                        'IMPOSE LAPSE RATE SUBMENU':   {'Impose lapse rate (T/F)': 'F',
                                                        'Lapse rate [K/m] (T/F)': -6.0e-3},
                        'Add diurnal variations (T/F)': 'F',
                        'TEMPERATURE PERTURBATION SUBMENU': {'Enable Temp. Pert. (T/F)': 'F',
                                                             'Temp. Perturb. Amplitude (double)': 1.0,
                                                             'Temp. Perturb. Timescale (min)': 6}},
    'DIAGNOSTIC MENU': {'netCDF filename format (string)': 'trac_avg.apcemm.hhmm',
                        'SPECIES TIMESERIES SUBMENU' : {'Save species timeseries (T/F)': 'F',
                                                        'Inst timeseries file (string)': 'ts_hhmm.nc',
                                                        'Species indices to include (list of ints)': 1,
                                                        'Save frequency [min] (double)': 6},
                        'AEROSOL TIMESERIES SUBMENU': {'Save aerosol timeseries (T/F)': 'T',
                                                       'Inst timeseries file (string)': 'ts_aerosol_hhmm.nc',
                                                       'Aerosol indices to include (list of ints)': 1,
                                                       'Save frequency [min] (double)': 6},
                        'PRODUCTION & LOSS SUBMENU': {'Turn on P/L diag (T/F)': 'F',
                                                      'Save O3 P/L (T/F)': 'F'}},
    'ADVANCED OPTIONS MENU': {'GRID SUBMENU': {'NX (positive int)' : 200,
                                               'NY (positive int)' : 180,
                                               'XLIM_RIGHT (positive double)': 1.0e+3,
                                               'XLIM_LEFT (positive double)': 1.0e+3,
                                               'YLIM_UP (positive double)': 300,
                                               'YLIM_DOWN (positive double)': 1.5e+3},
                             'INITIAL CONTRAIL SIZE SUBMENU': {'Base Contrail Depth [m] (double)': 0.0,
                                                               'Contrail Depth Scaling Factor [-] (double)': 1.0,
                                                               'Base Contrail Width [m] (double)': 0.0,
                                                               'Contrail Width Scaling Factor [-] (double)': 1.0,},
                             'Ambient Lapse Rate [K/km] (double)': -3.0,
                             'Tropopause Pressure [Pa] (double)': 2.0e+4}
    }                                  

    return d
