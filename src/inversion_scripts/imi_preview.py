#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import warnings
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import colorcet as cc
import cartopy.crs as ccrs

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.inversion_scripts.point_sources import get_point_source_coordinates
from src.inversion_scripts.utils import (
    sum_total_emissions,
    count_obs_in_mask,
    plot_field,
    read_and_filter_satellite,
    calculate_area_in_km,
    calculate_superobservation_error,
    species_molar_mass,
    mixing_ratio_conv_factor,
)
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)

def get_satellite_data(
    file_path, satellite_str, species, xlim, ylim, startdate_np64, enddate_np64
):
    """
    Returns a dict with the lat, lon, xspecies, and albedo_swir observations
    extracted from the given satellite file. Filters are applied to remove
    unsuitable observations
    Args:
        file_path : string
            path to the satellite file
        satellite_product : str
            name of satellite product
        xlim: list
            longitudinal bounds for region of interest
        ylim: list
            latitudinal bounds for region of interest
        startdate_np64: datetime64
            start date for time period of interest
        enddate_np64: datetime64
            end date for time period of interest
    Returns:
         satellite_data: dict
            dictionary of the extracted values
    """
    # satellite data dictionary
    satellite_data = {"lat": [], "lon": [], species: [], "swir_albedo": []}

    # Load the satellite data
    satellite, sat_ind = read_and_filter_satellite(
        file_path, satellite_str, startdate_np64, enddate_np64, xlim, ylim)

    # Loop over observations and archive
    num_obs = len(sat_ind[0])
    for k in range(num_obs):
        lat_idx = sat_ind[0][k]
        lon_idx = sat_ind[1][k]
        satellite_data["lat"].append(satellite["latitude"][lat_idx, lon_idx])
        satellite_data["lon"].append(satellite["longitude"][lat_idx, lon_idx])
        satellite_data[species].append(satellite[species][lat_idx, lon_idx])
        satellite_data["swir_albedo"].append(satellite["swir_albedo"][lat_idx, lon_idx])

    return satellite_data


def imi_preview(
    inversion_path, config_path, state_vector_path, preview_dir, species, satellite_cache
):
    """
    Function to perform preview
    Requires preview simulation to have been run already (to generate HEMCO diags)
    Requires satellite data to have been downloaded already
    """

    # ----------------------------------
    # Setup
    # ----------------------------------

    # Read config file
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    # redirect output to log file
    output_file = open(f"{inversion_path}/imi_output.log", "a")
    sys.stdout = output_file
    sys.stderr = output_file

    # Open the state vector file
    state_vector = xr.load_dataset(state_vector_path)
    state_vector_labels = state_vector["StateVector"]

    # Identify the last element of the region of interest
    last_ROI_element = int(
        np.nanmax(state_vector_labels.values) - config["nBufferClusters"]
    )

    # # Define mask for ROI, to be used below
    a, df, num_days, prior, outstrings = estimate_averaging_kernel(
        config, state_vector_path, preview_dir, satellite_cache, preview=True, kf_index=None
    )
    mask = state_vector_labels <= last_ROI_element

    # Count the number of observations in the region of interest
    num_obs = count_obs_in_mask(mask, df)
    if num_obs < 1:
        sys.exit("Error: No observations found in region of interest")
    outstring2 = f"Found {num_obs} observations in the region of interest"
    print("\n" + outstring2)

    # ----------------------------------
    # Estimate dollar cost
    # ----------------------------------

    # Estimate cost by scaling reference cost of $20 for one-month Permian inversion
    # Reference number of state variables = 243
    # Reference number of days = 31
    # Reference cost for EC2 storage = $50 per month
    # Reference area = area of 24-39 N 95-111W
    reference_cost = 20
    reference_num_compute_hours = 10
    reference_area_km = calculate_area_in_km(
        [(-111, 24), (-95, 24), (-95, 39), (-111, 39)]
    )
    hours_in_month = 31 * 24
    reference_storage_cost = 50 * reference_num_compute_hours / hours_in_month
    num_state_variables = np.nanmax(state_vector_labels.values)

    lats = [float(state_vector.lat.min()), float(state_vector.lat.max())]
    lons = [float(state_vector.lon.min()), float(state_vector.lon.max())]
    coords = [
        (lons[0], lats[0]),
        (lons[1], lats[0]),
        (lons[1], lats[1]),
        (lons[0], lats[1]),
    ]
    inversion_area_km = calculate_area_in_km(coords)

    if config["Res"] == "0.25x0.3125":
        res_factor = 1
    elif config["Res"] == "0.5x0.625":
        res_factor = 0.5
    elif config["Res"] == "2.0x2.5":
        res_factor = 0.125
    elif config["Res"] == "4.0x5.0":
        res_factor = 0.0625
    additional_storage_cost = ((num_days / 31) - 1) * reference_storage_cost
    expected_cost = (
        (reference_cost + additional_storage_cost)
        * (num_state_variables / 243)
        * (inversion_area_km / reference_area_km)
        * (num_days / 31)
        * res_factor
    )

    outstring6 = (
        f"approximate cost = ${np.round(expected_cost,2)} for on-demand instance"
    )
    outstring7 = f"                 = ${np.round(expected_cost/3,2)} for spot instance"
    print(outstring6)
    print(outstring7)

    # ----------------------------------
    # Output
    # ----------------------------------

    # Write preview diagnostics to text file
    outputtextfile = open(os.path.join(preview_dir, "preview_diagnostics.txt"), "w+")
    outputtextfile.write("##" + outstring2 + "\n")
    outputtextfile.write("##" + outstring6 + "\n")
    outputtextfile.write("##" + outstring7 + "\n")
    outputtextfile.write(outstrings)
    outputtextfile.close()

    # Prepare plot data for prior
    prior_kgkm2h = prior * (1000**2) * 60 * 60  # Units kg/km2/h

    # Prepare plot data for observations
    df_means = df.copy(deep=True)
    df_means["lat"] = np.round(df_means["lat"], 1)  # Bin to 0.1x0.1 degrees
    df_means["lon"] = np.round(df_means["lon"], 1)
    df_means = df_means.groupby(["lat", "lon"]).mean()
    ds = df_means.to_xarray()

    # Prepare plot data for observation counts
    df_counts = df.copy(deep=True).drop([species, "swir_albedo"], axis=1)
    df_counts["counts"] = 1
    df_counts["lat"] = np.round(df_counts["lat"], 1)  # Bin to 0.1x0.1 degrees
    df_counts["lon"] = np.round(df_counts["lon"], 1)
    df_counts = df_counts.groupby(["lat", "lon"]).sum()
    ds_counts = df_counts.to_xarray()

    plt.rcParams.update({"font.size": 18})

    # Plot prior emissions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        prior_kgkm2h,
        cmap=cc.cm.linear_kryw_5_100_c67_r,
        plot_type="pcolormesh",
        vmin=0,
        vmax=14,
        lon_bounds=None,
        lat_bounds=None,
        levels=21,
        title="Prior emissions",
        point_sources=get_point_source_coordinates(config),
        cbar_label="Emissions (kg km$^{-2}$ h$^{-1}$)",
        mask=mask,
        only_ROI=False,
    )
    plt.savefig(
        os.path.join(preview_dir, "preview_prior_emissions.png"),
        bbox_inches="tight",
        dpi=150,
    )

    # Plot observations
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        ds[species],
        cmap="Spectral_r",
        plot_type="pcolormesh",
        vmin=1800,
        vmax=1850,
        lon_bounds=None,
        lat_bounds=None,
        title=f"Satellite $X_{species}$",
        cbar_label="Column mixing ratio (ppb)",
        mask=mask,
        only_ROI=False,
    )

    plt.savefig(
        os.path.join(preview_dir, "preview_observations.png"),
        bbox_inches="tight",
        dpi=150,
    )

    # Plot albedo
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        ds["swir_albedo"],
        cmap="magma",
        plot_type="pcolormesh",
        vmin=0,
        vmax=0.4,
        lon_bounds=None,
        lat_bounds=None,
        title="SWIR Albedo",
        cbar_label="Albedo",
        mask=mask,
        only_ROI=False,
    )
    plt.savefig(
        os.path.join(preview_dir, "preview_albedo.png"), bbox_inches="tight", dpi=150
    )

    # Plot observation density
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        ds_counts["counts"],
        cmap="Blues",
        plot_type="pcolormesh",
        vmin=0,
        vmax=np.nanmax(ds_counts["counts"].values),
        lon_bounds=None,
        lat_bounds=None,
        title="Observation density",
        cbar_label="Number of observations",
        mask=mask,
        only_ROI=False,
    )
    plt.savefig(
        os.path.join(preview_dir, "preview_observation_density.png"),
        bbox_inches="tight",
        dpi=150,
    )

    sensitivities_da = map_sensitivities_to_sv(a, state_vector, last_ROI_element)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        sensitivities_da["Sensitivities"],
        cmap=cc.cm.CET_L19,
        lon_bounds=None,
        lat_bounds=None,
        title="Estimated Averaging kernel sensitivities",
        cbar_label="Sensitivity",
        only_ROI=True,
        state_vector_labels=state_vector_labels,
        last_ROI_element=last_ROI_element,
    )
    plt.savefig(
        os.path.join(preview_dir, "preview_estimated_sensitivities.png"),
        bbox_inches="tight",
        dpi=150,
    )
    expectedDOFS = np.round(sum(a), 5)
    if expectedDOFS < config["DOFSThreshold"]:
        print(
            f"\nExpected DOFS = {expectedDOFS} are less than DOFSThreshold = {config['DOFSThreshold']}. Exiting.\n"
        )
        print(
            "Consider increasing the inversion period, increasing the prior error, or using another prior inventory.\n"
        )
        # if run with sbatch this ensures the exit code is not lost.
        file = open(".error_status_file.txt", "w")
        file.write("Error Status: 1")
        file.close()
        sys.exit(1)


def map_sensitivities_to_sv(sensitivities, sv, last_ROI_element):
    """
    maps sensitivities onto 2D xarray Datarray for visualization
    """
    s = sv.copy().rename({"StateVector": "Sensitivities"})
    mask = s["Sensitivities"] <= last_ROI_element
    s["Sensitivities"] = s["Sensitivities"].where(mask)
    # map sensitivities onto corresponding xarray DataArray
    for i in range(1, last_ROI_element + 1):
        mask = sv["StateVector"] == i
        s = xr.where(mask, sensitivities[i - 1], s)

    return s


def estimate_averaging_kernel(
    config, species, state_vector_path, preview_dir, satellite_cache, preview=False, kf_index=None
):
    """
    Estimates the averaging kernel sensitivities using prior emissions
    and the number of observations available in each grid cell
    """

    # ----------------------------------
    # Setup
    # ----------------------------------

    # Open the state vector file
    state_vector = xr.load_dataset(state_vector_path)
    state_vector_labels = state_vector["StateVector"]

    # Identify the last element of the region of interest
    last_ROI_element = int(
        np.nanmax(state_vector_labels.values) - config["nBufferClusters"]
    )

    # Define mask for ROI, to be used below
    mask = state_vector_labels <= last_ROI_element

    # ----------------------------------
    # Total prior emissions
    # ----------------------------------

    # Prior emissions
    preview_cache = os.path.join(preview_dir, "OutputDir")
    hemco_diags_file = [
        f for f in os.listdir(preview_cache) if "HEMCO_diagnostics" in f
    ][0]
    prior_pth = os.path.join(preview_cache, hemco_diags_file)
    prior = xr.load_dataset(prior_pth)[f"Emis{species}_Total"].isel(time=0)
    
    # Start and end dates of the inversion
    startday = str(config["StartDate"])
    endday = str(config["EndDate"])

    # adjustments for when performing for dynamic kf clustering
    if kf_index is not None:
        # use different date range for KF inversion if kf_index is not None
        rundir_path = preview_dir.split('preview_run')[0]
        periods = pd.read_csv(f"{rundir_path}periods.csv")
        startday = str(periods.iloc[kf_index - 1]["Starts"])
        endday = str(periods.iloc[kf_index - 1]["Ends"])
        
        # use the nudged (prior) emissions for generating averaging kernel estimate
        sf = xr.load_dataset(f"{rundir_path}archive_sf/prior_sf_period{kf_index}.nc")
        prior = sf["ScaleFactor"] * prior
        

    # Compute total emissions in the region of interest
    areas = xr.load_dataset(prior_pth)["AREA"]
    total_prior_emissions = sum_total_emissions(prior, areas, mask)
    outstring1 = (
        f"Total prior emissions in region of interest = {total_prior_emissions} Tg/y \n"
    )
    print(outstring1)

    # ----------------------------------
    # Observations in region of interest
    # ----------------------------------

    # Paths to satellite data files
    satellite_files = [f for f in os.listdir(satellite_cache) if ".nc" in f]
    satellite_paths = [os.path.join(satellite_cache, f) for f in satellite_files]

    # Latitude/longitude bounds of the inversion domain
    xlim = [float(state_vector.lon.min()), float(state_vector.lon.max())]
    ylim = [float(state_vector.lat.min()), float(state_vector.lat.max())]

    start = f"{startday[0:4]}-{startday[4:6]}-{startday[6:8]} 00:00:00"
    end = f"{endday[0:4]}-{endday[4:6]}-{endday[6:8]} 23:59:59"
    startdate_np64 = np.datetime64(
        datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    )
    enddate_np64 = np.datetime64(
        datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        - datetime.timedelta(days=1)
    )

    # Only consider satellite files within date range (in case more are present)
    satellite_paths = [
        p
        for p in satellite_paths
        if int(p.split("____")[1][0:8]) >= int(startday)
        and int(p.split("____")[1][0:8]) < int(endday)
    ]
    satellite_paths.sort()

    # What satellite data product to use?
    satellite_str = config["SatelliteProduct"]

    # Open satellite files and filter data
    lat = []
    lon = []
    xspecies = []
    albedo = []

    # Read in and filter satellite observations (uses parallel processing)
    observation_dicts = Parallel(n_jobs=-1)(
        delayed(get_satellite_data)(
            file_path, satellite_str, species, xlim, ylim, startdate_np64, enddate_np64
        )
        for file_path in satellite_paths
    )
    # Remove any problematic observation dicts (eg. corrupted data file)
    observation_dicts = list(filter(None, observation_dicts))

    for dict in observation_dicts:
        lat.extend(dict["lat"])
        lon.extend(dict["lon"])
        xspecies.extend(dict[species])
        albedo.extend(dict["swir_albedo"])

    # Assemble in dataframe
    df = pd.DataFrame()
    df["lat"] = lat
    df["lon"] = lon
    df["count"] = np.ones(len(lat))
    df["swir_albedo"] = albedo
    df[species] = xspecies

    # Set resolution specific variables
    # L_native = Rough length scale of native state vector element [m]
    if config["Res"] == "0.25x0.3125":
        L_native = 25 * 1000
        lat_step = 0.25
        lon_step = 0.3125
    elif config["Res"] == "0.5x0.625":
        L_native = 50 * 1000
        lat_step = 0.5
        lon_step = 0.625
    elif config["Res"] == "2.0x2.5":
        L_native = 200 * 1000
        lat_step = 2.0
        lon_step = 2.5
    elif config["Res"] == "4.0x5.0":
        L_native = 400 * 1000
        lat_step = 4.0
        lon_step = 5.0

    # bin observations into gridcells and map onto statevector
    observation_counts = add_observation_counts(df, state_vector, lat_step, lon_step)

    # parallel processing function
    def process(i):
        mask = state_vector_labels == i
        # prior emissions for each element (in Tg/y)
        emissions_temp = sum_total_emissions(prior, areas, mask)
        # append the calculated length scale of element
        L_temp = L_native * state_vector_labels.where(mask).count().item()
        # append the number of obs in each element
        num_obs_temp = np.nansum(observation_counts["count"].where(mask).values)
        return emissions_temp, L_temp, num_obs_temp

    # in parallel, create lists of emissions, number of observations,
    # and rough length scale for each cluster element in ROI
    result = Parallel(n_jobs=-1)(
        delayed(process)(i) for i in range(1, last_ROI_element + 1)
    )

    # unpack list of tuples into individual lists
    emissions, L, num_obs = [list(item) for item in zip(*result)]

    if np.sum(num_obs) < 1:
        sys.exit("Error: No observations found in region of interest")
    outstring2 = f"Found {np.sum(num_obs)} observations in the region of interest"

    # ----------------------------------
    # Estimate information content
    # ----------------------------------

    time_delta = enddate_np64 - startdate_np64
    num_days = np.round((time_delta) / np.timedelta64(1, "D"))

    # State vector, observations
    emissions = np.array(emissions)
    m = np.array(num_days)  # Number of observation days
    L = np.array(L)

    # If Kalman filter mode, count observations per inversion period
    if config["KalmanMode"]:
        startday_dt = datetime.datetime.strptime(startday, "%Y%m%d")
        endday_dt = datetime.datetime.strptime(endday, "%Y%m%d")
        n_periods = np.floor((endday_dt - startday_dt).days / config["UpdateFreqDays"])
        n_obs_per_period = np.round(num_obs / n_periods)
        outstring2 = f"Found {int(np.sum(n_obs_per_period))} observations in the region of interest per inversion period, for {int(n_periods)} period(s)"
        m = config["UpdateFreqDays"] # number of days in inversion period

    print("\n" + outstring2)

    # Other parameters
    U = 5 * (1000 / 3600)  # 5 km/h uniform wind speed in m/s
    p = 101325  # Surface pressure [Pa = kg/m/s2]
    g = 9.8  # Gravity [m/s2]
    Mair = 0.029  # Molar mass of air [kg/mol]
    Mspecies = species_molar_mass(species)  # Molar mass of species [kg/mol]
    alpha = 0.4  # Simple parameterization of turbulence

    # Change units of total prior emissions
    emissions_kgs = emissions * mixing_ratio_conv_factor(species) / (3600 * 24 * 365)  # kg/s from Tg/y
    emissions_kgs_per_m2 = emissions_kgs / np.power(
        L, 2
    )  # kg/m2/s from kg/s, per element

    # Error standard deviations with updated units
    sA = config["PriorError"] * emissions_kgs_per_m2
    sO = config["ObsError"]

    # Calculate superobservation error to use in averaging kernel sensitivity equation
    # from P observations per grid cell = number of observations per grid cell / m days
    P = np.array(num_obs) / num_days # number of observations per grid cell (native state vector element)
    s_superO_1 = calculate_superobservation_error(sO, 1) # for handling cells with 0 observations (avoid divide by 0)
    s_superO_p = [calculate_superobservation_error(sO, element) if element >= 1.0 else s_superO_1 
                    for element in P] # list containing superobservation error per state vector element
    s_superO = np.array(s_superO_p) / mixing_ratio_conv_factor(species) # convert to ppb

    # Averaging kernel sensitivity for each grid element
    k = alpha * (Mair * L * g / (Mspecies * U * p))
    a = sA**2 / (sA**2 + (s_superO / k) ** 2 / m) # m is number of days

    outstring3 = f"k = {np.round(k,5)} kg-1 m2 s"
    outstring4 = f"a = {np.round(a,5)} \n"
    outstring5 = f"expectedDOFS: {np.round(sum(a),5)}"

    if config["KalmanMode"]:
        outstring5 += " per inversion period"

    print(outstring3)
    print(outstring4)
    print(outstring5)

    if preview:
        outstrings = (
            f"##{outstring1}\n" + f"##{outstring3}\n" + f"##{outstring4}\n" + outstring5
        )
        return a, df, num_days, prior, outstrings
    else:
        return a


def add_observation_counts(df, state_vector, lat_step, lon_step):
    """
    Given arbitrary observation coordinates in a pandas df, group
    them by gridcell and return the number of observations mapped
    onto the statevector dataset
    """
    to_lon = lambda x: np.floor(x / lon_step) * lon_step
    to_lat = lambda x: np.floor(x / lat_step) * lat_step

    df = df.rename(columns={"lon": "old_lon", "lat": "old_lat"})

    df["lat"] = to_lat(df.old_lat)
    df["lon"] = to_lon(df.old_lon)
    groups = df.groupby(["lat", "lon"])

    counts_ds = groups.sum().to_xarray().drop_vars(["old_lat", "old_lon"])
    return xr.merge([counts_ds, state_vector])


if __name__ == "__main__":
    inversion_path = sys.argv[1]
    config_path = sys.argv[2]
    state_vector_path = sys.argv[3]
    preview_dir = sys.argv[4]
    satellite_cache = sys.argv[5]

    imi_preview(
        inversion_path, config_path, state_vector_path, preview_dir, satellite_cache
    )
