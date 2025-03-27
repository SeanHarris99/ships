from datetime import timedelta
from io import BytesIO
from zipfile import ZipFile
import random
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point
import matplotlib.pyplot as plt
from pystac_client import Client
from planetary_computer import sign
from odc.stac import load
import xarray as xr
import sqlite3
import requests


def download_ais_for_day(target_date):
    fname = f"AIS_2023_{target_date.month:02d}_{target_date.day:02d}.zip"
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/{fname}"

    r = requests.get(url, timeout=60)
    with ZipFile(BytesIO(r.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            ais = pd.read_csv(f)

    # Clean
    ais = ais[ais.TransceiverClass == 'A']
    ais = ais[(ais.SOG > 1) & (ais.SOG < 80)]
    ais = ais[(ais.Length > 30) & (ais.Length < 400)]
    ais = ais.replace({'Heading': {511: np.nan}})

    mmsi_counts = ais.MMSI.value_counts()
    active = mmsi_counts[mmsi_counts >= 5].index
    ais = ais[ais["MMSI"].isin(active)].reset_index(drop=True)

    ais["BaseDateTime"] = pd.to_datetime(ais["BaseDateTime"], utc=True)
    return ais


def project_ship_position(lat, lon, sog_knots, cog_deg, ais_time, sar_time):
    sog_mps = sog_knots * 0.51444
    delta_t = (sar_time - ais_time).total_seconds()
    if sog_knots == 0 or np.isnan(cog_deg):
        return lat, lon

    distance_m = sog_mps * delta_t
    cog_rad = np.deg2rad(cog_deg)
    dx = distance_m * np.sin(cog_rad)
    dy = distance_m * np.cos(cog_rad)

    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.deg2rad(lat))

    delta_lat = dy / meters_per_deg_lat
    delta_lon = dx / meters_per_deg_lon
    return lat + delta_lat, lon + delta_lon


def get_matching_ais_rows(ais_df, target_date):
    api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    results = api.search(
        collections=["sentinel-1-grd"],
        bbox=[-160, 4, -50, 50],
        datetime=str(target_date),
    )

    sentinel_passes = [(shape(item.geometry), item.datetime, item) for item in results.get_all_items()]
    ais_gdf = gpd.GeoDataFrame(ais_df, geometry=gpd.points_from_xy(ais_df["LON"], ais_df["LAT"]), crs="EPSG:4326")

    TIME_WINDOW = timedelta(minutes=10)
    match_list = []

    for poly, dt, item in sentinel_passes:
        time_mask = (ais_df["BaseDateTime"] >= dt - TIME_WINDOW) & (ais_df["BaseDateTime"] <= dt + TIME_WINDOW)
        candidates = ais_gdf[time_mask]
        spatial = candidates[candidates.geometry.intersects(poly)]

        if spatial.empty:
            continue

        for mmsi, group in spatial.groupby("MMSI"):
            closest = group.iloc[(group["BaseDateTime"] - dt).abs().argsort()[:1]]
            match_list.append((item, dt, closest.iloc[0]))

    return match_list


def download_and_crop_image(item, ais_row, crop_buffer=0.03):
    item = sign(item)
    arr = load(
        [item],
        bands=["vv"],
        crs="EPSG:4326",
        resolution=0.0001,
        bbox=(
            ais_row["LON"] - crop_buffer,
            ais_row["LAT"] - crop_buffer,
            ais_row["LON"] + crop_buffer,
            ais_row["LAT"] + crop_buffer,
        ),
        groupby="solar_day",
    )

    if arr.vv.size == 0:
        return None

    sar_time = pd.to_datetime(arr.vv.time.values[0]).tz_localize("UTC")
    adj_lat, adj_lon = project_ship_position(
        ais_row["LAT"], ais_row["LON"], ais_row["SOG"], ais_row["COG"], ais_row["BaseDateTime"], sar_time
    )

    lat_slice = slice(adj_lat + crop_buffer, adj_lat - crop_buffer)
    lon_slice = slice(adj_lon - crop_buffer, adj_lon + crop_buffer)

    crop = arr.vv.sel(latitude=lat_slice, longitude=lon_slice).isel(time=0)
    return crop


def save_crop(crop, ais_row, out_dir):
    if crop is None:
        return None

    mmsi = ais_row["MMSI"]
    ts = ais_row["BaseDateTime"].strftime("%Y%m%dT%H%M%S")
    fname = f"{mmsi}_{ts}.npy"
    day_dir = os.path.join(out_dir, ais_row["BaseDateTime"].date().isoformat())
    os.makedirs(day_dir, exist_ok=True)
    fpath = os.path.join(day_dir, fname)

    if os.path.exists(fpath):
        return fpath

    img = crop.values
    if img.size == 0 or np.all(np.isnan(img)):
        return None

    np.save(fpath, img)
    return fpath  



def process_one_day(target_date, sentinel1_dir, conn):
    print(f"Processing {target_date}")
    target_date = pd.to_datetime(target_date).date()
    ais = download_ais_for_day(target_date)
    matches = get_matching_ais_rows(ais, target_date)

    saved_rows = []
    for item, dt, row in matches:
        crop = download_and_crop_image(item, row)
        path = save_crop(crop, row, sentinel1_dir)

        if path is not None:
            saved_rows.append({
                "mmsi": row["MMSI"],
                "timestamp": row["BaseDateTime"].isoformat(),
                "latitude": row["LAT"],
                "longitude": row["LON"],
                "sog": row["SOG"],
                "cog": row["COG"],
                "heading": row.get("Heading"),
                "vessel_name": row.get("VesselName"),
                "imo": row.get("IMO"),
                "callsign": row.get("CallSign"),
                "vessel_tyle": row.get("VesselType"),
                "status": row.get("Status"),
                "length": row["Length"],
                "width": row.get("Width"),
                "draft": row.get("Draft"),
                "cargo": row.get("Cargo"),
                "tranceiver_class": row["TransceiverClass"],
                "image_path": path,
            })

    if saved_rows:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR IGNORE INTO ais (
                mmsi, timestamp, latitude, longitude, sog, cog,
                heading, vessel_name, imo, callsign, vessel_tyle,
                status, length, width, draft, cargo,
                tranceiver_class, image_path
            ) VALUES (
                :mmsi, :timestamp, :latitude, :longitude, :sog, :cog,
                :heading, :vessel_name, :imo, :callsign, :vessel_tyle,
                :status, :length, :width, :draft, :cargo,
                :tranceiver_class, :image_path
            )
        """, saved_rows)
        conn.commit()

