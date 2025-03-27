import numpy as np
import xarray as xr
import hvplot.xarray

def plot_ais_row(row, buffer=0.03, false_color=True):
    img = np.load(row["image_path"])

    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)

    if img.shape[0] != 2:
        print("Expected 2-channel image (VV, VH). Got", img.shape)
        return None

    if np.all(np.isnan(img)):
        print("Image is empty or NaN.")
        return None

    # Normalize channels
    vmin_vv, vmax_vv = np.percentile(img[0][~np.isnan(img[0])], (2, 98))
    vmin_vh, vmax_vh = np.percentile(img[1][~np.isnan(img[1])], (2, 98))

    vv_scaled = np.clip((img[0] - vmin_vv) / (vmax_vv - vmin_vv), 0, 1)
    vh_scaled = np.clip((img[1] - vmin_vh) / (vmax_vh - vmin_vh), 0, 1)

    lat_center = row["latitude"]
    lon_center = row["longitude"]
    H, W = vv_scaled.shape
    lat = np.linspace(lat_center + buffer, lat_center - buffer, H)
    lon = np.linspace(lon_center - buffer, lon_center + buffer, W)

    if false_color:
        # Stack as (band, y, x)
        rgb = np.stack([
            vh_scaled,                            # Red = VH
            (vv_scaled + vh_scaled) / 2,          # Green = blend
            vv_scaled                             # Blue = VV
        ], axis=0)

        da = xr.DataArray(
            rgb,
            coords={
                "band": ["R", "G", "B"],
                "latitude": lat,
                "longitude": lon
            },
            dims=["band", "latitude", "longitude"]
        )

        return da.hvplot.rgb(
            x="longitude",
            y="latitude",
            bands="band",
            frame_width=500,
            frame_height=500,
            invert=True,
            title=f"False-color VV/VH | Ship {row['mmsi']} @ {row['timestamp']}"
        )

    else:
        da = xr.DataArray(
            vv_scaled,
            coords={"latitude": lat, "longitude": lon},
            dims=["latitude", "longitude"]
        )
        return da.hvplot.image(
            x="longitude",
            y="latitude",
            cmap="gray",
            frame_width=500,
            frame_height=500,
            invert=True,
            title=f"VV | Ship {row['mmsi']} @ {row['timestamp']}"
        )
