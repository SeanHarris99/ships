import numpy as np
import xarray as xr
import hvplot.xarray
from scipy.ndimage import gaussian_filter1d, convolve


def get_ais_img(row, buffer=.03):
    return np.load(row["image_path"])

def plot_ais_img(row, buffer=0.03, mode="false_color"):
    img = np.load(row["image_path"])

    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)

    if img.shape[0] != 2:
        print("Expected 2-channel image (VV, VH). Got", img.shape)
        return None

    if np.all(np.isnan(img)):
        print("Image is empty or NaN.")
        return None

    vv = img[0]
    vh = img[1]

    # Normalize each channel
    vmin_vv, vmax_vv = np.percentile(vv[~np.isnan(vv)], (2, 98))
    vmin_vh, vmax_vh = np.percentile(vh[~np.isnan(vh)], (2, 98))

    vv_scaled = np.clip((vv - vmin_vv) / (vmax_vv - vmin_vv), 0, 1)
    vh_scaled = np.clip((vh - vmin_vh) / (vmax_vh - vmin_vh), 0, 1)

    lat_center = row["latitude"]
    lon_center = row["longitude"]
    H, W = vv.shape
    lat = np.linspace(lat_center + buffer, lat_center - buffer, H)
    lon = np.linspace(lon_center - buffer, lon_center + buffer, W)

    if mode == "false_color":
        rgb = np.stack([
            vh_scaled,
            (vv_scaled + vh_scaled) / 2,
            vv_scaled
        ], axis=0)

        da = xr.DataArray(
            rgb,
            coords={"band": ["R", "G", "B"], "latitude": lat, "longitude": lon},
            dims=["band", "latitude", "longitude"]
        )

        return da.hvplot.rgb(
            x="longitude", y="latitude", bands="band",
            frame_width=500, frame_height=500, invert=True,
            title=f"False-color VV/VH | Ship {row['mmsi']} @ {row['timestamp']}"
        )

    elif mode == "vv":
        da = xr.DataArray(vv_scaled, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
        return da.hvplot.image(x="longitude", y="latitude", cmap="gray", frame_width=500, frame_height=500, invert=True,
                               title=f"VV | Ship {row['mmsi']} @ {row['timestamp']}")

    elif mode == "vh":
        da = xr.DataArray(vh_scaled, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
        return da.hvplot.image(x="longitude", y="latitude", cmap="gray", frame_width=500, frame_height=500, invert=True,
                               title=f"VH | Ship {row['mmsi']} @ {row['timestamp']}")

    elif mode == "ratio":
        ratio = vv / (vh + 1e-6)
        da = xr.DataArray(ratio, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
        return da.hvplot.image(x="longitude", y="latitude", cmap="plasma", frame_width=500, frame_height=500, invert=True,
                               title=f"VV/VH Ratio | Ship {row['mmsi']}")

    elif mode == "difference":
        diff = vv - vh
        da = xr.DataArray(diff, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
        return da.hvplot.image(x="longitude", y="latitude", cmap="coolwarm", frame_width=500, frame_height=500, invert=True,
                               title=f"VV - VH Difference | Ship {row['mmsi']}")

    elif mode == "ndpi":
        ndpi = (vv - vh) / (vv + vh + 1e-6)
        da = xr.DataArray(ndpi, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
        return da.hvplot.image(x="longitude", y="latitude", cmap="viridis", frame_width=500, frame_height=500, invert=True,
                               title=f"NDPI | Ship {row['mmsi']}")

    elif mode == "wake_score":
            wake = vh / (vv + 1e-6)
            da = xr.DataArray(wake, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
            return da.hvplot.image(x="longitude", y="latitude", cmap="magma", frame_width=500, frame_height=500, invert=True,
                                title=f"Wake Score (VH/VV) | Ship {row['mmsi']}")

    elif mode == "wake_score_filtered":
        wake_score = vh / (vv + 1e-6)
        filtered = directional_wake_filter(wake_score, angle_deg=30, sigma=2)
        da = xr.DataArray(filtered, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"])
        return da.hvplot.image(x="longitude", y="latitude", cmap="magma", frame_width=500, frame_height=500, invert=True,
                               title=f"Wake Score (VH/VV) | Ship {row['mmsi']}")

    else:
        print(f"Unknown mode '{mode}'. Supported modes: false_color, vv, vh, ratio, difference, ndpi, wake_score, wake_score_filtered")
        return None




def directional_wake_filter(wake_score, angle_deg=0, sigma=2.0):
    """
    Applies a directional Gaussian filter to the wake score image.

    Args:
        wake_score (np.ndarray): 2D wake score image (VH/VV).
        angle_deg (float): Direction to emphasize (degrees from horizontal).
        sigma (float): Standard deviation of Gaussian kernel.

    Returns:
        np.ndarray: Enhanced image emphasizing linear structures in the given direction.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Create directional Gaussian filters
    # Gaussian along axis perpendicular to angle, smooth minimal in other
    gx = gaussian_filter1d(wake_score, sigma=sigma, axis=1, mode='reflect')
    gy = gaussian_filter1d(wake_score, sigma=sigma, axis=0, mode='reflect')

    # Weighted sum in the desired direction
    filtered = cos_a * gx + sin_a * gy
    return filtered

