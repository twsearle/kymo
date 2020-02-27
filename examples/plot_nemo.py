"""
Plot NEMO AMM15 using datashader to speed things up
"""

import datashader
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import datashader
from datashader import transfer_functions as tf, reductions as rd
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [4, 3]


def datashader_stretch(xarr, x_name, y_name, x_range, y_range):
    """ rasterise data using datashader """

    h = xarr.values.shape[0]
    w = xarr.values.shape[1]
    canvas = datashader.Canvas(
        plot_height=1440, plot_width=1920, x_range=x_range, y_range=y_range
    )

    image = canvas.quadmesh(xarr, x=x_name, y=y_name)
    return image


if __name__ == "__main__":
    model_data = xr.open_dataset("/scratch/tsearle/model.amm15.nc")
    model_surf = model_data.isel(axis_nbounds=0, time_counter=0, deptht=0)
    model_sst = model_surf["votemper"]
    # gx = model_surf["nav_lon"].data
    # gy = model_surf["nav_lat"].data

    # confine to AMM7/AMM15 region for comparison
    im_data = datashader_stretch(model_sst, "nav_lon", "nav_lat", (-24, 18), (30, 66))

    im_values = np.ma.masked_array(im_data.values, mask=np.isnan(im_data.values))

    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax = plt.axes(projection=ccrs.RotatedPole(pole_longitude=0.0, pole_latitude=60.0))
    ax.pcolormesh(im_data["nav_lon"].values, im_data["nav_lat"].values, im_values)
    ax.coastlines()
    plt.savefig("test.png")
