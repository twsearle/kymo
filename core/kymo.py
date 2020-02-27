"""
Plot fields using datashader and matplotlib pcolormesh
"""

import logging
import dataclasses
import yaml
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import datashader
from datashader import transfer_functions as tf, reductions as rd
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [4, 3]
# logging.basicConfig(level=logging.DEBUG)


@dataclasses.dataclass
class FieldConfig:
    """Class for holding configuration information for a single plot"""

    output_figure_name: str = ""
    measure_name: str = ""
    model_name: str = ""
    input_file_name: str = ""
    level_number: int = 0
    depth: int = 0
    time_counter: int = 0
    x_range: tuple = (-24, 18)
    y_range: tuple = (30, 66)
    pcolormesh_kwargs: dict = dataclasses.field(default_factory=dict)

    def load_field(self):
        """Return a 2D xarray of the data for a given Field configuration"""
        logging.debug("Reading: %s", self.input_file_name)
        model_data = xr.open_dataset(self.input_file_name)
        model_surf = model_data.isel(
            time_counter=self.time_counter, deptht=self.level_number
        )
        return model_surf[self.measure_name]


def datashader_stretch(xarr, x_name, y_name, x_range, y_range):
    """ rasterise data using datashader """

    h = xarr.values.shape[0]
    w = xarr.values.shape[1]
    canvas = datashader.Canvas(
        plot_height=1440, plot_width=1920, x_range=x_range, y_range=y_range
    )

    image = canvas.quadmesh(xarr, x=x_name, y=y_name)
    return image


def plot(ax, im_data, pcolormesh_kwargs={}):
    """Plot datashader image using pcolourmesh"""

    im_values = np.ma.masked_array(im_data.values, mask=np.isnan(im_data.values))

    s = ax.pcolormesh(
        im_data["nav_lon"].values,
        im_data["nav_lat"].values,
        im_values,
        **pcolormesh_kwargs
    )
    return s


def config_read(config_file_name=None):
    """
    Read elements of a config file into the FieldConfig dataclass

    At some point this will want to be able to generate multiple FieldConfig
    objects from a single configuration option.
    """

    if config_file_name == None:
        config_file_name = "cfg.yml"
    with open(config_file_name, "r") as fp:
        field_configs = yaml.full_load(fp)

    logging.debug("configuration file contents:")
    logging.debug(field_configs)

    fields = [FieldConfig(**v) for v in field_configs]

    for f in fields:
        logging.debug(f)

    return fields


if __name__ == "__main__":

    fields = config_read()

    for field in fields:
        slab = field.load_field()

        im_data = datashader_stretch(
            slab, "nav_lon", "nav_lat", field.x_range, field.y_range
        )

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax = plt.axes(projection=ccrs.PlateCarree())
        s = plot(ax, slab)
        ax.coastlines(resolution="110m")
        fig.savefig(field.output_figure_name)
