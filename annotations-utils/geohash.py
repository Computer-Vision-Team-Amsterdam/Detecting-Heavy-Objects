import timeit

import matplotlib
matplotlib.use("TkAgg")
import cartopy.crs
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy
import pandas

import pyinterp


def _sort_colors(colors):
    """Sort colors by hue, saturation, value and name in descending order."""
    by_hsv = sorted(
        (tuple(matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))),
         name) for name, color in colors.items())
    return [name for hsv, name in reversed(by_hsv)]


def _plot_box(ax, code, color, caption=True):
    """Plot a GeoHash bounding box."""
    box = pyinterp.GeoHash.from_string(code.decode()).bounding_box()
    x0 = box.min_corner.lon
    x1 = box.max_corner.lon
    y0 = box.min_corner.lat
    y1 = box.max_corner.lat
    dx = x1 - x0
    dy = y1 - y0
    box = matplotlib.patches.Rectangle((x0, y0),
                                       dx,
                                       dy,
                                       alpha=0.5,
                                       color=color,
                                       ec='black',
                                       lw=1,
                                       transform=cartopy.crs.PlateCarree())
    ax.add_artist(box)
    if not caption:
        return
    rx, ry = box.get_xy()
    cx = rx + box.get_width() * 0.5
    cy = ry + box.get_height() * 0.5
    ax.annotate(code.decode(), (cx, cy),
                color='w',
                weight='bold',
                fontsize=16,
                ha='center',
                va='center')


def plot_geohash_grid(precision,
                      polygon=None,
                      caption=True,
                      color_list=None,
                      inc=7):
    """Plot geohash bounding boxes."""
    color_list = color_list or matplotlib.colors.CSS4_COLORS
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
    if polygon is not None:
        box = polygon.envelope() if isinstance(
            polygon, pyinterp.geodetic.Polygon) else polygon

        ax.set_extent(
            [
                box.min_corner.lon,
                box.max_corner.lon,
                box.min_corner.lat,
                box.max_corner.lat,
            ],
            crs=cartopy.crs.PlateCarree(),
        )
    else:
        box = None
    colors = _sort_colors(color_list)
    ic = 0
    codes = pyinterp.geohash.bounding_boxes(polygon, precision=precision)

    color_codes = {codes[0][0]: colors[ic]}
    for item in codes:
        prefix = item[precision - 1]
        if prefix not in color_codes:
            ic += inc
            color_codes[prefix] = colors[ic % len(colors)]
        _plot_box(ax, item, color_codes[prefix], caption)
    ax.stock_img()
    ax.coastlines()
    ax.grid(visible=True)


plot_geohash_grid(1)
