# -*- coding: utf-8 -*-
"""

@author: Steinn Ymir Agustsson

    Copyright (C) 2018 Steinn Ymir Agustsson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np

cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
         'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
         'Dark2', 'Set1', 'Set2', 'Set3',
         'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
         'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

def donut_mask(dimx, dimy, center, big_radius, small_radius):
    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (small_radius <= distance_from_center) & (distance_from_center <= big_radius)
    return mask


def circle_mask(dimx, dimy, center, radius):
    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (distance_from_center <= radius)
    return mask


def cross_quadrant_mask(dimx, dimy):
    Y, X = np.ogrid[:dimx, :dimy]
    a = (X - Y >= 0) & (dimx - X - Y >= 0)
    b = (X - Y >= 0) & (dimx - X - Y < 0)
    c = (X - Y < 0) & (dimx - X - Y < 0)
    d = (X - Y < 0) & (dimx - X - Y >= 0)
    return a, b, c, d

def color_list(nColors, cmapRange=(0,1),colormap='Blues', invert=False ):
    """ Create list of colors based on a colormap

    for use in iterative plotting. using for c in enumerate(colors)
    :param nColors:
    :param cmapRange:
    :param colormap:
    :param invert:
    :return:
    """
    from matplotlib import cm
    cm_subsection = np.linspace(cmapRange[0], cmapRange[1], nColors)
    cmap = getattr(cm,colormap)

    if invert:
        colors = [cmap(1-x) for x in cm_subsection]
    else:
        colors = [cmap(x) for x in cm_subsection]
    return colors

def main():
    pass


if __name__ == '__main__':
    main()