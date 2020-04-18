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
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets, interactive_output

from ..misc import isnotebook
from ..plot import cmaps

def main():
    pass


def orthoslices_3D(data, normalize=True, continuous_update=False, **kwargs):
    if not isnotebook:
        print('Function not suited for working outside of Jupyter notebooks!')
    else:
        get_ipython().magic('matplotlib notebook')
        get_ipython().magic('matplotlib notebook')

    e_range, x_range, y_range = data.shape
    dmin = np.amin(data)
    dmax = np.amax(data)

    w_e = widgets.IntSlider(value=e_range // 2, min=0, max=e_range - 1, step=1, description='Energy:',
                            disabled=False, continuous_update=continuous_update, orientation='horizontal', readout=True,
                            readout_format='d')
    w_kx = widgets.IntSlider(value=x_range // 2, min=0, max=x_range - 1, step=1, description='kx:',
                             disabled=False, continuous_update=continuous_update, orientation='horizontal',
                             readout=True,
                             readout_format='d')
    w_ky = widgets.IntSlider(value=y_range // 2, min=0, max=y_range - 1, step=1, description='ky:', disabled=False,
                             continuous_update=continuous_update, orientation='horizontal', readout=True,
                             readout_format='d')
    w_clim = widgets.FloatRangeSlider(value=[.1, .9], min=0, max=1, step=0.001, description='Contrast:', disabled=False,
                                      continuous_update=continuous_update, orientation='horizontal', readout=True,
                                      readout_format='.1f')
    w_cmap = widgets.Dropdown(options=cmaps, value='terrain', description='colormap:', disabled=False)
    w_bin = widgets.BoundedIntText(value=1, min=1, max=min(data.shape), step=1, description='resample:', disabled=False)
    w_interpolate = widgets.Checkbox(value=True, description='Interpolate', disabled=False)
    w_grid = widgets.Checkbox(value=False, description='Grid', disabled=False)
    w_trackers = widgets.Checkbox(value=True, description='Trackers', disabled=False)
    w_trackercol = widgets.ColorPicker(concise=False, description='tracker line color', value='orange')

    ui_pos = widgets.HBox([w_e, w_kx, w_ky])
    ui_color = widgets.HBox([widgets.VBox([w_clim, w_cmap]),
                             widgets.VBox([w_bin, w_interpolate, w_grid]),
                             widgets.VBox([w_trackers, w_trackercol]),
                             ])

    children = [ui_pos, ui_color]
    tab = widgets.Tab(children=children, )
    tab.set_title(0, 'data select')
    tab.set_title(1, 'colormap')

    figsize = kwargs.pop('figsize', (5, 5))
    fig = plt.figure(figsize=figsize, **kwargs)
    plt.tight_layout()
    # [left, bottom, width, height]
    # fig.locator_params(nbins=4)

    # cbar_ax = fig.add_axes([.05,.4,.05,4], xticklabels=[], yticklabels=[])
    # cbar_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    img_ax = fig.add_axes([.15, .4, .4, .4], xticklabels=[], yticklabels=[])
    img_ax.xaxis.set_major_locator(plt.LinearLocator(5))
    img_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    xproj_ax = fig.add_axes([.15, .1, .4, .28], xticklabels=[], yticklabels=[])
    xproj_ax.set_xlabel('$k_x$')
    xproj_ax.xaxis.set_major_locator(plt.LinearLocator(5))

    yproj_ax = fig.add_axes([.57, .4, .28, .4], xticklabels=[], yticklabels=[])
    yproj_ax.yaxis.set_label_position("right")
    yproj_ax.set_ylabel('$k_y$')
    yproj_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    for ax in [img_ax, yproj_ax, xproj_ax]:  # ,cbar_ax]:
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, which='both')

    clim_ = 0.01, .99

    e_img = norm_img(data[data.shape[0] // 2, :, :])
    y_img = norm_img(data[:, data.shape[1] // 2, :])
    x_img = norm_img(data[:, :, data.shape[2] // 2].T)
    e_plot = img_ax.imshow(e_img, cmap='terrain', aspect='auto', interpolation='gaussian',
                           clim=clim_, )  # origin='lower')
    x_plot = yproj_ax.imshow(x_img, cmap='terrain', aspect='auto', interpolation='gaussian',
                             clim=clim_, )  # origin='lower')
    y_plot = xproj_ax.imshow(y_img, cmap='terrain', aspect='auto', interpolation='gaussian',
                             clim=clim_, )  # origin='lower')

    pe_x = img_ax.axvline(x_range / 2, c='orange')
    pe_y = img_ax.axhline(y_range / 2, c='orange')
    px_x = xproj_ax.axvline(x_range / 2, c='orange')
    px_e = xproj_ax.axhline(e_range / 2, c='orange')
    py_y = yproj_ax.axhline(y_range / 2, c='orange')
    py_e = yproj_ax.axvline(e_range / 2, c='orange')

    def update(e, kx, ky, clim, cmap, binning, interpolate, grid, trackers, trackerscol):
        if normalize:
            e_img = norm_img(data[e, :, :][::binning, ::binning])
            y_img = norm_img(data[:, ky, :][::binning, ::binning])
            x_img = norm_img(data[:, :, kx][::binning, ::binning])
        else:
            e_img = data[e, :, :][::binning, ::binning]
            y_img = data[:, ky, :][::binning, ::binning]
            x_img = data[:, :, kx][::binning, ::binning]
        for axis, plot, img in zip([img_ax, yproj_ax, xproj_ax], [e_plot, x_plot, y_plot], [e_img, x_img.T, y_img]):

            plot.set_data(img)
            plot.set_clim(clim)
            plot.set_cmap(cmap)
            axis.grid(grid)
            if interpolate:
                plot.set_interpolation('gaussian')
            else:
                plot.set_interpolation(None)
            if trackers:
                pe_x.set_xdata(kx)
                pe_x.set_color(trackerscol)
                pe_y.set_ydata(ky)
                pe_y.set_color(trackerscol)
                px_x.set_xdata(kx)
                px_x.set_color(trackerscol)
                px_e.set_ydata(e)
                px_e.set_color(trackerscol)
                py_y.set_ydata(ky)
                py_y.set_color(trackerscol)
                py_e.set_xdata(e)
                py_e.set_color(trackerscol)

    interactive_plot = interactive_output(update, {'e': w_e,
                                                   'kx': w_kx,
                                                   'ky': w_ky,
                                                   'clim': w_clim,
                                                   'cmap': w_cmap,
                                                   'binning': w_bin,
                                                   'interpolate': w_interpolate,
                                                   'grid': w_grid,
                                                   'trackers': w_trackers,
                                                   'trackerscol': w_trackercol, });
    display(interactive_plot, tab)
    # display(tab)

    # return fig


def orthoslices_4D(data, axis_order=['E', 'kx', 'ky', 'kz'], normalize=True, continuous_update=True, **kwargs):
    if not isnotebook:
        raise EnvironmentError('Function not suited for working outside of Jupyter notebooks!')
    else:
        get_ipython().magic('matplotlib notebook')
        get_ipython().magic('matplotlib notebook')

    assert len(data.shape) == 4, 'Data should be 4-dimensional, but data has {} dimensions'.format(data.shape)

    # make controls for data slicers
    # slicers = []
    # for shape, name in zip(data.shape, axis_order):
    #     slicers.append(widgets.IntSlider(value=shape // 2,
    #                                      min=0,
    #                                      max=shape - 1,
    #                                      step=1,
    #                                      description=name,
    #                                      disabled=False,
    #                                      continuous_update=False,
    #                                      orientation='horizontal',
    #                                      readout=True,
    #                                      readout_format='d'
    #                                      ))

    e_range, x_range, y_range, z_range = data.shape

    w_e = widgets.IntSlider(value=e_range // 2, min=0, max=e_range - 1, step=1, description='Energy:',
                            disabled=False, continuous_update=continuous_update, orientation='horizontal', readout=True,
                            readout_format='d')
    w_kx = widgets.IntSlider(value=x_range // 2, min=0, max=x_range - 1, step=1, description='kx:',
                             disabled=False, continuous_update=continuous_update, orientation='horizontal',
                             readout=True,
                             readout_format='d')
    w_ky = widgets.IntSlider(value=y_range // 2, min=0, max=y_range - 1, step=1, description='ky:', disabled=False,
                             continuous_update=continuous_update, orientation='horizontal', readout=True,
                             readout_format='d')
    w_kz = widgets.IntSlider(value=z_range // 2, min=0, max=y_range - 1, step=1, description='ky:', disabled=False,
                             continuous_update=continuous_update, orientation='horizontal', readout=True,
                             readout_format='d')

    slicers = [w_e, w_kx, w_ky, w_kz]
    ui_slicers = widgets.HBox(slicers)

    # make controls for graphics appearance
    w_clim = widgets.FloatRangeSlider(value=[.1, .9], min=0, max=1, step=0.001, description='Contrast:', disabled=False,
                                      continuous_update=True, orientation='horizontal', readout=True,
                                      readout_format='.1f')
    w_cmap = widgets.Dropdown(options=cmaps, value='terrain', description='colormap:', disabled=False)
    w_bin = widgets.BoundedIntText(value=1, min=1, max=min(data.shape), step=1, description='resample:', disabled=False)
    w_interpolate = widgets.Checkbox(value=True, description='Interpolate', disabled=False)
    w_grid = widgets.Checkbox(value=False, description='Grid', disabled=False)
    w_trackers = widgets.Checkbox(value=True, description='Trackers', disabled=False)
    w_trackercol = widgets.ColorPicker(concise=False, description='tracker line color', value='orange')
    ui_color = widgets.HBox([widgets.VBox([w_clim, w_cmap]),
                             widgets.VBox([w_bin, w_interpolate, w_grid]),
                             widgets.VBox([w_trackers, w_trackercol]),
                             ])

    tab = widgets.Tab(children=[ui_slicers, ui_color], )
    tab.set_title(0, 'Data slicing')
    tab.set_title(1, 'Graphics')

    figsize = kwargs.pop('figsize', (5, 5))
    fig = plt.figure(figsize=figsize, **kwargs)
    plt.tight_layout()

    img_ax = fig.add_axes([.15, .4, .4, .4], xticklabels=[], yticklabels=[])
    img_ax.xaxis.set_major_locator(plt.LinearLocator(5))
    img_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    xproj_ax = fig.add_axes([.15, .1, .4, .28], xticklabels=[], yticklabels=[])
    xproj_ax.set_xlabel('$k_x$')
    xproj_ax.xaxis.set_major_locator(plt.LinearLocator(5))

    yproj_ax = fig.add_axes([.57, .4, .28, .4], xticklabels=[], yticklabels=[])
    yproj_ax.yaxis.set_label_position("right")
    yproj_ax.set_ylabel('$k_y$')
    yproj_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    for ax in [img_ax, yproj_ax, xproj_ax]:  # ,cbar_ax]:
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, which='both')

    clim_ = 0.01, .99

    e_img = norm_img(data[data.shape[0] // 2, :, :, data.shape[3] // 2])
    y_img = norm_img(data[:, data.shape[1] // 2, :, data.shape[3] // 2])
    x_img = norm_img(data[:, :, data.shape[2] // 2, data.shape[3] // 2].T)
    e_plot = img_ax.imshow(e_img, cmap='terrain', aspect='auto', interpolation='gaussian',
                           clim=clim_, )  # origin='lower')
    x_plot = yproj_ax.imshow(x_img, cmap='terrain', aspect='auto', interpolation='gaussian',
                             clim=clim_, )  # origin='lower')
    y_plot = xproj_ax.imshow(y_img, cmap='terrain', aspect='auto', interpolation='gaussian',
                             clim=clim_, )  # origin='lower')

    pe_x = img_ax.axvline(x_range / 2, c='orange')
    pe_y = img_ax.axhline(y_range / 2, c='orange')
    px_x = xproj_ax.axvline(x_range / 2, c='orange')
    px_e = xproj_ax.axhline(e_range / 2, c='orange')
    py_y = yproj_ax.axhline(y_range / 2, c='orange')
    py_e = yproj_ax.axvline(e_range / 2, c='orange')

    def update(e, kx, ky, kz, clim, cmap, binning, interpolate, grid, trackers, trackerscol):
        if normalize:
            e_img = norm_img(data[e, :, :, kz][::binning, ::binning])
            y_img = norm_img(data[:, ky, :, kz][::binning, ::binning])
            x_img = norm_img(data[:, :, kx, kz][::binning, ::binning])
        else:
            e_img = data[e, :, :, kz][::binning, ::binning]
            y_img = data[:, ky, :, kz][::binning, ::binning]
            x_img = data[:, :, kx, kz][::binning, ::binning]
        for axis, plot, img in zip([img_ax, yproj_ax, xproj_ax], [e_plot, x_plot, y_plot], [e_img, x_img.T, y_img]):

            plot.set_data(img)
            plot.set_clim(clim)
            plot.set_cmap(cmap)
            axis.grid(grid)
            if interpolate:
                plot.set_interpolation('gaussian')
            else:
                plot.set_interpolation(None)
            if trackers:
                pe_x.set_xdata(kx)
                pe_x.set_color(trackerscol)
                pe_y.set_ydata(ky)
                pe_y.set_color(trackerscol)
                px_x.set_xdata(kx)
                px_x.set_color(trackerscol)
                px_e.set_ydata(e)
                px_e.set_color(trackerscol)
                py_y.set_ydata(ky)
                py_y.set_color(trackerscol)
                py_e.set_xdata(e)
                py_e.set_color(trackerscol)

    interactive_plot = interactive_output(update, {'e': w_e,
                                                   'kx': w_kx,
                                                   'ky': w_ky,
                                                   'kz': w_kz,
                                                   'clim': w_clim,
                                                   'cmap': w_cmap,
                                                   'binning': w_bin,
                                                   'interpolate': w_interpolate,
                                                   'grid': w_grid,
                                                   'trackers': w_trackers,
                                                   'trackerscol': w_trackercol, });
    display(interactive_plot, tab)
    # display(tab)

    # return fig


def norm_img(data, mode='max'):
    out = np.nan_to_num(data)
    out -= np.amin(out)
    if mode == 'max':
        out /= np.amax(out)
    elif mode == 'mean':
        out /= np.amean(out)
    return out


def k_par(px, bz_width_px=219, bz_width=2.215):
    '''k parallel from pixel size
    dgg: distance between 2 gamma points'''
    return px * bz_width / bz_width_px


def k_z(d, kf=37.285, bz_width_px=219):
    '''k_z from pixel distance from k-center
    d: distance in pixels
    kf photon momentum'''
    return kf * (1 - np.cos(np.arcsin(k_par(d, bz_width_px) / kf)))


def px_to_xyz(px, k_center, bz_width_px, bz_width, kf):
    """ convert detector pixel to momentum coordinates
    param:
        px: tuple
            detector coordinates (row,col)
        k_center: tuple
            momentum center on detector
        bz_width_px: int
            size of a brillouin zone in pixels
        bz_width: float
            size of a brillouin zone in inverse Angstroms
    return:
        K: tuple:
            kx,ky,kz, in inverse angstroms
    """
    kx_px = k_center[1] - px[1]
    ky_px = k_center[0] - px[0]
    kz_px = point_distance(px, k_center)
    return k_par(kx_px, bz_width_px, bz_width), k_par(ky_px, bz_width_px, bz_width), k_z(kz_px, kf, bz_width_px)


def point_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def get_k_xyz_idx(pt, k_center=(179, 737), bz_width_px=218, kf=37.285):
    """ return the kx,ky,kz coordinates of a point.
    param:
        pt: tuple(float,float)
            point to evaluate
        k_center:
            center of momentum space
        dgg: float
            distance between 2 gamma points - BZ width
        kf: float
            final momentum from energy of photoemission photon

    given distance from gamma point and size of the brillouin zone"""
    x_c, y_c = k_center
    #     x = k_par((pt[0]-x_c)%dgg - dgg/2 ,dgg)
    #     y = k_par((pt[1]-y_c)%dgg- dgg/2,dgg)
    x = (pt[0] - x_c) % bz_width_px
    if x > bz_width_px / 2: x -= bz_width_px
    y = (pt[1] - y_c) % bz_width_px
    if y > bz_width_px / 2: y -= bz_width_px

    z = get_kz(pt, k_center, kf, bz_width_px)
    return (k_par(x), k_par(y), z)


def get_k_xyz(pt, k_center=(179, 737), dgg=218, kf=37.285):
    """ return the kx,ky,kz coordinates of a point.
    param:
        pt: tuple(float,float)
            point to evaluate
        k_center:
            center of momentum space
        dgg: float
            distance between 2 gamma points - BZ width
        kf: float
            final momentum from energy of photoemission photon

    given distance from gamma point and size of the brillouin zone"""
    x_c, y_c = k_center
    #     x = k_par((pt[0]-x_c)%dgg - dgg/2 ,dgg)
    #     y = k_par((pt[1]-y_c)%dgg- dgg/2,dgg)
    x = (pt[0] - x_c) % dgg
    if x > dgg / 2: x -= dgg
    y = (pt[1] - y_c) % dgg
    if y > dgg / 2: y -= dgg

    z = get_kz(pt, k_center, kf, dgg)
    return (k_par(x), k_par(y), z)


if __name__ == '__main__':
    main()
