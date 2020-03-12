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
import matplotlib.pyplot as plt
from sytools.misc import isnotebook
from ipywidgets import interact, interactive, fixed, interact_manual, widgets, interactive_output
cmaps = ['viridis', 'plasma', 'inferno', 'magma','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c','flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

def plot_interactive_orthogonal_slices(data, binning=1, guidelines=True):
    if not isnotebook:
        print('Function not suited for working outside of Jupyter notebooks!')
    else:
        get_ipython().magic('matplotlib notebook')
        get_ipython().magic('matplotlib notebook')

    e_range,x_range,y_range = data.shape
    dmin = np.amin(data)
    dmax = np.amax(data)


    w_e = widgets.IntSlider(
        value=e_range // 2,
        min=0,
        max=e_range - 1,
        step=1,
        description='Energy:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    w_kx = widgets.IntSlider(
        value=x_range // 2,
        min=0,
        max=x_range - 1,
        step=1,
        description='kx:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    w_ky = widgets.IntSlider(
        value=y_range // 2,
        min=0,
        max=y_range - 1,
        step=1,
        description='ky:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    w_clim = widgets.FloatRangeSlider(
        value=[.1, .9],
        min=0,
        max=1,
        step=0.001,
        description='Contrast:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    w_cmap = widgets.Dropdown(
        options=cmaps,
        value='terrain',
        description='colormap:',
        disabled=False
    )
    w_bin = widgets.BoundedIntText(
        value=1,
        min=1,
        max=min(data.shape),
        step=1,
        description='resample:',
        disabled=False
    )

    w_interpolate = widgets.Checkbox(
        value=True,
        description='Interpolate',
        disabled=False
    )
    w_grid = widgets.Checkbox(
        value=False,
        description='Grid',
        disabled=False
    )
    w_trackers = widgets.Checkbox(
        value=True,
        description='Trackers',
        disabled=False
    )
    w_trackercol = widgets.ColorPicker(
    concise=False,
    description='tracker line color',
    value='orange'
    )

    ui_pos = widgets.HBox([w_e, w_kx, w_ky])
    ui_color = widgets.HBox([widgets.VBox([w_clim, w_cmap]),
                             widgets.VBox([w_bin, w_interpolate,w_grid]),
                             widgets.VBox([w_trackers,w_trackercol]),
                             ])

    children = [ui_pos, ui_color]
    tab = widgets.Tab(children=children, )
    tab.set_title(0, 'data select')
    tab.set_title(1, 'colormap')

    fig, ax = makeaxis()
    ax0, ax1, ax2 ,cbar_ax = ax


    clim_ = 0.01, .99

    e_img = norm_img(data[data.shape[0] // 2, :, :])
    x_img = norm_img(data[:, data.shape[1] // 2, :])
    y_img = norm_img(data[:, :, data.shape[2] // 2])
    e_plot = ax0.imshow(e_img, cmap='terrain', aspect='auto', interpolation='gaussian', clim=clim_)
    x_plot = ax1.imshow(x_img, cmap='terrain', aspect='auto', interpolation='gaussian', clim=clim_)
    y_plot = ax2.imshow(y_img, cmap='terrain', aspect='auto', interpolation='gaussian', clim=clim_)




    pe_x = ax0.axvline(x_range/2 , c='orange')
    pe_y = ax0.axhline(y_range/2 , c='orange')
    px_y = ax1.axvline(y_range/2 , c='orange')
    px_e = ax1.axhline(e_range/2 , c='orange')
    py_x = ax2.axvline(x_range/2 , c='orange')
    py_e = ax2.axhline(e_range/2 , c='orange')



    def update(e, kx, ky, clim, cmap, binning, interpolate, grid, trackers,trackerscol):

        e_img = norm_img(data[e, ...][::binning, ::binning])
        x_img = norm_img(data[:, kx, ...][::binning, ::binning])
        y_img = norm_img(data[..., ky][::binning, ::binning])
        for axis, plot, img in zip(ax,[e_plot, x_plot, y_plot], [e_img, x_img, y_img.T]):
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

                px_y.set_xdata(ky)
                px_y.set_color(trackerscol)

                px_e.set_ydata(e)
                px_e.set_color(trackerscol)

                py_x.set_xdata(kx)
                py_x.set_color(trackerscol)

                py_e.set_ydata(e)
                py_e.set_color(trackerscol)






    interactive_plot = interactive_output(update, {'e': w_e,
                                                   'kx': w_kx,
                                                   'ky': w_ky,
                                                   'clim': w_clim,
                                                   'cmap': w_cmap,
                                                   'binning': w_bin,
                                                   'interpolate': w_interpolate,
                                                   'grid':w_grid,
                                                   'trackers':w_trackers,
                                                   'trackerscol':w_trackercol,});
    display(interactive_plot, tab)

def norm_img(data,mode='max'):
    out = data - np.amin(data)
    if mode == 'max':
        out /= np.amax(out)
    elif mode == 'mean':
        out /= np.amean(out)
    return out


def makeaxis(**kwargs):
    figsize = kwargs.pop('figsize', (5, 5))
    fig = plt.figure(figsize=figsize, **kwargs)
    plt.tight_layout()
    # [left, bottom, width, height]
    # fig.locator_params(nbins=4)

    cbar_ax = fig.add_axes([.05,.4,.05,4], xticklabels=[], yticklabels=[])
    cbar_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    img_ax   = fig.add_axes([.15, .4, .4, .4], xticklabels=[], yticklabels=[])
    img_ax.xaxis.set_major_locator(plt.LinearLocator(5))
    img_ax.yaxis.set_major_locator(plt.LinearLocator(5))

    xproj_ax = fig.add_axes([.15, .1, .4, .28], xticklabels=[], yticklabels=[])
    xproj_ax.set_xlabel('$k_x$')
    xproj_ax.xaxis.set_major_locator(plt.LinearLocator(5))

    yproj_ax = fig.add_axes([.57, .4, .28, .4], xticklabels=[], yticklabels=[])
    yproj_ax.yaxis.set_label_position("right")
    yproj_ax.set_ylabel('$k_y$')
    yproj_ax.yaxis.set_major_locator(plt.LinearLocator(5))



    for ax in [img_ax, yproj_ax, xproj_ax,cbar_ax]:
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, which='both')
    return fig, (img_ax, xproj_ax, yproj_ax,cbar_ax)



def main():
    pass


if __name__ == '__main__':
    main()