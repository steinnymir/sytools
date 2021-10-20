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

import sys,os
from collections import OrderedDict
from copy import deepcopy
import tifffile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import rotate as sprotate, gaussian_filter
from skimage.transform import rotate as skrotate
# from symmetrize import pointops as po
from xarray import DataArray

from ..symm import mirror
from ..misc import repr_byte_size

# from mpes import fprocessing as fp, analysis as aly, utils as u, visualization as vis


def main():
    bs = BandStructure()


class BandStructure(DataArray):
    """
    Data structure for storage and manipulation of a single band structure (1-3D) dataset.
    Instantiation of the BandStructure class can be done by specifying a (HDF5 or mat) file path
    or by separately specify the data, the axes values and their names.
    """

    keypair = OrderedDict({'ADC': 'tpp', 'X': 'kx', 'Y': 'ky', 't': 'E'})
    dimorder = ['e', 'kx', 'ky', 'kz']
    __slots__ = (
        'rot_sym_order',
        'mir_sym_order',
        'kcenter',
        'high_sym_points',
        'sym_points_dict',
        'axesdict',
        'd',
    )

    def __init__(self, data=None, coords=None, dims=None, faddr=None, **kwds):

        # self.faddr = faddr
        # self.axesdict = OrderedDict() # Container for axis coordinates

        # Specify the symmetries of the band structure
        self.rot_sym_order = kwds.pop('rot_sym_order', 1)  # Lowest rotational symmetry
        self.mir_sym_order = kwds.pop('mir_sym_order', 0)  # No mirror symmetry
        self.kcenter = [0, 0]
        self.high_sym_points = []
        self.sym_points_dict = {}
        self.axesdict = coords

        if faddr:
            data, coords, dims = self._read_h5(faddr)
        super().__init__(data, coords=coords, dims=dims, **kwds)

        # TODO: re-implment file loading/saving in h5 format, as well as other formats
        # Initialization by loading data from an hdf5 file (details see mpes.fprocessing)
        # if self.faddr is not None:
        #
        #     hdfdict = fp.readBinnedhdf5(self.faddr, typ=typ)
        #     data = hdfdict.pop(datakey)
        #
        #     for k, v in self.keypair.items():
        #         # When the file already contains the converted axes, read in directly
        #         try:
        #             self.axesdict[v] = hdfdict[v]
        #         # When the file contains no converted axes, rename coordinates according to the keypair correspondence
        #         except:
        #             self.axesdict[v] = hdfdict[k]
        #
        #     super().__init__(data, coords=self.axesdict, dims=self.axesdict.keys(), **kwds)
        #
        # # Initialization by direct connection to existing data
        # elif self.faddr is None:
        #
        #     self.axesdict = coords
        #     super().__init__(data, coords=coords, dims=dims, **kwds)

        # setattr(self.data, 'datadim', self.data.ndim)
        # self['datadim'] = self.data.ndim

    def symmetrize_mirror(self, axes, method='xr', update=True,
                          ret=False):  # TODO: fix n-dimensional rotation and 0/nan handling

        if isinstance(axes, int):
            axes = (axes,)
        elif isinstance(axes[0], int):
            axes = axes
        elif isinstance(axes[0], str):
            axes = [self.dims.index(x) for x in axes]

        if method == 'np':
            reflected = self.data.copy()
            for axis in axes:
                reflected = np.mean([reflected, mirror(reflected, axes=axis)], axis=0)
            symdata = reflected

        elif method == 'xr':
            reflected = [xr.DataArray(data=mirror(self.data, axes=axis), coords=self.coords, dims=self.dims) for axis in
                         axes]
            symdata = xr.concat([self.where(self > 0), *[r.where(r > 0) for r in reflected]], dim='temp').mean('temp')

        if update:
            self.data = symdata
        if ret:
            return symdata

        # for angle in angles: # TODO: implement arbitrary angle for mirror plane
        #     angle = float(angle) % 360
        #     reflected = np.array(symdata, copy=True)  # reflected image
        #     kwargs = {"center": None, "mode": "constant", "cval": 0, "preserve_range": True}
        #     reflected = rotate(reflected, -angle, **kwargs)
        #     reflected = mirror(reflected, axes=0)
        #     reflected = rotate(reflected, angle, **kwargs)
        #     symdata = reflected

    def symmetrize_rotation(self, rot_plane, order, spline_order=3, mode='wrap', method='xr', update=True,
                            ret=False):  # TODO: fix n-dimensional rotation and 0/nan handling

        if isinstance(rot_plane[0], int):
            rot_plane = rot_plane
        elif isinstance(rot_plane[0], str):
            rot_plane = [self.dims.index(x) for x in rot_plane]

        if 360 % order:
            raise ValueError(
                f"{order}-fold rotational symmetry is not valid (not a divisor of 360)."
            )
        angles = range(0, 360, int(360 / order))

        if method == 'np':
            symdata = self.data.copy()  # np.array(self.data, dtype=np.float, copy=True)

            # bring plane on which to perform symmetrization to the "front"
            symdata = symdata.swapaxes(0, rot_plane[0]).swapaxes(1, rot_plane[1])

            wt = np.ones_like(symdata, dtype=np.uint8)
            kwargs = {"mode": "constant", "cval": 0, "preserve_range": True}  # TODO: re-introduce "center": center

            weights = [skrotate(wt, angle, **kwargs) for angle in angles]
            rotated = [skrotate(symdata, angle, **kwargs) for angle in angles]
            symdata = np.average(rotated, weights=weights, axis=0)
            symdata = symdata.swapaxes(rot_plane[1], 1).swapaxes(rot_plane[0], 0)
        elif method == 'xr':
            rotargs = {'axes': rot_plane, 'reshape': True, 'output': None, 'order': spline_order, 'mode': mode,
                       'cval': 0.0, 'prefilter': True}
            rotated = [xr.DataArray(data=sprotate(self.data, angle, **rotargs), coords=self.coords, dims=self.dims) for
                       angle in angles]
            symdata = xr.concat([self.where(self > 0), *[r.where(r > 0) for r in rotated]], dim='temp').mean('temp')

        if update:
            self.data = symdata
        if ret:
            return symdata

    def symmetrize_translation(self, shift_dict, method='xr', update=True, ret=False):
        axis = [self.dims.index(x) for x in shift_dict.keys()]
        shift = [shift_dict[key] for key in shift_dict.keys()]
        if method == 'xr':
            # translated = self.roll(shifts=shifts,roll_coords=False,)
            # symdata = np.nanmean([self.data,translated],axis=0)
            rolled = xr.DataArray(np.roll(self.data, shift=shift, axis=axis), coords=self.coords, dims=self.dims)
            symdata = xr.concat([self.where(self > 0), rolled.where(rolled > 0)], 'temp').mean('temp')  # da + da_roll

        elif method == 'np':
            w = np.ones_like(self.data)
            w[self.data == 0] = 0
            translated = np.roll(self.data, axis=axis, shift=shift)
            w_t = np.roll(w, axis=axis, shift=shift)
            print('w_t: {}, w: {}'.format(sum(w_t), sum(w)))
            symdata = translated + self.data
            norm = w + w_t
            symdata = np.nan_to_num(symdata / norm)
            # symdata = np.average([translated,self.data],weights=[w_t,w],axis=0)
        if update:
            self.data = symdata
        if ret:
            return symdata

    def blur(self, sigma, order=0, truncate=4.0, update=True, ret=False):
        if isinstance(sigma, int) or isinstance(sigma, float) or isinstance(sigma, tuple) or isinstance(sigma, list):
            s = sigma
        elif isinstance(sigma, dict):
            s = []
            for axis in self.dims:
                try:
                    s.append(sigma[axis])
                except KeyError:
                    s.append(0)
        blurred = gaussian_filter(self.data, sigma=s, order=order, mode='wrap', truncate=truncate)
        if update:
            self.data = blurred
        if ret:
            return BandStructure(data=blurred,coords=self.coords,dims=self.dims)

    def interpolate(self, dims, update=True, ret=False, method='spline', **kwargs):
        intp = self.copy()
        intp = [intp.interpolate_na(dim=dim, method=method, **kwargs) for dim in dims]
        interpolated = xr.concat(intp, dim='temp').mean('temp')
        # for dim in dims:
        #     interpolated = interpolated.interpolate_na(dim=dim, method='spline', **kwargs)
        if update:
            self.data = interpolated.data
        if ret:
            return interpolated

    def save_h5(self, fname, form='h5', save_addr='./'):
        """ save to h5 in simple format."""
        with h5py.File(fname, 'w') as f:
            f.create_dataset('binned/binned', data=self.data)
            for k, v in self.coords.items():
                f.create_dataset('binned/axes/{}'.format(k), data=v)
            for k, v in self.attrs.items():
                f.create_dataset('attrs/{}'.format(k), data=v)

    @staticmethod
    def _read_h5(fname):
        """ Read data saved in h5 format to a list of data, coords and dims."""
        coords = {}
        with h5py.File(fname, 'r') as f:
            data = f['binned/binned'][...]
            for name in f['binned/axes']:
                coords[name] = f['binned/axes'][name][...]
            dims = coords.keys()
            return data, coords, dims

    def load_h5(self, fname):

        if len(self.coords):
            print('instance is not empty, its not safe to load data here!')
        else:
            da, co, di = self._read_h5(fname)
            self.__init__(data=da, coords=co, dims=di)
        # # print(coords)
        #         self.data = data
        #         # self.dims = dims
        #         if len(attrs):
        #             self.attrs = attrs
        #         self.coords.update(coords)

    def load_multiple_h5(self, folder, dim='temp'):

        files = os.listdir(folder)
        da, co, di = self._read_h5(folder + files[0])
        bs = xr.DataArray(data=da, coords=co, dims=di)
        for file in files[1:]:
            da, co, di = self._read_h5(folder + file)
            nbs = xr.DataArray(data=da, coords=co, dims=di)
            bs = xr.concat([bs, nbs], dim='e')
        self.__init__(data=bs.data, coords=bs.coords, dims=bs.dims)

    def export_tiff(self,file):
        tifffile.imwrite(file,data=self.data, dtype=np.float32)

    # -------------- from MPES, not working -----------------------------

    def keypoint_estimate(self, img, dimname='E', pdmethod='daofind', display=False, update=False, ret=False, **kwds):
        """
        Estimate the positions of momentum local maxima (high symmetry points) in the isoenergetic plane.
        """

        if dimname not in self.coords.keys():
            raise ValueError('Need to specify the name of the energy dimension if different from default (E)!')

        else:
            direction = kwds.pop('direction', 'cw')
            pks = po.peakdetect2d(img, method=pdmethod, **kwds)

            # Select center and non-center peaks
            center, verts = po.pointset_center(pks)
            hsp = po.order_pointset(verts, direction=direction)

            if update:
                self.center = center
                self.high_sym_points = hsp

            if display:
                self._view_result(img)
                for ip, p in enumerate(hsp):
                    self['ax'].scatter(p[1], p[0], s=20, c='k')
                    self['ax'].text(p[1] + 3, p[0] + 3, str(ip), c='r')
                self['ax'].text(center[0], center[1], 'C', color='r')

            if ret:
                return center, hsp

    def scale(self, axis, scale_array, update=True, ret=False):
        """
        Scaling and masking of band structure data.

        :Parameters:
            axis : str/tuple
                Axes along which to apply the intensity transform.
            scale_array : nD array
                Scale array to be applied to data.
            update : bool | True
                Options to update the existing array with the intensity-transformed version.
            ret : bool | False
                Options to return the intensity-transformed data.

        :Return:
            scdata : nD array
                Data after intensity scaling.
        """

        scdata = aly.apply_mask_along(self.data, mask=scale_array, axes=axis)

        if update:
            self.data = scdata

        if ret:
            return scdata

    def update_axis(self, axes=None, vals=None, axesdict=None):
        """
        Update the values of multiple axes.

        :Parameters:
            axes : list/tuple | None
                Collection of axis names.
            vals : list/tuple | None
                Collection of axis values.
            axesdict : dict | None
                Axis-value pair for update.
        """

        if axesdict:
            self.coords.update(axesdict)
        else:
            axesdict = dict(zip(axes, vals))
            self.coords.update(axesdict)

    @classmethod
    def resize(cls, data, axes, factor, method='mean', ret=True, **kwds):
        """
        Reduce the size (shape-changing operation) of the axis through rebinning.

        :Parameters:
            data : nD array
                Data to resize (e.g. self.data).
            axes : dict
                Axis values of the original data structure (e.g. self.coords).
            factor : list/tuple of int
                Resizing factor for each dimension (e.g. 2 means reduce by a factor of 2).
            method : str | 'mean'
                Numerical operation used for resizing ('mean' or 'sum').
            ret : bool | False
                Option to return the resized data array.

        :Return:
            Instance of resized n-dimensional array along with downsampled axis coordinates.
        """

        binarr = u.arraybin(data, factor, method=method)

        axesdict = OrderedDict()
        # DataArray sizes cannot be changed, need to create new class instance
        for i, (k, v) in enumerate(axes.items()):
            fac = factor[i]
            axesdict[k] = v[::fac]

        if ret:
            return cls(data=binarr, coords=axesdict, dims=axesdict.keys(), **kwds)

    def rotate(self, data, axis, angle, angle_unit='deg', update=True, ret=False):
        """
        Primary axis rotation that preserves the data size.
        """

        # Slice out
        rdata = np.moveaxis(self.data, axis, 0)
        # data =
        rdata = np.moveaxis(self.data, 0, axis)

        if update:
            self.data = rdata
            # No change of axis values

        if ret:
            return rdata

    def orthogonalize(self, center, update=True, ret=False):
        """
        Align the high symmetry axes in the isoenergetic plane to the row and
        column directions of the image coordinate system.
        """

        pass

    def _view_result(self, img, figsize=(5, 5), cmap='terrain_r', origin='lower'):
        """
        2D visualization of intermediate result.
        """

        self['fig'], self['ax'] = plt.subplots(figsize=figsize)
        self['ax'].imshow(img, cmap=cmap, origin=origin)

    def slicediff(self, slicea, sliceb, slicetype='index', axreduce=None, ret=False, **kwds):
        """
        Calculate the difference of two hyperslices (hs), hsa - hsb.

        :Parameters:
            slicea, sliceb : dict
                Dictionaries for slicing.
            slicetype : str | 'index'
                Type of slicing, 'index' (DataArray.isel) or 'value' (DataArray.sel)
            axreduce : tuple of int | None
                Axes to sum over.
            ret : bool | False
                Options for return.
            **kwds : keyword arguments
                Those passed into DataArray.isel() and DataArray.sel()

        :Return:
            sldiff : class
                Sliced class instance.
        """

        drop = kwds.pop('drop', False)

        # Calculate hyperslices
        if slicetype == 'index':  # Index-based slicing

            sla = self.isel(**slicea, drop=drop)
            slb = self.isel(**sliceb, drop=drop)

        elif slicetype == 'value':  # Value-based slicing

            meth = kwds.pop('method', None)
            tol = kwds.pop('tol', None)

            sla = self.sel(**slicea, method=meth, tolerance=tol, drop=drop)
            slb = self.sel(**sliceb, method=meth, tolerance=tol, drop=drop)

        # Calculate the difference between hyperslices
        if axreduce:
            sldiff = sla.sum(axis=axreduce) - slb.sum(axis=axreduce)

        else:
            sldiff = sla - slb

        if ret:
            return sldiff

    def maxdiff(self, vslice, ret=False):
        """
        Find the hyperslice with maximum difference from the specified one.
        """

        raise NotImplementedError

    def subset(self, axis, axisrange):
        """
        Spawn an instance of the BandStructure class from axis slicing.

        :Parameters:
            axis : str/list
                Axes to subset from.
            axisrange : slice object/list
                The value range of axes to be sliced out.

        :Return:
            An instances of BandStructure, MPESDataset or DataArray class.
        """

        # Determine the remaining coordinate keys using set operations
        restaxes = set(self.coords.keys()) - set(axis)
        bsaxes = set(['kx', 'ky', 'E'])

        # Construct the subset data and the axes values
        axid = self.get_axis_num(axis)
        subdata = np.moveaxis(self.data, axid, 0)

        try:
            subdata = subdata[axisrange, ...].mean(axis=0)
        except:
            subdata = subdata[axisrange, ...]

        # Copy the correct axes values after slicing
        tempdict = deepcopy(self.axesdict)
        tempdict.pop(axis)

        # When the remaining axes are only a set of (kx, ky, E),
        # Create a BandStructure instance to contain it.
        if restaxes == bsaxes:
            bs = BandStructure(subdata, coords=tempdict, dims=tempdict.keys())

            return bs

        # When the remaining axes contain a set of (kx, ky, E) and other parameters,
        # Return an MPESDataset instance to contain it.
        elif bsaxes < restaxes:
            mpsd = MPESDataset(subdata, coords=tempdict, dims=tempdict.keys())

            return mpsd

        # When the remaining axes don't contain a full set of (kx, ky, E),
        # Create a normal DataArray instance to contain it.
        else:
            dray = DataArray(subdata, coords=tempdict, dims=tempdict.keys())

            return dray

def computeBinning(datafolder,axes,bins,ranges,jitter_amplitude,ncores=1):

    sys.path.append('D:/code/')
    import mpes.mpes.fprocessing as fp
    print(f'Binning data from {datafolder}')
    print('binning parameters:')
    bsize = repr_byte_size(64*np.prod([np.float64(x) for x in bins]))
    print(f'axes: {axes}\nshape: {bins}\nbyte size: {bsize}')

    dfp = fp.dataframeProcessor(datafolder=datafolder, ncores=ncores)
    dfp.read(source='folder', ftype='parquet')
    coords = {}
    for a, b, r in zip(axes, bins, ranges):
        coords[a] = np.linspace(r[0], r[1], b)
    dfp.distributedBinning(axes=axes, nbins=bins, ranges=ranges,
                           scheduler='threads', ret=False, jittered=True,
                           jitter_amplitude=jitter_amplitude, pbenv='notebook',
                           weight_axis='processed', binmethod='original')

    return BandStructure(dfp.histdict['binned'], coords=coords, dims=axes)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_fast(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


if __name__ == '__main__':
    main()
