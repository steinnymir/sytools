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
import json
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import cv2 as cv
import h5py
import numpy as np
import tifffile
from skimage.filters import gaussian
import skimage.filters as skfilt
from scipy.ndimage import rotate
from ..misc import iterable
from .utils import point_distance


def main():

    raw = DldArray()

class DldArray(xr.DataArray):
    """ Data structure for a 3D dataset from Momenutm microscopy measurements """

    dimorder = ['ToF', 'X', 'Y']
    __slots__ = ('load_tiff','load_h5')

    def __init__(self, data=None, coords=None, dims=None, chunks=None, fname=None, h5addr=None, **kwds):

        if fname:
            self.attrs['fname'] = fname
        if h5addr:
            self.attrs['h5addr'] = h5addr
            data, coords, dims = self._load_h5(h5addr)

        if data:
            data = data
        super().__init__(data, coords=coords, dims=dims, **kwds)
        if chunks:
            if isinstance(chunks,dict):
                self.chunk(**chunks)
            else:
                self.chunk(chunks=chunks)


    def load_tiff(self,file,coords=None,dims=None):
        """ load data from tiff file"""
        self.data = tifffile.imread(file)
        # TODO: set up coordinates and dimensions

    def load_h5(self,fname=None, h5addr=None, ret=False, update=True):
        """ read data from the given hdf5 address.

        This might be an hdf5 file or a group in an hdf5.
        File only for now...."""
        # TODO: extend to address inside an h5file
        if fname is None:
            fname = self.attrs['fname']
        if h5addr is None:
            h5addr = self.attrs['h5addr']
        if h5addr[-1] != '/':  h5addr+='/'

        coords = {}
        with h5py.File(fname, 'r') as f:
            data = f[f'{h5addr}data'][...]
            for name in f[f'{h5addr}axes']:
                coords[name] = f[f'{h5addr}axes'][name][...]
            dims = coords.keys()
        if ret:
            return data, coords, dims
        if update:
            super().__init__(data, coords=coords, dims=dims)

        # with h5py.File(h5addr, mode='r') as f:
        #     self['data'] = f[h5addr+'/data']
        #     self['data'] = f[h5addr+'/data']
        #     self['data'] = f[h5addr+'/data']
        #
        #     try:
        #         self.raw_data = f['raw/data'][...]
        #     except KeyError:
        #         pass
        #     self.data = f['processed/data'][...]
        #     self.mask = f['processed/mask'][...]
        #     self.hist_str = f['history'][()]

    def update_h5(self,file=None):
        """ update the h5 file linked to this DataArray."""
        raise NotImplementedError

class DldDataset(xr.Dataset):

    def __init__(self, faddr=None, raw_data=None, processed=None, data_vars=None, mask=None,
                 coords=None, attrs=None, **kwargs):

        if faddr:
            self._load_h5(faddr)
            self._faddr
        if processed:
            self['processed'] = processed
        if raw_data:
            self['raw'] = raw_data
        if mask:
            self['mask'] = mask
        if history:
            self.attrs['history'] = history



        super().__init__(data_vars=data_vars,coords=coords,attrs=attrs)

    def _load_h5(self,file):
        print(f'loading data from file "{file}"...')
        with h5py.File(file, mode='r') as f:
            try:
                self.raw_data = f['raw/data'][...]
            except KeyError:
                pass
            self._data = f['processed/data'][...]
            self._mask = f['processed/mask'][...]
            self._hist_str = f['history'][()]


class DetectorData(object):

    def __init__(self, faddr=None, raw_data=None, processed=None, history=None, mask=None):
        self.data = None
        self.raw_data = None
        self.history = OrderedDict()
        self.mask = None
        self.faddr = None

        self.bz_width = None
        self.k_center = None

        if faddr is not None:
            self.load(faddr)
            self.faddr = faddr

        if isinstance(history, dict):  # TODO: add tracking of data imports
            self.history = history
        elif isinstance(history, str):
            self.history = json.loads(history)
        else:
            self.history = OrderedDict()

        if raw_data is not None:
            if isinstance(raw_data, np.ndarray):
                self.raw_data = raw_data.astype(np.uint16)
            elif isinstance(raw_data, str):
                if os.path.isfile(raw_data):
                    self.raw_data_address = raw_data
                    self.raw_data = tifffile.imread(raw_data).astype(np.uint16)
                else:
                    raise ValueError(f'invalid entry "{raw_data}" for Raw Data')
            if self.data is None:
                self.data = self.raw_data.astype(np.float64,copy=True)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                self.raw_data = mask.astype(np.bool_)
            elif isinstance(mask, str):
                self.mask = np.load(mask).astype(np.bool)

        if processed is not None:
            self.data = processed


    @property
    def hist_str(self):
        return json.dumps(self.history)
    @hist_str.setter
    def hist_str(self,str):
        self.history = json.loads(str)

    @property
    def masked_data(self):
        return self.data*self.mask

    def reset_data(self):
        self.data = self.raw_data.astype(np.float64,copy=True)

    def reset_history(self):
        self.history = OrderedDict()

    def run_command_history(self, history=None):
        if history is None:
            hist = history
        else:
            hist = self.history
        for k, v in hist.items():
            if not iterable(v):
                v = [v]
            getattr(self, k)(*v)

    def renormalize_DOS(self, use_mask=True):
        """ normalize the """
        print('Renormalizing DOS and applying mask...')
        if use_mask and self.mask is not None:
            data = self.data * self.mask
            raw_data = self.raw_data * self.mask
        else:
            data = self.data
            raw_data = self.raw_data

        norm = data.sum((1, 2)) / raw_data.sum((1, 2)).astype(np.float64)
        self.data = data / norm[:, None, None]
        self.history['renormalize_DOS'] = {'use_mask': True}

    def make_finite(self, substitute=0.0):
        """ set all nans and infs to 0. or whatever specified value in substitute"""
        print('Handling nans and infs...')
        self.data[~np.isfinite(self.data)] = substitute
        self.history['make_finite'] = {'substitute': True}

    def filter_diffraction(self, sigma=15):
        """ divide by self, gaussian blurred along energy direction to remove diffraction pattern"""
        print('Removing diffraction pattern...')

        self.data = self.data / skfilt.gaussian(self.data, sigma=(sigma, 0, 0))
        self.history['filter_diffraction'] = {'sigma': sigma}

    def high_pass_isoenergy(self, sigma=50, truncate=2.0):
        """ gaussian band pass to remove low frequency fluctuations """
        print('Applying high pass filter to each energy slice...')

        lp = skfilt.gaussian(self.data, sigma=(0, sigma, sigma), preserve_range=True, truncate=truncate)
        self.data = self.data / lp
        self.history['high_pass_isoenergy'] = {'sigma': sigma, 'truncate': truncate}

    def low_pass_isoenergy(self, sigma=(2, 2, 2), truncate=4.0):
        """ gaussian band pass to remove low frequency fluctuations """
        print('Applying low pass filter to each energy slice...')

        self.data = skfilt.gaussian(self.data, sigma=sigma, preserve_range=True, truncate=truncate)
        self.history['low_pass_isoenergy'] = {'sigma': sigma, 'truncate': truncate}

    def rotate(self,angle, axes=(1,2), reshape=False, **kwargs):
        """ Rotate the plane defined by axes, around its center."""

        self.data = rotate(self.data, angle, axes=axes, reshape=reshape,**kwargs)
        self.history['rotate'] = {'angle':angle}

    def describe_str(self, data=None, print=False):
        if data is None:
            data = self.data
        s = 'min {:9.3f} | max {:9.3f} | mean {:9.3f} | sum {:9.3f}'.format(np.amin(data),
                                                                            np.amax(data),
                                                                            np.mean(data),
                                                                            np.sum(data))
        if print:
            print(s)
        else:
            return s

    def set_mask(self, mask=None, faddr=None):
        print('loading mask...')

        if mask is not None:
            self.mask = mask
        elif faddr is not None:
            self.mask = np.load(faddr)
        self.history['mask'] = {'faddr': faddr}

    def apply_mask(self, mask=None):
        print('Applying mask...')

        if mask is None:
            mask = self.mask
        self.data = self.data * mask

    def load_grid_data(self,grid_dict):
        with open(grid_dict, 'r') as f:
            grid_dict = json.load(f)
        self.grid_reg = grid_dict['regular']
        self.grid_dist = grid_dict['distorted']
        self.bz_width = grid_dict['bz_width']
        self.k_center = grid_dict['k_center']

    def warp_grid(self,grid_dict, mask=True, ret=False,replace=True):
        """ use the given points to create a grid on which to perform perpective warping"""
        if isinstance(grid_dict,dict):
            pass
        elif isinstance(grid_dict,str):
            with open(grid_dict, 'r') as f:
                grid_dict = json.load(f)
        else:
            raise KeyError('grid_dict is neither a dictionary nor a file')
        g_reg = grid_dict['regular']
        g_dist = grid_dict['distorted']
        g_bz = grid_dict['bz_width']
        g_kc = grid_dict['k_center']
        print('Warping data...')


        # Divide the xy plane in squares and triangles defined by the simmetry points given
        squares = []
        triangles = []

        def get_square_corners(pt, dd):
            tl = pt
            tr = pt[0] + dd, pt[1]
            bl = pt[0], pt[1] + dd
            br = pt[0] + dd, pt[1] + dd
            return [tl, tr, br, bl]
        print('  - making squares...')

        for i, pt in enumerate(g_reg):
            corner_pts = get_square_corners(pt, g_bz)
            corners = []
            # ensure at least one vertex is inside the figure
            if not any([all([x[0] < 0 for x in corner_pts]),
                        all([x[0] > self.data.shape[1] for x in corner_pts]),
                        all([y[1] < 0 for y in corner_pts]),
                        all([y[1] > self.data.shape[2] for y in corner_pts])]):
                for c in corner_pts:
                    for j in range(len(g_reg)):
                        dist = point_distance(g_reg[j], c)
                        if dist < 0.1:
                            corners.append(j)
                            break

            if len(corners) == 4:
                squares.append(corners)
            elif len(corners) == 3:
                triangles.append(corners)
        # Add padding to account for areas out of selected points
        pads = []
        pads.append(int(np.round(max(0, -min([x[0] for x in g_reg])))))
        pads.append(int(np.round(max(0, max([x[0] for x in g_reg]) - self.data.shape[1]))))
        pads.append(int(np.round(max(0, -min([x[1] for x in g_reg])))))
        pads.append(int(np.round(max(0, max([x[1] for x in g_reg]) - self.data.shape[2]))))
        for i in range(4):
            if pads[i] == 0:
                pads[i] = g_bz
        xpad_l, xpad_r, ypad_l, ypad_r = pads

        warped_data_padded = np.zeros((self.data.shape[0], self.data.shape[1] + xpad_l + xpad_r, self.data.shape[2] + ypad_l + ypad_r))
        if mask:
            warped_mask_padded = np.zeros(
                (self.data.shape[0], self.data.shape[1] + xpad_l + xpad_r, self.data.shape[2] + ypad_l + ypad_r))

        print('  - calculate warp...')
        for e in tqdm(range(self.data.shape[0])):
            if mask and True not in self.mask[e,...]:
                pass
            else:
                img_pad = np.zeros((self.data.shape[1] + xpad_l + xpad_r, self.data.shape[2] + ypad_l + ypad_r))
                img_pad[xpad_l:-xpad_r, ypad_l:-ypad_r] = self.data[e, ...]
                if mask:
                    mask_pad = np.zeros((self.data.shape[1] + xpad_l + xpad_r, self.data.shape[2] + ypad_l + ypad_r))
                    mask_pad[xpad_l:-xpad_r, ypad_l:-ypad_r] = self.mask[e, ...].astype(np.float)
                for corners in squares:
                    xf, yf = g_reg[corners[0]]
                    xt, yt = g_reg[corners[2]]
                    xf += xpad_l
                    xt += xpad_l
                    yf += ypad_l
                    yt += ypad_l

                    pts1 = np.float32([(x + xpad_l, y + ypad_l) for x, y in
                                       [g_dist[x] for x in corners]])  # [pts[39],pts[41],pts[22],pts[24]]
                    pts2 = np.float32([(x + xpad_l, y + ypad_l) for x, y in [g_reg[x] for x in corners]])

                    M = cv.getPerspectiveTransform(pts1, pts2)
                    dst = cv.warpPerspective(img_pad, M, img_pad.shape[::-1])
                    if mask:
                        dst_mask = cv.warpPerspective(mask_pad, M, mask_pad.shape[::-1])

                        # print( warped_data_padded[e,yf:yt,xf:xt].shape,dst[yf:yt,xf:xt].shape)
                    try:
                        warped_data_padded[e, yf:yt, xf:xt] = dst[yf:yt, xf:xt]
                        if mask:
                            warped_mask_padded[e, yf:yt, xf:xt] = dst_mask[yf:yt, xf:xt]

                    except Exception as e:
                        print(e)
        warped_data = warped_data_padded[:, xpad_l:-xpad_r, ypad_l:-ypad_r]
        if mask:
            warped_mask = warped_mask_padded[:, xpad_l:-xpad_r, ypad_l:-ypad_r].astype(np.bool_)

        if replace:
            self.data = warped_data
            if mask:
                self.mask = warped_mask
        if ret:
            if mask:
                return warped_data,warped_mask
            else:
                return warped_data

    def save(self, file, save_raw=False, format='h5', mode='a',
             overwrite=False, chunked=True, compression='gzip'):
        """ Store data to disk.

        Allowed formats are h5, numpy and tiff
        """
        if f'.{format}' not in file:
            file += f'.{format}'
        dir = os.path.dirname(file)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        elif os.path.isfile(file) and not overwrite: # TODO: check for "mode"
            raise FileExistsError(f'File {file} already exists, set new name or allow overwriting')
        print(f'Saving processed data as "{file}"...')

        if format == 'h5':
            with h5py.File(file, mode=mode) as f:
                if chunked:
                    chunks = 1, self.data.shape[1], self.data.shape[2]
                else:
                    chunks = False
                f.create_dataset('processed/data', data=self.data, chunks=chunks, compression=compression)  # processed data
                f.create_dataset('processed/mask', data=self.mask, chunks=chunks, compression=compression)
                if save_raw:
                    f.create_dataset('raw/data', data=self.raw_data, chunks=chunks, dtype=np.uint16,
                                     compression=compression)  # raw detector data
                f.create_dataset("history", data=self.hist_str)  # metadata as string

        elif format == 'npy':
            np.save(file, self.data)

        elif format == 'tiff':
            tifffile.imsave(file, self.data, description=self.hist_str)

    def load(self, file):
        """"""
        print(f'loading data from file "{file}"...')
        with h5py.File(file, mode='r') as f:
            try:
                self.raw_data = f['raw/data'][...]
            except KeyError:
                pass
            self.data = f['processed/data'][...]
            self.mask = f['processed/mask'][...]
            self.hist_str = f['history'][()]


class DetectorDataH5(DetectorData):

    def __init__(self, raw_data=None, faddr=None, processed=None, history=None,
                 mask=None, use_file=False, compression='gzip'):
        super(DetectorData, self).__init__()

        if faddr is not None:
            folder = os.path.dirname(faddr)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            self.faddr = faddr
            self.file = h5py.File(faddr, 'a')  # TODO: lookup for some fancier options
        else:
            self.file = None

        self._use_file = use_file
        self._data = None
        self._raw_data = None
        self._mask = None

        if compression is not None:
            self._compression = compression
        else:
            self._compression = 'gzip'

        if isinstance(history, dict):  # TODO: add tracking of data imports
            self.history = history
        elif isinstance(history, str):
            self.history = json.loads(history)
        else:
            self.history = OrderedDict()

        if raw_data is not None:
            if isinstance(raw_data, np.ndarray):
                self.raw_data = raw_data.astype(np.uint16)
            elif isinstance(raw_data, str):
                if os.path.isfile(raw_data):
                    self.raw_data_address = raw_data
                    self.raw_data = tifffile.imread(raw_data).astype(np.uint16)
                else:
                    raise ValueError(f'invalid entry "{raw_data}" for Raw Data')
            self.data = np.zeros_like(self.raw_data, dtype=np.float64)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                self.raw_data = mask.astype(np.bool_)
            elif isinstance(mask, str):
                self.mask = np.load(mask).astype(np.bool)

        if processed is not None:
            self.data = processed  # .astype(np.float64)
        # elif self.raw_data is not None:
        #     self.data = self.raw_data.astype(np.float64)


    def from_disk(self):
        return self._use_file and self.file is not None

    @property
    def raw_data(self):
        if self.from_disk():
            return self.file['raw/data']
        else:
            return self._raw_data

    @raw_data.setter
    def raw_data(self, dd):
        if self.from_disk():
            try:
                self.file['raw/data'][...] = dd
            except KeyError:
                chunks = 1, dd.shape[1], dd.shape[2]
                self.file.create_dataset('raw/data', data=dd, chunks=chunks, dtype=np.uint16,
                                         compression=self._compression)
        else:
            self._raw_data = dd

    @property
    def data(self):
        if self.from_disk():
            return self.file['processed/data']
        else:
            if self._data is None:
                self._data = np.zeros_like(self._raw_data, dtype=np.float64)

    @data.setter
    def data(self, dd):
        if self.from_disk():
            try:
                self.file['processed/data'][...] = dd
            except KeyError:
                chunks = 1, dd.shape[1], dd.shape[2]
                self.file.create_dataset('processed/data', data=dd, chunks=chunks, dtype=np.float64,
                                         compression=self._compression)
        else:
            self._data = dd

    @property
    def mask(self):
        if self.from_disk():
            try:
                return self.file['raw/mask']
            except KeyError:
                self.mask = np.ones_like(self.data, dtype=np.bool_)
        else:
            return self._raw_data

    @mask.setter
    def mask(self, dd):
        if self.from_disk():
            try:
                self.file['processed/mask'][...] = dd
            except KeyError:
                chunks = 1, dd.shape[1], dd.shape[2]
                self.file.create_dataset('processed/mask', data=dd, chunks=chunks, dtype=np.bool_,
                                         compression=self._compression)
            self.file['raw/mask'][...] = dd
        else:
            self._raw_data = dd

    def make_finite(self, substitute=0.0):
        """ set all nans and infs to 0. or whatever specified value in substitute"""
        print('Handling nans and infs...')
        if self.from_disk():
            data = self.data[...]
            data[~np.isfinite(self.data)] = substitute
            self.data = data
        else:
            self.data[~np.isfinite(self.data)] = substitute
        self.history['make_finite'] = {'substitute': True}

    def __del__(self):
        # self.thefile.close()
        try:
            self.file.close()
        except:
            pass


def main():
    pass


if __name__ == '__main__':
    main()
