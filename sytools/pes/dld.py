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

import cv2 as cv
import h5py
import skimage.filters as skfilt
import tifffile
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.ndimage import rotate
from tqdm import tqdm

from .utils import *


def main():
    pass


class DldProcessor(object):
    PARAMETERS = ['k_center', 'k_final',
                  'unit_cell', 'aoi_px', 'aoi_k',
                  'warp_grid_reg', 'warp_grid_dist', 'warp_grid_spacing',
                  'tof_to_ev', 'tof_0', 'h5_file', 'raw_file','mask_file',
                  'sigma_tof_blur', 'sigma_highpass', 'sigma_lowpass',
                  'rotation_angle', 'rotation_center',
                  ]

    def __init__(self, h5_file=None, raw_data=None, processed=None, mask=None, chunks=None):

        # data containers
        self.processed = None
        self.raw = None
        self.mask = None

        # disk file properties
        self.h5_file = None
        self.raw_file = None
        self.mask_file = None
        self.chunks = chunks

        self.k_center = None, None  # as row,col => y, x
        self.k_final = None
        self.unit_cell = None, None, None
        self.aoi_px = None, None
        self.aoi_k = None, None
        self.warp_grid_reg = None
        self.warp_grid_dist = None
        self.warp_grid_spacing = None
        self.tof_to_ev = None
        self.tof_0 = None

        # Processing parameters
        self.sigma_tof_blur = None
        self.sigma_highpass = None
        self.sigma_lowpass = None
        self.rotation_angle = None
        self.rotation_center = None

        self._dos = None

        self.history = ''

        if h5_file is not None:
            self.load_h5(h5_file)
            self.h5_file = h5_file
            self.update_history('load_h5', f'faddr:{h5_file}')

        if raw_data is not None:
            if isinstance(raw_data, np.ndarray):
                data = raw_data.astype(np.uint16)

            elif isinstance(raw_data, str):
                if os.path.isfile(raw_data):
                    self.raw_file = raw_data
                    data = tifffile.imread(raw_data).astype(np.uint16)
                else:
                    raise ValueError(f'invalid entry "{raw_data}" for Raw Data')
            coords = {'ToF': np.arange(data.shape[0]),
                      'X': np.arange(data.shape[1]),
                      'Y': np.arange(data.shape[2]), }
            self.raw = xr.DataArray(data=data, coords=coords, dims=['ToF', 'X', 'Y'])

            if self.processed is None:
                self.processed = xr.DataArray(data=self.raw.values.astype(np.float64, copy=True), coords=coords,
                                              dims=['ToF', 'X', 'Y'])

        if mask is not None:
            self.load_mask(mask)

        if processed is not None:
            self.processed = processed


    @property
    def metadata_dict(self):
        d = {}
        for par in self.PARAMETERS:
            d[par] = getattr(self, par)
        return d
        # d = {'bz_width': self.bz_width,
        #      'k_center': self.k_center,
        #      'k_final': self.k_final,
        #      'unit_cell': self.unit_cell,
        #      'aoi_px': self.aoi_px,
        #      'aoi_k': self.aoi_k,
        #      'warp_grid_reg': self.warp_grid_reg,
        #      'warp_grid_dist': self.warp_grid_dist,
        #      'warp_grid_spacing': self.warp_grid_spacing,
        #      }

    @property
    def dos(self):
        if self._dos is not None:
            return self._dos
        else:
            self._dos = self.raw.sum(dim=('X', 'Y'))
            return self._dos

    @property
    def dos_masked(self):
        if self._mdos is not None:
            return self._mdos
        else:
            self._mdos = self.raw.where(self.mask).sum(dim=('X', 'Y'))
            return self._mdos

    @property
    def reciprocal_unit_cell(self):
        return tuple([2 * np.pi / x for x in self.unit_cell])

    @property
    def masked_data(self):
        return self.processed * self.mask

    def update_history(self, method, attributes):
        self.history += f'{method}:{attributes}\n'

    def reset_data(self):
        self.processed.values = self.raw.astype(np.float64, copy=True)

    def reset_history(self):
        self.history = ''

    def chunk(self, chunks=None):
        if chunks is not None:
            self.chunks = chunks
        if self.chunks is None:
            self.chunks = {'ToF': 1, 'X': 128, 'Y': 128}

        self.processed = self.processed.chunk(chunks=self.chunks)
        self.raw = self.raw.chunk(chunks=self.chunks)
        self.mask = self.mask.chunk(chunks=self.chunks)
        self.update_history('chunk', f'chunks:{chunks}')

    def renormalize_DOS(self, use_mask=True):
        """ normalize the """
        print('Renormalizing DOS and applying mask...')
        if use_mask and self.mask is not None:
            data = self.processed * self.mask
            raw_data = self.raw * self.mask
        else:
            data = self.processed
            raw_data = self.raw

        norm = data.sum((1, 2)) / raw_data.sum((1, 2)).astype(np.float64)
        self.processed.values = data / norm[:, None, None]
        self.update_history('renormalize_DOS', f'use_mask:{use_mask}')

    def make_finite(self, substitute=0.0):
        """ set all nans and infs to 0. or whatever specified value in substitute"""
        print('Handling nans and infs...')
        self.processed.values[~np.isfinite(self.processed.values)] = substitute
        self.update_history('make_finite','substitute:{substitute}')

    def filter_diffraction(self, sigma=None):
        """ divide by self, gaussian blurred along energy direction to remove diffraction pattern"""
        print('Removing diffraction pattern...')
        if sigma is not None:
            self.sigma_tof_blur = sigma
        self.processed = self.processed / skfilt.gaussian(self.processed, sigma=(self.sigma_tof_blur, 0, 0))
        self.update_history('filter_diffraction', f'sigma:{self.sigma_tof_blur}')

    def high_pass_isoenergy(self, sigma=None, truncate=2.0):
        """ gaussian band pass to remove low frequency fluctuations """
        print('Applying high pass filter to each energy slice...')
        if sigma is not None:
            self.sigma_highpass = sigma
        lp = skfilt.gaussian(self.processed, sigma=(0, self.sigma_highpass, self.sigma_highpass), preserve_range=True,
                             truncate=truncate)
        self.processed = self.processed - lp
        self.update_history('high_pass_isoenergy', f'sigma:{self.sigma_highpass}, truncate:{truncate}')

    def low_pass_isoenergy(self, sigma=(2, 2, 2), truncate=4.0):
        """ gaussian band pass to remove low frequency fluctuations """
        print('Applying low pass filter to each energy slice...')
        if sigma is not None:
            self.sigma_lowpass = sigma
        self.processed.values = skfilt.gaussian(self.processed, sigma=self.sigma_lowpass, preserve_range=True,
                                                truncate=truncate)
        self.update_history('low_pass_isoenergy', f'sigma:{self.sigma_lowpass}, truncate:{truncate}')

    def rotate(self, angle, axes=(1, 2), center=None, **kwargs):
        """ Rotate the plane defined by axes, around its center."""
        if angle is not None:
            self.rotation_angle = angle
        if center is not None:
            self.rotation_center = center  # TODO: implement off center rotation
        self.processed.values = rotate(self.processed, angle, reshape=False, axes=axes, **kwargs)
        self.mask.values = rotate(self.mask, angle, reshape=False, axes=axes, **kwargs)
        hist_str = f'angle:{self.rotation_angle}, center:{self.rotation_center}'
        for k, v in kwargs.items():
            hist_str += f', {k}:{v}'
        self.update_history('rotate', hist_str)

    def describe_str(self, data=None, print=False):
        if data is None:
            data = self.processed
        s = 'min {:9.3f} | max {:9.3f} | mean {:9.3f} | sum {:9.3f}'.format(np.amin(data),
                                                                            np.amax(data),
                                                                            np.mean(data),
                                                                            np.sum(data))
        if print:
            print(s)
        else:
            return s

    def load_mask(self, mask=None):
        print('loading mask...')
        if isinstance(mask, xr.DataArray):
            self.mask = mask
        else:
            coords, dims = None, None
            if isinstance(mask, str):
                self.mask_file = mask
                if '.np' in mask:
                    mask = np.load(mask)

                elif '.h5' in mask:
                    with h5py.File(mask, 'r') as f:
                        mask = f['mask/data'][...]
                        try:
                            coords = {}
                            for key in f['mask/axes']:
                                coords[key] = f[f'mask/axes/{key}']
                            dims = [x for x in ['ToF', 'X', 'Y'] if x in coords]
                        except KeyError:
                            pass

            if coords is None and dims is None:
                coords = {'ToF': np.arange(0, mask.shape[0]),
                          'X': np.arange(0, mask.shape[1]),
                          'Y': np.arange(0, mask.shape[2])}
                dims = ['ToF', 'X', 'Y']
            self.mask = xr.DataArray(data=mask.astype(np.bool_), coords=coords, dims=dims)

    def warp_grid(self, grid_dict, mask=True, ret=False, replace=True):
        """ use the given points to create a grid on which to perform perpective warping"""
        if isinstance(grid_dict, dict):
            pass
        elif isinstance(grid_dict, str):
            with open(grid_dict, 'r') as f:
                grid_dict = json.load(f)
        elif None in [self.k_center, self.warp_grid_spacing, self.warp_grid_dist, self.warp_grid_reg]:
            pass
        else:
            raise KeyError('grid_dict is neither a dictionary nor a file')
        self.warp_grid_reg = grid_dict['regular']
        self.warp_grid_dist = grid_dict['distorted']
        self.k_center = grid_dict['k_center']
        self.warp_grid_spacing = grid_dict['spacing']

        hist_str = f'k_center:{self.k_center}, ' + \
                   f'warp_grid_spacing:{self.warp_grid_spacing}, ' + \
                   f'warp_grid_reg:{self.rotation_angle}, ' + \
                   f'warp_grid_dist:{self.rotation_center}'

        self.update_history('warp_grid', hist_str)

        print('Warping data...')

        # Divide the xy plane in squares and triangles defined by the simmetry points given
        # At the moment, only the squares are being used.
        squares = []
        triangles = []

        def get_square_corners(pt, dd):
            tl = pt
            tr = pt[0] + dd, pt[1]
            bl = pt[0], pt[1] + dd
            br = pt[0] + dd, pt[1] + dd
            return [tl, tr, br, bl]

        print('  - making squares...')

        for i, pt in enumerate(self.warp_grid_reg):
            corner_pts = get_square_corners(pt, self.warp_grid_spacing)
            corners = []
            # ensure at least one vertex is inside the figure
            if not any([all([x[0] < 0 for x in corner_pts]),
                        all([x[0] > self.processed.shape[1] for x in corner_pts]),
                        all([y[1] < 0 for y in corner_pts]),
                        all([y[1] > self.processed.shape[2] for y in corner_pts])]):
                for c in corner_pts:
                    for j in range(len(self.warp_grid_reg)):
                        dist = point_distance(self.warp_grid_reg[j], c)
                        if dist < 0.1:
                            corners.append(j)
                            break

            if len(corners) == 4:
                squares.append(corners)
            elif len(corners) == 3:
                triangles.append(corners)
        # Add padding to account for areas out of selected points
        pads = []
        pads.append(int(np.round(max(0, -min([x[0] for x in self.warp_grid_reg])))))
        pads.append(int(np.round(max(0, max([x[0] for x in self.warp_grid_reg]) - self.processed.shape[1]))))
        pads.append(int(np.round(max(0, -min([x[1] for x in self.warp_grid_reg])))))
        pads.append(int(np.round(max(0, max([x[1] for x in self.warp_grid_reg]) - self.processed.shape[2]))))
        for i in range(4):
            if pads[i] == 0:
                pads[i] = self.warp_grid_spacing
        xpad_l, xpad_r, ypad_l, ypad_r = pads

        warped_data_padded = np.zeros(
            (self.processed.shape[0], self.processed.shape[1] + xpad_l + xpad_r,
             self.processed.shape[2] + ypad_l + ypad_r))
        if mask:
            warped_mask_padded = np.zeros(
                (self.processed.shape[0], self.processed.shape[1] + xpad_l + xpad_r,
                 self.processed.shape[2] + ypad_l + ypad_r))

        print('  - calculate warp...')
        for e in tqdm(range(self.processed.shape[0])):
            # if mask and True not in self.mask[e, ...]:
            if mask and True not in self.mask[e, ...].values:
                pass
            else:
                img_pad = np.zeros(
                    (self.processed.shape[1] + xpad_l + xpad_r, self.processed.shape[2] + ypad_l + ypad_r))
                img_pad[xpad_l:-xpad_r, ypad_l:-ypad_r] = self.processed[e, ...]
                if mask:
                    mask_pad = np.zeros(
                        (self.processed.shape[1] + xpad_l + xpad_r, self.processed.shape[2] + ypad_l + ypad_r))
                    mask_pad[xpad_l:-xpad_r, ypad_l:-ypad_r] = self.mask[e, ...].astype(np.float)
                for corners in squares:
                    xf, yf = self.warp_grid_reg[corners[0]]
                    xt, yt = self.warp_grid_reg[corners[2]]
                    xf += xpad_l
                    xt += xpad_l
                    yf += ypad_l
                    yt += ypad_l

                    pts1 = np.float32([(x + xpad_l, y + ypad_l) for x, y in
                                       [self.warp_grid_dist[x] for x in corners]])  # [pts[39],pts[41],pts[22],pts[24]]
                    pts2 = np.float32([(x + xpad_l, y + ypad_l) for x, y in [self.warp_grid_reg[x] for x in corners]])

                    M = cv.getPerspectiveTransform(pts1, pts2)
                    dst = cv.warpPerspective(img_pad, M, img_pad.shape[::-1])
                    if mask:
                        dst_mask = cv.warpPerspective(mask_pad, M, mask_pad.shape[::-1])

                        # print( warped_data_padded[e,yf:yt,xf:xt].shape,dst[yf:yt,xf:xt].shape)
                    try:
                        warped_data_padded[e, yf:yt, xf:xt] = dst[yf:yt, xf:xt]
                        if mask:
                            warped_mask_padded[e, yf:yt, xf:xt] = dst_mask[yf:yt, xf:xt]

                    except Exception as ex:
                        print(ex)
        warped_data = warped_data_padded[:, xpad_l:-xpad_r, ypad_l:-ypad_r]
        if mask:
            warped_mask = warped_mask_padded[:, xpad_l:-xpad_r, ypad_l:-ypad_r].astype(np.bool_)

        if replace:
            self.processed.values = warped_data
            if mask:
                self.mask.values = warped_mask
        if ret:
            if mask:
                return warped_data, warped_mask
            else:
                return warped_data

    def create_dataframe(self, data='processed', masked=True, chunks={'ToF': 1, 'X': 128, 'Y': 128}):
        da = getattr(self, data)
        da.name = data
        if da.chunks is None:
            self.chunk(chunks)

        if masked:
            da = da.where(self.mask, other=0.0)
        self.df = da.to_dataset().to_dask_dataframe().dropna(subset=[data])
        self.df = self.df[self.df['processed'] != 0]

    def to_parquet(self, file, cols=None):
        if cols is None:
            df = self.df
        else:
            df = self.df[cols]
        with ProgressBar():
            df.to_parquet(file)

    def compute_energy_momentum(self, k_center=None, aoi_px=None, aoi_k=None, tof_to_ev=None, tof_0=None):
        # TODO: generalize for arbitrary unit cell and energy momentum conversion parameters
        if k_center is not None:
            self.k_center = k_center
        if aoi_px is not None:
            self.aoi_px = aoi_px
        if aoi_k is not None:
            self.aoi_k = aoi_k
        if tof_to_ev is not None:
            self.tof_to_ev = tof_to_ev
        if tof_0 is not None:
            self.tof_0 = tof_0

        kx = to_reduced_scheme(
            to_k_parallel(self.processed.X, self.k_center[1], aoi_px=self.aoi_px[1], aoi_k=self.aoi_k[1]),
            aoi_k=self.aoi_k[1])
        ky = to_reduced_scheme(
            to_k_parallel(self.processed.Y, self.k_center[0], aoi_px=self.aoi_px[0], aoi_k=self.aoi_k[0]),
            aoi_k=self.aoi_k[0])
        kz = to_reduced_scheme(to_k_perpendicular((self.processed.Y, self.processed.X), self.k_center,
                                                  kf=self.k_final, aoi_px=np.mean(self.aoi_px),
                                                  aoi_k=np.mean(self.aoi_k) - self.reciprocal_unit_cell[2] / 2),
                               self.reciprocal_unit_cell[2])
        e = slice_to_ev(self.processed.ToF, ToF_to_ev=self.tof_to_ev, t0=self.tof_0)
        self.processed = self.processed.assign_coords({'kx': kx, 'ky': ky, 'kz': kz, 'e': e})
        self.mask = self.mask.assign_coords({'kx': kx, 'ky': ky, 'kz': kz, 'e': e})

        hist_str = f'k_center:{self.k_center}, ' + \
                   f'aoi_px:{self.aoi_px}, ' + \
                   f'aoi_k:{self.aoi_k}, ' + \
                   f'tof_to_ev:{self.tof_to_ev}, ' + \
                   f'tof_0:{self.tof_0}'
        self.update_history('compute_energy_momentum', hist_str)

    def save(self, file, save_raw=True, format='h5', mode='a',
             overwrite=False, chunks='auto', compression='gzip'):
        """ Store data to disk.

        Allowed formats are h5, numpy and tiff
        """
        if f'.{format}' not in file:
            file += f'.{format}'
        dir = os.path.dirname(file)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        elif os.path.isfile(file):
            if not overwrite:  # TODO: check for "mode"
                raise FileExistsError(f'File {file} already exists, set new name or allow overwriting')
            else:
                os.remove(file)
        print(f'Saving processed data as "{file}"...')

        if format == 'h5':
            with h5py.File(file, mode=mode) as f:
                errors = []
                if self.processed.chunks is not None:
                    # TODO: auto define chunks size from xarray chunks
                    pass
                if chunks == 'auto':

                    chunks = 1, self.processed.shape[1] // 16, self.processed.shape[2] // 16
                elif chunks and self.chunks is not None:
                    chunks = [self.chunks[k] for k in self.processed.dims]

                f.create_dataset('processed/data', data=self.processed, chunks=chunks, compression=compression)
                f.create_dataset('processed/axes/ToF', data=self.processed.ToF, compression=compression)
                f.create_dataset('processed/axes/X', data=self.processed.X, compression=compression)
                f.create_dataset('processed/axes/Y', data=self.processed.Y, compression=compression)

                f.create_dataset('mask/data', data=self.mask, chunks=chunks, compression=compression)
                f.create_dataset('mask/axes/ToF', data=self.mask.ToF, compression=compression)
                f.create_dataset('mask/axes/X', data=self.mask.X, compression=compression)
                f.create_dataset('mask/axes/Y', data=self.mask.Y, compression=compression)

                if save_raw:
                    f.create_dataset('raw/data', data=self.raw, chunks=chunks, dtype=np.uint16,
                                     compression=compression)  # raw detector data
                    f.create_dataset('raw/axes/ToF', data=self.raw.ToF, compression=compression)
                    f.create_dataset('raw/axes/X', data=self.raw.X, compression=compression)
                    f.create_dataset('raw/axes/Y', data=self.raw.Y, compression=compression)

                for par in self.PARAMETERS:
                    v = getattr(self, par)
                    if v is not None:
                        try:
                            f.create_dataset(f'metadata/{par}', data=v)
                        except Exception as e:
                            errors.append((par, v, e))
                if len(errors) > 0:
                    for par, v, e in errors:
                        print(f'Failed writing {par} = {v}. Error: {e}')

                f.create_dataset("history", data=self.history)  # metadata as string

        elif format == 'npy':
            np.save(file, self.processed)

        elif format == 'tiff':
            tifffile.imsave(file, self.processed, description=self.hist_str)

    def load_h5(self, file, read_groups=None):
        """"""
        print(f'loading data from "{file}".')
        with h5py.File(file, mode='r') as f:
            groups = f.keys()
            if read_groups is None:
                read_groups = groups
            print(f'Found {len(groups)} groups: {groups}\nLoading:')
            for group in f.keys():
                if group not in read_groups:
                    print(f'  - {group} ignored')
                else:
                    print(f'  - {group}...')
                    if group == 'history':
                        self.hist_str = f['history'][()]
                    if group == 'metadata':
                        for key, value in f[f'{group}'].items():
                            v = value[...]
                            if getattr(self,key) is None: #TODO: improve metatadata reading
                                try:
                                    v = float(v)
                                    if v%1 == 0:
                                        v = int(v)
                                except ValueError:
                                    v = str(v)
                                except TypeError:
                                    pass
                            else:
                                v = tuple(v)
                            setattr(self, key, v)
                    elif group in ['raw','processed','mask']:
                        data = f[f'{group}/data'][...]
                        coords = {}
                        try:
                            for key in f[f'{group}/axes']:
                                coords[key] = f[f'{group}/axes/{key}']
                            dims = [x for x in ['ToF', 'X', 'Y'] if x in coords]
                        except KeyError:
                            coords = {'ToF': np.arange(0, data.shape[0]),
                                      'X': np.arange(0, data.shape[1]),
                                      'Y': np.arange(0, data.shape[2])}
                            dims = ['ToF', 'X', 'Y']
                        setattr(self, group, xr.DataArray(data=data, coords=coords, dims=dims, name=group))
        # self.update_history('load_h5', f'file:{file}, read_groups:{read_groups}')


def main():
    pass


if __name__ == '__main__':
    main()
