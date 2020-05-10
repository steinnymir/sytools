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
import sys, os
import gc
import time
import psutil
from sytools.pes import *
from sytools.misc import repr_byte_size
sys.path.append('D:/code/')
import mpes.mpes.fprocessing as fp


force_preprocess = False # force repeating raw data processing
force_parquet = False
# Input Data
raw_data = 'E:/data/YbRh2Si2/eval_1/combined_LT_VB.tif'
mask = 'E:/data/YbRh2Si2/mask_full.npy'
grid_dict = 'd:/data/YbRh2Si2/warp_grids_padded.txt'

# detector corrections
ToF_blur = 15   # smoothing along tof to remove diffraction
sigma_high = 50 # gaussian blur sigma for high pass filter in XY plane
sigma_low = None   # gaussian blur sigma for high pass filter in XY plane
rotate_deg = -3 # rotation in XY plane to align kx and ky t X and Y
warp = True

# material parameters for momentum reconstruction
a, b, c = 4.01, 4.01, 9.841 # lattice parameters in Angstrom
ka, kb, kc = (2 * np.pi / a, 2 * np.pi / b, 2 * np.pi / c) # reciprocal lattice vectors
k_center = 719, 181  # center of momentum space on the detector, in pixels, as row,col => y, x
k_final = 27.285 # kz value corresponding to the photon energy
aoi_px = (220, 220)  # X and Y dimensions of the reciprocal unit cell repeated in the measurement. in pixels row,col => y, x
aoi_k = (np.sqrt(ka ** 2 + kb ** 2),np.sqrt(ka ** 2 + kb ** 2)) # kx and ky dimension of the reciprocal unit cell used.
tof_to_ev = -0.04
tof_0 = 90

# files and directories
savename = 'LT'
h5_dir = 'D:/data/YbRh2Si2/processed/'
parquet_dir = 'D:/data/YbRh2Si2/parquet/'
binned_dir = 'D:/data/YbRh2Si2/binned/'

# binning parameters:
ncores = 'auto'
axes = ['e', 'kx', 'ky', 'kz']
bins = [100, 100, 100, 50]
ranges = [(-4.3,.5), (-aoi_k[0]/2, aoi_k[0]/2), (-aoi_k[1]/2, aoi_k[1]/2), (-kc/2, kc/2)]
jitter_amplitude = [.04, 0.2, 0.2, 0.2]

#%% generate file name
print(f'Expected binned array size: {repr_byte_size(np.prod(bins).astype(np.int64)*64)}\nncores set to {ncores}')

if ToF_blur is not None:
    savename += f'_e{ToF_blur}'
if sigma_high is not None:
    savename += f'_h{sigma_high}'
if sigma_low is not None:
    savename += f'_l{sigma_low}'
if rotate_deg is not None:
    savename += f'_rot{rotate_deg}'
if warp is not None:
    savename += '_warp'

#%% Detector correction
print(f'\n looking for processing file name: {savename}....\n')

if os.path.isfile(f'{h5_dir}{savename}.h5') and not force_preprocess: # if file with same parameters exists, do not recalculate
    print('...found!\n')
    dd = None
else:
    print('...not found!\n')
    print(f'Loading raw data from {raw_data}...')
    t0 = time.time()
    dd = DldProcessor(raw_data=raw_data, mask=mask)
    dd.k_center = k_center
    dd.k_final = k_final
    dd.unit_cell = (a, b, c)
    dd.aoi_px = aoi_px
    dd.aoi_k = aoi_k
    dd.tof_to_ev = tof_to_ev
    dd.tof_0 = tof_0

    if ToF_blur is not None:
        dd.filter_diffraction(sigma=ToF_blur)
        dd.make_finite()

    if sigma_high is not None:
        dd.high_pass_isoenergy(sigma=sigma_high)
        dd.make_finite()

    if sigma_low is not None:
        dd.low_pass_isoenergy(sigma=(0,sigma_low,sigma_low))
        dd.make_finite()

    if rotate_deg is not None:
        dd.rotate(rotate_deg)
        dd.make_finite()

    if warp is not None:
        dd.warp_grid(grid_dict=grid_dict, replace=True, mask=True)
        dd.make_finite()

    dd.compute_energy_momentum()

    dd.save(f'{h5_dir}{savename}.h5', overwrite=False)
    print(f'processed raw data in {time.time()-t0:.2f} s')

#%% Generate dataframe
if not os.path.isdir(f'{parquet_dir}{savename}/') or force_parquet:
    if dd is None:
        print(f'Loading processed data from {h5_dir}{savename}.h5')
        dd = DldProcessor(h5_file=f'{h5_dir}{savename}.h5')
    print(f'\n - Creating dataframe... \n')
    dd.create_dataframe()
    print(f'Saving dataframe as parquet at {parquet_dir}{savename}...')
    dd.to_parquet(f'{parquet_dir}{savename}/', cols=['kx', 'ky', 'kz', 'e', 'processed'])
    del dd
gc.collect()
#%% binning
print(f'\n - Loading parquet data from: {parquet_dir}{savename}....\n')
if ncores == 'auto':
    m = psutil.virtual_memory()
    ncores = int(m[1]*.75/(np.prod(bins).astype(np.int64)*64))
    print(f'Binning using {ncores} cores. Memory usage: {np.prod(bins).astype(np.int64)*64:,.0f}')
    if ncores < 1:
        raise MemoryError(f'Binning too large for current memory:\nBinnned array size: {repr_byte_size(np.prod(bins).astype(np.int64)*64)}\nMemory Status: {m}')

dfp = fp.dataframeProcessor(datafolder=f'{parquet_dir}{savename}', ncores=ncores)
dfp.read(source='folder', ftype='parquet')

coords = {}
for a, b, r in zip(axes, bins, ranges):
    coords[a] = np.linspace(r[0], r[1], b)
dfp.distributedBinning(axes=axes, nbins=bins, ranges=ranges, scheduler='threads', ret=False, jittered=True,
                       jitter_amplitude=jitter_amplitude, pbenv='classic', weight_axis='processed', binmethod='original')

#%% Post processing

print(f'\n - Genrating Band structure:\n')

bs = BandStructure(dfp.histdict['binned'], coords=coords, dims=axes)
translation_vector = {'kx':int(len(bs.kx)//2),'ky':int(len(bs.ky)//2),'kz':int(len(bs.kz)//2)}
print(f'Symmetrizing...')
bs.symmetrize_mirror(('kx','ky','kz'), update=True, ret=False)
bs.symmetrize_translation(shift_dict=translation_vector, update=True, ret=False)
# bs.symmetrize_rotation(rot_plane=('kx','ky'),method='xr',order=4, update=True, ret=False)
bs.interpolate(('kx','ky','kz'))
bs.symmetrize_mirror(('kx','ky'), update=True, ret=False)
# bs.symmetrize_rotation(rot_plane=('ky','kx'),method='xr',order=4, update=True, ret=False)
bs.symmetrize_translation(shift_dict=translation_vector, update=True, ret=False)
# bs.symmetrize_mirror(('kz','ky','kx'), update=True, ret=False)
bs.data = np.nan_to_num(bs.data)
bs.blur({'kx':2,'ky':2,'kz':1,'e':1})

if os.path.isfile(f'{binned_dir}{savename}_symm.h5'):
    os.remove(f'{binned_dir}{savename}_symm.h5')
bs.save_h5(f'{binned_dir}{savename}_symm.h5')
print(f'Saved symmetrized band structure as {binned_dir}{savename}_symm.h5')

#bs.export_tiff(f'{binned_dir}{savename}_symm.tif')
