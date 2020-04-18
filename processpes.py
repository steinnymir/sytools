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
import time
import numpy as np
import sys,os
from sytools.pes.dld import DetectorData

def main():
    t = [time.time()]
    dd = DetectorData(faddr='c:/data/YbRh2Si2/LT_test.h5',
                      raw_data='E:/data/YbRh2Si2/eval_1/combined_RT_VB.tif',
                      use_file=True,compression=False)
    t.append(time.time())
    dd.filter_diffraction(sigma=15)
    t.append(time.time())
    dd.make_finite()
    t.append(time.time())
    # dd.high_pass_isoenergy(sigma=50)
    # t.append(time.time())
    # dd.make_finite()
    # t.append(time.time())
    # dd.low_pass_isoenergy(sigma=(2, 2, 2))
    # t.append(time.time())
    m = np.load('E:/data/YbRh2Si2/mask_82-200.npy')
    mask = np.zeros_like(dd.data)
    mask[82:200] = m
    dd.mask = mask
    t.append(time.time())
    dd.make_finite()
    t.append(time.time())
    dd.renormalize_DOS(use_mask=True)
    t.append(time.time())
    dd.save('E:/data/YbRh2Si2/eval_2/RT_selection',save_raw=True,overwrite=True)
    print(f'elapsed time: {t[-1]-t[0]:.2f}s')
    s = ''
    for dt in [t1 - t0 for t0, t1 in zip(t[:-1], t[1:])]:
        s += f'{dt:.2f} s | '
    print(s)


def run(faddr, s_e, s_h, s_l):
    t0 = time.time()
    name = f'LT_e{s_e}_h{s_h}_l{s_l}'
    print(f'\n\n - Run: {name}\n\n')
    dd = DetectorData(faddr=faddr)
    dd.filter_diffraction(sigma=s_e)
    dd.make_finite()
    dd.high_pass_isoenergy(sigma=s_h)
    dd.make_finite()
    dd.low_pass_isoenergy(sigma=(s_l, s_l, s_l))
    m = np.load('E:/data/YbRh2Si2/mask_82-200.npy')
    mask = np.zeros_like(dd.data)
    mask[82:200] = m
    dd.set_mask(mask)
    dd.make_finite()
    dd.renormalize_DOS(use_mask=True)
    dd.save('E:/data/YbRh2Si2/eval_2/' + name)
    print(f'run duration: {time.time() - t0:.2f}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    dd= DetectorData(faddr='C:/data/YbRh2Si2/LT_e15_h50_l2_rot-3.h5')
    plt.imshow(dd.data[122,...],clim=(.9,1.4))
    plt.show()
    warped,mask = dd.warp_grid(grid_dict='d:/data/YbRh2Si2/warp_grids_padded.txt',ret=True,replace=False, mask=True)

    plt.imshow(warped[122,...],clim=(.9,1.4))
    plt.show()
    plt.imshow(warped[122,...]*mask[122,...],clim=(.9,1.4))
    plt.show()
    # dd.rotate(-3)
    # dd.high_pass_isoenergy(sigma=50)
    # dd.make_finite()
    # dd.low_pass_isoenergy(sigma=(2, 2, 2))
    # dd.mask = np.load('E:/data/YbRh2Si2/mask_full.npy')
    # dd.make_finite()
    # # dd.renormalize_DOS(use_mask=False)
    # # dd.make_finite()
    # dd.save('C:/data/YbRh2Si2/LT_e15_h50_l2_rot-3.h5',save_raw=True)
    # dd_rt = DetectorData(raw_data='E:/data/YbRh2Si2/eval_1/combined_RT_VB.tif')
    # dd_lt = DetectorData(raw_data='E:/data/YbRh2Si2/eval_1/combined_LT_VB.tif')
    # dd_rt.mask = np.load('E:/data/YbRh2Si2/mask_full.npy')
    # dd_lt.mask = np.load('E:/data/YbRh2Si2/mask_full.npy')
    # s_e = [1,2,3,4,5,6,7,8,9,31,32,33,34,35,36,37,38,39,40]
    # for s in s_e:
    #
    #     name = f'e{s}_h{50}_l{2}'
    #
    #     print(f'\nRUN {s-9}: name: {name}\n')
    #     dd_rt.filter_diffraction(sigma=s)
    #     dd_rt.make_finite()
    #     dd_rt.save('E:/data/YbRh2Si2/zblurs/' + 'RT_' + name,overwrite=True)
    #     dd_rt.reset_data()
    #     dd_rt.reset_history()
    #
    #     dd_lt.filter_diffraction(sigma=s)
    #     dd_lt.make_finite()
    #     dd_lt.save('E:/data/YbRh2Si2/zblurs/' + 'LT_' + name,overwrite=True)
    #     dd_lt.reset_data()
    #     dd_rt.reset_history()
