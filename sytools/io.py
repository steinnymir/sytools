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
import os
from zipfile import ZipFile
import tifffile

def read_zipped_tiff(key,folder):
    data = None
    for file in os.listdir(folder):
        if key in file and '.zip' in file:
            print('reading file {}'.format(file))
            filename = os.path.join(folder,file)
            with ZipFile(filename, 'r') as zipObj:
                zipObj.extractall('temp')
            tempfile = os.path.join('temp',os.path.split(filename)[1][:-4])
            if data is None:
                data = tifffile.imread(tempfile)
            else:
                data += tifffile.imread(tempfile)
            os.remove(tempfile)
    return data


def get_list_of_files(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

_imagej_metadata = """ImageJ=1.47a
    images={nr_images}
    channels={nr_channels}
    slices={nr_slices}
    hyperstack=true
    mode=color
    loop=false"""
# _imagej_metadata = {'ImageJ':'1.47a',
#                     'images':f'{nr_images}',
#                     'channels':f'{nr_channels}',
#                     'slices':f'{nr_slices}',
#                     'hyperstack':True,
#                     'mode':'color',
#                     'loop':False,
#                     }


def output_hyperstack(zs, oname):
    '''
    Write out a hyperstack to ``oname``

    Parameters
    ----------
    zs : 4D ndarray
        dimensions should be (c,z,x,y)
    oname : str
        filename to write to
    '''


    import tempfile
    import shutil
    from os import system
    import tifffile
    try:
        # We create a directory to save the results
        tmp_dir = tempfile.mkdtemp(prefix='hyperstack')

        # Channels are in first dimension
        nr_channels = zs.shape[0]
        nr_slices = zs.shape[1]
        nr_images = nr_channels*nr_slices
        # metadata = _imagej_metadata.format(
        #                 nr_images=nr_images,
        #                 nr_slices=nr_slices,
        #                 nr_channels=nr_channels)
        metadata = {'ImageJ': '1.47a',
           'images': f'{nr_images}',
           'channels': f'{nr_channels}',
           'slices': f'{nr_slices}',
           'hyperstack': True,
           'mode': 'color',
           'loop': False,
           }
        frames = []
        next = 0
        for s1 in range(zs.shape[1]):
            for s0 in range(zs.shape[0]):
                fname = '{}/s{:03}.tiff'.format(tmp_dir, next)
                # Do not forget to output the metadata!
                tifffile.imwrite(fname, zs[s0, s1], metadata=metadata)
                frames.append(fname)
                next += 1
        cmd = "tiffcp {inputs} {tmp_dir}/stacked.tiff".format(inputs=" ".join(frames), tmp_dir=tmp_dir)
        r = system(cmd)
        if r != 0:
            raise IOError('tiffcp call failed')
        shutil.copy('{tmp_dir}/stacked.tiff'.format(tmp_dir=tmp_dir), oname)
    finally:
        shutil.rmtree(tmp_dir)


def main():
    pass


if __name__ == '__main__':
    main()