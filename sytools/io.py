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


def main():
    pass


if __name__ == '__main__':
    main()