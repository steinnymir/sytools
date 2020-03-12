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
import psutil

def get_system_memory_status(print_=False):
    mem_labels = ('total', 'available', 'percent', 'used', 'free')
    mem = psutil.virtual_memory()
    memstatus = {}
    for i, val in enumerate(mem):
        memstatus[mem_labels[i]] = val
    if print_:
        for key, value in memstatus.items():
            if key == 'percent':
                print('{}: {:0.3}%'.format(key, value))
            else:
                print('{}: {:0,.4} GB'.format(key, value / 2 ** 30))
    return memstatus



def main():
    pass


if __name__ == '__main__':
    main()