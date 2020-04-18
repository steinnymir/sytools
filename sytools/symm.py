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
from numpy import average, nan_to_num
from skimage.transform import rotate as skrotate

def mirror(arr, axes=None):
    """
    Reverse array over many axes. Generalization of arr[::-1] for many dimensions.
    Parameters

    Adapted from scikit-ued: https://github.com/LaurentRDC/scikit-ued
    ----------
    arr : `~numpy.ndarray`
        Array to be reversed
    axes : int or tuple or None, optional
        Axes to be reversed. Default is to reverse all axes.

    Returns
    -------
    out : `~numpy.ndarray`
        Mirrored array.
    """
    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim

        if isinstance(axes, int):
            axes = (axes,)

        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]



def nfold(im, mod, img_plane=(0,1), center=None, mask=None, fill_value=0.0):
    """
    Returns an images averaged according to n-fold rotational symmetry. This can be used to
    boost the signal-to-noise ratio on an image with known symmetry, e.g. a diffraction pattern.
    Parameters
    ----------
    im : array_like, ndim 2
        Image to be azimuthally-symmetrized.
    center : array_like, shape (2,) or None, optional
        Coordinates of the center (in pixels). If ``center=None``, the image is rotated around
        its center, i.e. ``center=(rows / 2 - 0.5, cols / 2 - 0.5)``.
    mod : int
        Fold symmetry number. Valid numbers must be a divisor of 360.
    mask : `~numpy.ndarray` or None, optional
        Mask of `image`. The mask should evaluate to `True` (or 1) on valid pixels.
        If None (default), no mask is used.
    fill_value : float, optional
        In the case of a mask that overlaps with itself when rotationally averaged,
        the overlapping regions will be filled with this value.
    Returns
    -------
    out : `~numpy.ndarray`, dtype float
        Symmetrized image.
    Raises
    ------
    ValueError : If `mod` is not a divisor of 360 deg.
    """
    if 360 % mod:
        raise ValueError(
            f"{mod}-fold rotational symmetry is not valid (not a divisor of 360)."
        )
    angles = range(0, 360, int(360 / mod))

    ax = (5, 3)


    # Data-type must be float because of use of NaN
    im = np.array(im, dtype=np.float, copy=True)

    # sort axis to have image plane "in front"
    im = im.swapaxes(0, img_plane[0]).swapaxes(1, img_plane[1])
    if mask is not None:
        im[np.logical_not(mask)] = np.nan

    kwargs = {"center": center, "mode": "constant", "cval": 0, "preserve_range": True}

    # Use weights because edges of the pictures, which might be cropped by the rotation
    # should not count in the average
    wt = np.ones_like(im, dtype=np.uint8)
    weights = (skrotate(wt, angle, **kwargs) for angle in angles)
    rotated = (skrotate(im, angle, **kwargs) for angle in angles)

    avg = average(rotated, weights=weights)#, ignore_nan=True)
    # sort axis back to original positions
    avg = avg.swapaxes(img_plane[1],1).swapaxes(img_plane[0],0)
    return nan_to_num(avg, fill_value, copy=False)


def reflection(im, angle, img_plane=(0,1), center=None, mask=None, fill_value=0.0):
    """
    Symmetrize an image according to a reflection plane.
    Parameters
    ----------
    im : array_like, ndim 2
        Image to be symmetrized.
    angle : float
        Angle (in degrees) of the line that defines the reflection plane. This angle
        increases counter-clockwise from the positive x-axis. Angles
        larger that 360 are mapped back to [0, 360). Note that ``angle`` and ``angle + 180``
        are equivalent.
    center : array_like, shape (2,) or None, optional
        Coordinates of the center (in pixels). If ``center=None``, the image is rotated around
        its center, i.e. ``center=(rows / 2 - 0.5, cols / 2 - 0.5)``.
    mask : `~numpy.ndarray` or None, optional
        Mask of `image`. The mask should evaluate to `True` (or 1) on valid pixels.
        If None (default), no mask is used.
    fill_value : float, optional
        In the case of a mask that overlaps with itself when rotationally averaged,
        the overlapping regions will be filled with this value.
    Returns
    -------
    out : `~numpy.ndarray`, dtype float
        Symmetrized image.
    """
    angle = float(angle) % 360

    # Data-type must be float because of use of NaN
    im = np.array(im, dtype=np.float, copy=True)
    # sort axis to have image plane "in front"
    im = im.swapaxes(0, img_plane[0]).swapaxes(1, img_plane[1])
    reflected = np.array(im, copy=True)  # reflected image

    if mask is not None:
        invalid_pixels = np.logical_not(mask)
        im[invalid_pixels] = np.nan
        reflected[invalid_pixels] = np.nan

    kwargs = {"center": center, "mode": "constant", "cval": 0, "preserve_range": True}

    # Rotate the 'reflected' image so that the reflection line is the x-axis
    # Flip the image along the y-axis
    # Rotate back to original orientation
    # FIXME: this will not work properly for images that are offcenter
    # print(type(reflected))


    reflected = skrotate(reflected, -angle, **kwargs)
    reflected = mirror(reflected, axes=0)
    reflected = skrotate(reflected, angle, **kwargs)

    out = nan_to_num(average([im, reflected]), fill_value, copy=False)
    out = out.swapaxes(img_plane[1],1).swapaxes(img_plane[0],0)
    return out


def main():
    pass


if __name__ == '__main__':
    main()
