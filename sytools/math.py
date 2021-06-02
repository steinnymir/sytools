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
from scipy.special import erf


def monotonically_increasing(l):
    return all(x < y for x, y in zip(l, l[1:]))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_fwhm(x, A, x0, fwhm, c):
    sig = fwhm * 2 / 2.355
    return A * np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.))) + c


def sech2_fwhm(x, A, x0, fwhm, c):
    tau = fwhm * 2 / 1.76
    return A / (np.cosh((x - x0) / tau)) ** 2 + c


def sech2_fwhm_wings(x, a, xc, fwhm, off, wing_sep, wing_ratio, wings_n):
    """ sech squared with n wings."""
    res = sech2_fwhm(x, a, xc, fwhm, off)
    for n in range(1, wings_n):
        res += sech2_fwhm(x, a * (wing_ratio ** n), xc - n * wing_sep, fwhm, off)
        res += sech2_fwhm(x, a * (wing_ratio ** n), xc + n * wing_sep, fwhm, off)
    return res


def sin(x, A, f, p, o):
    return A * np.sin(x / f + p) + o


def globalcounter(idx, M):
    counterlist = idx[::-1]
    maxlist = M[::-1]
    for i in range(len(counterlist)):
        counterlist[i] = counterlist[i] * np.prod(maxlist[:i])
    return int(np.sum(counterlist))


def transient_1expdec(t, A1, tau1, sigma, y0, off, t0):
    """ Fitting function for transients, 1 exponent decay.
    A: Amplitude
    Tau: exp decay
    sigma: pump pulse duration
    y0: whole curve offset
    off: slow dynamics offset"""
    t = t - t0
    tmp = erf((sigma ** 2. - 5.545 * tau1 * t) / (2.7726 * sigma * tau1))
    tmp = .5 * (1 - tmp) * np.exp(sigma ** 2. / (11.09 * tau1 ** 2.))
    return y0 + tmp * (A1 * (np.exp(-t / tau1)) + off)


def update_average(new, avg, n):
    'recalculate average with new dataset.'
    return (avg * (n - 1) + new) / n

tiny = np.finfo(np.float64).eps


def gaussian2dcorr(x, y=0.0, offset=0.0, corr= 0.0, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0,
               sigmay=1.0):
    """Return a 2-dimensional Gaussian with correlations as implemented in Igor Gauss2D
    """
    return offset+amplitude*np.exp((-1.0/max(tiny,2*(1-corr)))*(((x-centerx)/max(tiny,sigmax))**2+((y-centery)/max(tiny,sigmay))**2-((2*corr*(x-centerx)*(y-centery))/max(tiny,sigmax*sigmay))))

def gaussian2drot(x, y=0.0, offset=0.0, angle= 0.0, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0,
               sigmay=1.0):
    """Return a 2-dimensional Gaussian with rotation angle
    """
    xp = (x - centerx)*np.cos(angle) - (y - centery)*np.sin(angle)
    yp = (x - centerx)*np.sin(angle) + (y - centery)*np.cos(angle)
    return offset+amplitude*np.exp((-1.0/max(tiny,2*(1)))*(((xp)/max(tiny,sigmax))**2+((yp)/max(tiny,sigmay))**2))

def lorentzian(x,A,x0,gamma):
    """ Lorentzian distribution """
    return A/(np.pi * gamma) * np.power(gamma,2) / (np.power(gamma,2) + np.power(x-x0,2))

def lorentzian_distribution2D(x,y,A,x0,y0,gamma):
    """ Lorentzian distribution, symmetric in x,y """
    return A/(np.pi * gamma) * np.power(gamma,2) / (np.power(gamma,2) + np.power(x-x0,2) +  np.power(y-y0,2))

def lorentzian_asymm2D(x,y,A,x0,y0,gx,gy):
    """ Lorentzian distribution, asymmetric in x,y """
    xp = (x - centerx)*np.cos(angle) - (y - centery)*np.sin(angle)
    yp = (x - centerx)*np.sin(angle) + (y - centery)*np.cos(angle)
    return A * lorentzian_distribution2D(xp,1,0,gx) * lorentzian_distribution2D(yp,1,0,gy)



def gaussian2D(x,y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#     x, y = M
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    return offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))

def lorentzian2D(x,y, amp, mux, muy, g, c):
    numerator = np.abs(amp * g)
    denominator = ((x - mux) ** 2 + (y - muy) ** 2 + g ** 2) ** 1.5
    return numerator / denominator + c

def multi_lorentzian2D(M,*args):
    x,y = M
    arr = np.zeros(x.shape)
    n=7
    for i in range(len(args)//n):
        arr += lorentzian2D(M, *args[i*n:i*n+n])
    return arr

def multi_gaussian2D(M, *args):
    x,y = M
    arr = None
    n=7
    for i in range(len(args)//n):
        if arr is None:
            arr = gaussian2D(x,y, *args[i*n:i*n+n])
        else:
            arr += gaussian2D(x,y, *args[i*n:i*n+n])
    return arr
def main():
    pass


if __name__ == '__main__':
    main()
