import numpy as np

from mypkgs.processor.numericalmethod import central_diff, central_diff_4thorder

def divergence(u, v, x_1D, y_1D, xaxis = -1, yaxis = -2, ndiff = 2):
    if ndiff == 2:
        _central_diff = central_diff
    elif ndiff == 4:
        _central_diff = central_diff_4thorder
    dudx = _central_diff(u, x_1D, xaxis, broadX=True)
    dvdy = _central_diff(v, y_1D, yaxis, broadX=True)
    div = dudx + dvdy
    return div

def vorticity(u, v, x_1D, y_1D, xaxis = -1, yaxis = -2, ndiff = 2):
    if ndiff == 2:
        _central_diff = central_diff
    elif ndiff == 4:
        _central_diff = central_diff_4thorder
    dudy = _central_diff(u, y_1D, yaxis, broadX=True)
    dvdx = _central_diff(v, x_1D, xaxis, broadX=True)
    div = - dudy + dvdx
    return div