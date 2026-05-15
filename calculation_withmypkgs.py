import numpy as np

from mypkgs.processor.numericalmethod import central_diff, central_diff_4thorder
from atmospkg.constant import constants

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

def gradient(var, x_1D, y_1D, xaxis = -1, yaxis = -2, ndiff = 2):
    if ndiff == 2:
        _central_diff = central_diff
    elif ndiff == 4:
        _central_diff = central_diff_4thorder
    gradient_y = _central_diff(var, y_1D, yaxis, broadX=True)
    gradient_x = _central_diff(var, x_1D, xaxis, broadX=True)
    return gradient_x, gradient_y

def frontal_variable(vorticity, temperature_gradient_magnitude):
    """F^*=zeta_p|Nabla(T_p)| \n
    zeta_p: the isobaric relative vorticity\n
    Nabla(T_p): horizontal temperature gradient on a pressure surface\n
    please check methodology of Parfitt, R., A.Czaja, and H.Seo (2017), A simple diagnostic for the detection of atmospheric fronts, Geophys. Res. Lett., 44, 4351–4358, doi:10.1002/2017GL073662.\n
    can also replace T_p by \\theta_e_p (temperature -> equivalent potential temperature)\n
    (suggested by Cornejo, I. C., A. K. Rowe, K. L. Rasmussen, and J. C. DeHart, 2024: Orographic Controls on Extreme Precipitation Associated with a Mei-Yu Front. Mon. Wea. Rev., 152, 531–551, https://doi.org/10.1175/MWR-D-23-0170.1.)
    Args:
        vorticity (float): standard units
        temperature_gradient_magnitude (float): standard units
        return (float): standard units
    """
    return vorticity * temperature_gradient_magnitude

def frontal_diagnostic(frontal_variable, latitude):
    """F=F^*/(f|Nabla T|_0)\n
    |Nabla T|_0=0.45 K / (100 km)\n
    Args:
        frontal_variable (float): standard units
        latitude (float): in degree
        return (float): no units
    """
    t_gradient_0 = 0.45 / 100 / 1000
    latitude_rad = latitude / 180 * np.pi
    f = constants.f(latitude_rad)
    return frontal_variable / f / t_gradient_0