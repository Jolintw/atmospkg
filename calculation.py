import numpy as np
class constants:
    # gas constant (J / K / mol)
    R = 8.314462618 # https://physics.nist.gov/cgi-bin/cuu/Value?r
    # molecular weight (kg / kmol) of 
    Md = 28.964 # dry air (U.S. standard atmosphere 1976 https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf
    Mv = 18.0153 # water vapor (https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185
    # gas constant  (J / K / kg)
    Rd = R / Md * 1000 # dry air 
    Rv = R / Mv * 1000 # water vapor
    # parameter
    epsilon = Rd / Rv
    # specific heat
    Cp = 1005.7
    kappa = Rd / Cp

def saturation_vapor_pressure(T, Tunit = "K"):
    """estimate saturation vapor pressure (es)
    es(T) = 6.1094 * e^(17.625T/(243.04+T))
    T unit: K
    es unit: Pa
    recommended by Alduchov and Eskridge (1996) https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2
    :param T: temperature
    :param Tunit: unit of temperature K or degC
    """
    T = Tunitconversion(T, Tunit, aimunit = "degC")
    return 6.1094 * np.exp(T*17.625/(243.04+T))

def saturation_mixingratio(T, P, Tunit = "K", Punit = "Pa"):
    es = saturation_vapor_pressure(T, Tunit)
    P = Punitconversion(P, Punit, aimunit="Pa")
    return mixingratio_from_pressure(es, P)

def mixingratio_from_pressure(e, P):
    """
    :param e: vapor pressure
    :param P: air pressure
    """
    return constants.epsilon*e/(P-e)

def potential_temperature(T, P, Tunit = "K", Punit = "Pa"):
    """estimate potential temperature
    theta = T * (100000 / P) ^ (Rd / Cp)
    T unit: K
    P unit: Pa
    :param T: temperature
    :param Tunit: unit of temperature K or degC
    :param P: pressure
    :param Punit: unit of pressure Pa or hPa
    """
    T = Tunitconversion(T, Tunit, aimunit = "K")
    P = Punitconversion(P, Punit, aimunit="Pa")
    pt = T * (1e5 / P)**(constants.kappa)
    return pt

def wswd_to_uv(ws, wd, wdunit = "rad", wdtype = "met"):
    wd = angleunitconversion(wd, wdunit, "rad")
    wd = angletypeconversion(wd, wdtype, "met")
    u = - np.sin(wd) * ws
    v = - np.cos(wd) * ws
    return u, v

T_standard_unit = "K"
def Tunitconversion(T, nowunit, aimunit = T_standard_unit):
    if aimunit != T_standard_unit and nowunit != T_standard_unit:
        T = Tunitconversion(T, nowunit)
        nowunit = T_standard_unit
    if nowunit == "K" and aimunit == "degC":
        T = T - 273.15
    if nowunit == "degC" and aimunit == "K":
        T = T + 273.15
    return T

P_standard_unit = "Pa"
def Punitconversion(P, nowunit, aimunit = P_standard_unit):
    if aimunit != P_standard_unit and nowunit != P_standard_unit:
        P = Punitconversion(P, nowunit)
        nowunit = P_standard_unit
    if nowunit == "Pa" and aimunit == "hPa":
        P = P / 100.0
    if nowunit == "hPa" and aimunit == "Pa":
        P = P * 100.0
    return P

angle_standard_unit = "rad"
def angleunitconversion(angle, nowunit, aimunit = angle_standard_unit):
    if nowunit == "rad" and aimunit == "deg":
        angle = angle/np.pi*180.0
    if nowunit == "deg" and aimunit == "rad":
        angle = angle/180.0*np.pi
    return angle

def angletypeconversion(angle, nowtype, aimtype):
    if nowtype != aimtype:
        angle = np.pi / 2 - angle
    return angle