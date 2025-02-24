import numpy as np
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
    """
    angle: radius angle\n
    type: math (0 for east and anticlockwise), met (0 for north and clockwise)
    """
    if nowtype != aimtype:
        angle = np.pi / 2 - angle
    return angle

def Qunitconversion(Q, nowunit, aimunit):
    if nowunit == "g/kg" and (aimunit in ["kg/kg", ""]):
        Q = Q / 1e3
    if (nowunit in ["kg/kg", ""]) and aimunit == "g/kg":
        Q = Q * 1e3
    return Q
