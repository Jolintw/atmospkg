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
    g = 9.81
    def latent_heat(T, Tunit="K"): # J / kg
        T = Tunitconversion(T, nowunit=Tunit, aimunit="degC")
        return (2.501 - 0.00237 * T) * 1e6

def saturation_vapor_pressure(T, Tunit = "K"):
    """estimate saturation vapor pressure (es)
    es(T) = 6.1094 * e^(17.625T/(243.04+T))
    T unit: degC
    es unit: hPa
    recommended by Alduchov and Eskridge (1996) https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2
    :param T: temperature
    :param Tunit: unit of temperature K or degC
    """
    T = Tunitconversion(T, Tunit, aimunit = "degC")
    return 6.1094 * np.exp(T*17.625/(243.04+T))

def saturation_mixingratio(T, P, Tunit = "K", Punit = "Pa"):
    es = saturation_vapor_pressure(T, Tunit)
    P = Punitconversion(P, Punit, aimunit="hPa")
    return mixingratio_from_pressure(es, P)

def mixingratio_from_pressure(e, P):
    """
    :param e: vapor pressure
    :param P: air pressure
    """
    return constants.epsilon*e/(P-e)

def vapor_pressure_from_mixingratio(qv, P, qvunit = "kg/kg", Punit = "Pa"):
    """
    qv: mixingratio
    P: air pressure
    return: vapor pressure (Pa)
    """
    P  = Punitconversion(P, Punit, aimunit="Pa")
    qv = Qunitconversion(qv, qvunit, aimunit="kg/kg")
    return qv / constants.epsilon * P / (1 + qv / constants.epsilon)


def potential_temperature(T, P, P_ref = 100000, Tunit = "K", Punit = "Pa", P_refunit = "Pa"):
    """estimate potential temperature (adiabatic lifting of downwelling to P_base)
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
    P_ref = Punitconversion(P_ref, P_refunit, aimunit="Pa")
    pt = T * (P_ref / P)**(constants.kappa)
    return pt

def wswd_to_uv(ws, wd, wdunit = "rad", wdtype = "met"):
    wd = angleunitconversion(wd, wdunit, "rad")
    wd = angletypeconversion(wd, wdtype, "met")
    u = - np.sin(wd) * ws
    v = - np.cos(wd) * ws
    return u, v

def calculate_geopotential_height(P, T, e, Tunit = "K", Punit = "Pa", eunit = "Pa"):
    """
    P: pressure
    T: temperature
    e: water vapor pressure
    rho: density
    z: geopotential height
    """
    T  = Tunitconversion(T, Tunit, aimunit="K")
    P  = Punitconversion(P, Punit, aimunit="Pa")
    e  = Punitconversion(e, eunit, aimunit="Pa")
    dP = P[1:] - P[:-1]
    T_mid   = (T[1:] + T[:-1]) / 2
    e_mid   = (e[1:] + e[:-1]) / 2
    Pd_mid  = (P[1:] + P[:-1]) / 2 - e_mid
    rho_mid = (Pd_mid / constants.Rd + e_mid / constants.Rv) / T_mid
    dz      = - dP / (rho_mid * constants.g)
    z       = np.zeros_like(P)
    z[1:]   = np.cumsum(dz)
    return z

def calculate_LCL(P, T, qv, dP, Tunit = "K", Punit = "Pa", qvunit = "kg/kg"):
    """
    P: pressure
    dP: 10 times of pressure resolution
    T: temperature
    qv: water vapor mixingratio
    """
    T  = Tunitconversion(T, Tunit, aimunit="K")
    P  = Punitconversion(P, Punit, aimunit="Pa")
    dP  = Punitconversion(dP, Punit, aimunit="Pa")
    qv = Qunitconversion(qv, qvunit, aimunit="kg/kg")
    # PT_1000hPa = potential_temperature(T, P, Tunit = Tunit, Punit = Punit)
    Pnow = P
    Tnow = potential_temperature(T, P, P_ref=Pnow)
    qvsnow = saturation_mixingratio(T=Tnow, P=Pnow)
    while qvsnow < qv:
        Pnow += dP
        Tnow = potential_temperature(T, P, P_ref=Pnow)
        qvsnow = saturation_mixingratio(T=Tnow, P=Pnow)

    dP = dP * 0.1
    while qvsnow > qv:
        Pnow -= dP
        Tnow = potential_temperature(T, P, P_ref=Pnow)
        qvsnow = saturation_mixingratio(T=Tnow, P=Pnow)
    Pnow += dP

    return Pnow

def equivalent_potential_temperature(T, P, qv, Tunit="K", Punit="Pa", qvunit="kg/kg"):
    """
    The Computation of Equivalent Potential Temperature (David Bolton 1980)
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    TK = Tunitconversion(T, nowunit=Tunit, aimunit="K")
    P  = Punitconversion(P, nowunit=Punit, aimunit="hPa")
    qv = Qunitconversion(qv, nowunit=qvunit, aimunit="g/kg")
    e  = vapor_pressure_from_mixingratio(qv, P, qvunit="g/kg", Punit="hPa") 
    e  = Punitconversion(e, nowunit="Pa", aimunit="hPa")
    TL = 2840 / (3.5*np.log(TK) - np.log(e) - 4.805) + 55
    EPT = TK*((1000 / P) ** (0.2854 * (1 - 0.28 * 1e-3 * qv)))
    EPT = EPT * np.exp((3.376 / TL - 0.00254) * qv * (1 + 0.81 * 1e-3 * qv))
    return EPT


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

def Qunitconversion(Q, nowunit, aimunit):
    if nowunit == "g/kg" and (aimunit in ["kg/kg", ""]):
        Q = Q / 1e3
    if (nowunit in ["kg/kg", ""]) and aimunit == "g/kg":
        Q = Q * 1e3
    return Q

if __name__ == "__main__":
    T = 25
    P = 1000
    qv = 16
    LCL = calculate_LCL(P, T, qv, dP=0.1, Tunit = "degC", Punit = "hPa", qvunit = "g/kg")
    # print(LCL)
    EPT = equivalent_potential_temperature(T, P, qv, Tunit="degC", Punit="hPa", qvunit="g/kg")
    
    TL = potential_temperature(T, P, P_ref=LCL, Tunit="degC", Punit="hPa")
    print(TL)
    print(EPT)
    