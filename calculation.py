import numpy as np

from atmospkg.constant import constants
from atmospkg.unit import Tunitconversion, Qunitconversion, Punitconversion
from atmospkg.unit import angletypeconversion, angleunitconversion


def saturation_vapor_pressure(T, Tunit = "K"):
    """estimate saturation vapor pressure (es)\n
    es(T) = 6.1094 * e^(17.625T/(243.04+T))\n
    T unit: degC\n
    es unit: hPa\n
    recommended by Alduchov and Eskridge (1996) https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2\n
    :param T: temperature\n
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
    :param e: vapor pressure\n
    :param P: air pressure
    """
    return constants.epsilon*e/(P-e)

def vapor_pressure_from_mixingratio(qv, P, qvunit = "kg/kg", Punit = "Pa"):
    """
    qv: mixingratio\n
    P: air pressure\n
    return: vapor pressure (Pa)
    """
    P  = Punitconversion(P, Punit, aimunit="Pa")
    qv = Qunitconversion(qv, qvunit, aimunit="kg/kg")
    return qv / constants.epsilon * P / (1 + qv / constants.epsilon)

def specific_humidity_from_mixingratio(qv, qvunit = "kg/kg"):
    qv = Qunitconversion(qv, qvunit, aimunit="kg/kg")
    return qv / (1 + qv)

def potential_temperature(T, P, P_ref = 100000, Tunit = "K", Punit = "Pa", P_refunit = "Pa"):
    """estimate potential temperature (adiabatic lifting of downwelling to P_base)\n
    theta = T * (100000 / P) ^ (Rd / Cp)\n
    T unit: K\n
    P unit: Pa\n
    :param T: temperature\n
    :param Tunit: unit of temperature K or degC\n
    :param P: pressure\n
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
    P: pressure\n
    T: temperature\n
    e: water vapor pressure\n
    rho: density\n
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
    P: pressure\n
    dP: 10 times of pressure resolution\n
    T: temperature\n
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

def sea_level_pressure(T, P, z, lapse_rate, z_ref = 0, Tunit="K", Punit="Pa"):
    """
    z: meter\n
    lapse_rate: K / meter\n
    return slp (Pa)
    """
    TK = Tunitconversion(T, nowunit=Tunit, aimunit="K")
    P  = Punitconversion(P, nowunit=Punit, aimunit="Pa")
    g = constants.g
    Rd = constants.Rd
    SLP = P * ((TK - (z_ref - z)*lapse_rate) / TK) ** (g / Rd / lapse_rate)
    return SLP

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


def column_water_vapor(P, qv, Punit="Pa", qvunit="kg/kg", zaxis = 0):
    """
    also called precipitable water\n
    zaxis: which axis (dim) is for height (or pressure)
    """
    P  = Punitconversion(P, nowunit=Punit, aimunit="Pa")
    qv = Qunitconversion(qv, nowunit=qvunit, aimunit="kg/kg")
    sh = specific_humidity_from_mixingratio(qv)
    P = P.swapaxes(0, zaxis)
    sh = sh.swapaxes(0, zaxis)
    dP = - (P[1:] - P[:-1])
    sh_mid = (sh[1:] + sh[:-1]) / 2.0
    cwv = (dP * sh_mid) / constants.g
    cwv = cwv.swapaxes(0, zaxis)
    cwv = np.nansum(cwv, axis=zaxis)
    return cwv

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
    