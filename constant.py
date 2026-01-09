from atmospkg.unit import Tunitconversion
import numpy as np

class constants:
    # gas constant (J / K / mol)
    R = 8.314462618 # https://physics.nist.gov/cgi-bin/cuu/Value?r
    # molecular weight (kg / kmol) of 
    Md = 28.964 # dry air (U.S. standard atmosphere 1976 https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf
    Mv = 18.0153 # water vapor (https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185
    # gas constant  (J / K / kg)
    Rd = R / Md * 1000 # dry air, about 287
    Ra = Rd
    Rv = R / Mv * 1000 # water vapor, about 461
    # parameter
    epsilon = Rd / Rv
    # specific heat capacity (J / K / kg)
    Cvv = 1418 # at constant volume of water vapor (Romps 2017: https://doi.org/10.1175/JAS-D-17-0102.1)
    Cpv = Cvv + Rv # about 1879
    Cvd = 719 # at constant volume of dry air ("Cva" in Romps 2017: https://doi.org/10.1175/JAS-D-17-0102.1)
    Cva = Cvd
    Cpd = Cvd + Rd # about 1006
    Cpa = Cpd
    Cp = Cpd
    Cvl = 4119 # liquid water
    Cvs = 1861 # solid
    # Cp = 1005.7
    kappa = Rd / Cp
    # triple point
    ptrip = 611.65 # Pa
    Ttrip = 273.16 # K
    E0v = 2.3740e6 # J / kg difference in specific internal energy between water vapor and liquid at the triple point
    E0s = 0.3337e6 # J / kg difference in specific internal energy between liquid and solid at the triple point
    
    g = 9.81
    earth_omega = 2 * np.pi / 86400

    @classmethod
    def f(cls, lat):
        return 2 * cls.earth_omega * np.sin(lat)
    
    @staticmethod
    def latent_heat(T, Tunit="K"): # J / kg
        T = Tunitconversion(T, nowunit=Tunit, aimunit="degC")
        return (2.501 - 0.00237 * T) * 1e6