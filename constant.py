from atmospkg.unit import Tunitconversion
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
    earth_omega = 2 * np.pi / 86400

    @classmethod
    def f(cls, lat):
        return 2 * cls.earth_omega * np.sin(lat)
    
    @staticmethod
    def latent_heat(T, Tunit="K"): # J / kg
        T = Tunitconversion(T, nowunit=Tunit, aimunit="degC")
        return (2.501 - 0.00237 * T) * 1e6