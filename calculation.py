import numpy as np

def saturation_vapor_pressure(T, Tunit = "K"):
    """estimate saturation vapor pressure (es)
    es(T) = 6.1094 * e^(17.625T/(243.04+T))
    T unit: K
    es unit: Pa
    recommended by Alduchov and Eskridge (1996) https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2
    :param T: temperature
    :param Tunit: unit of temperature K or degC
    """
    return 6.1094 * np.exp(T*17.625/(243.04+T))

def 