import numpy as np
from atmospkg.constant import constants
from atmospkg.parameterization import turbulent_fluxes_Deardorff1975
from mypkgs.processor.numericalmethod import central_diff_4thorder, central_diff
from mypkgs.processor.array_process import broadcast_to_any

# https://journals.ametsoc.org/view/journals/atot/34/11/jtech-d-17-0073.1.xml
# https://journals.ametsoc.org/view/journals/mwre/113/12/1520-0493_1985_113_2142_rotffm_2_0_co_2.xml

def broadcast_var(aimvarname, varnamelist=[]):
    def decorator(func):
        def wrapper(**kwargs):
            aimvar = kwargs[aimvarname]
            if isinstance(aimvar, np.ndarray):
                aimshape = aimvar.shape
            elif isinstance(aimvar, list) or isinstance(aimvar, tuple):
                aimshape = aimvar[0].shape
            for varname in varnamelist:
                if isinstance(kwargs[varname], np.ndarray):
                    if kwargs[varname].shape != aimshape:
                        kwargs[varname] = broadcast_to_any(kwargs[varname], shape=aimshape)
            return func(**kwargs)
        return wrapper
    return decorator

@broadcast_var("U", varnamelist=["theta_rho_bar", "rho", "lat"])
def dpidx(U = [], X = [], theta_rho_bar = np.array([]), rho = np.array([]), lat = np.array([]), diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    A (with turbulence term) in eq10 in Foerster and Bell 2017 (actually minus A)\n
    U: [w(nz, ny, nx), v(nz, ny, nx), u(nz, ny, nx)]\n
    X: [z(nz), y(ny), x(nx)]\n
    theta_rho_bar(nz) or (nz, ny, nx)\n
    rho(nz) or (nz, ny, nx)\n
    lat(ny, nx)\n
    diffrential_func(Y, X)\n
    theta_rho = T_rho(p/p0)^kappa, T_rho = p/rho/Rd, rho = rho_d+rho_v
    """
    w, v, u = U
    z, y, x = X
    Cp = constants.Cp
    f = constants.f(lat)
    dudx = diffrential_func_adv_horizontal(u, x, 2, broadX=True)
    dudy = diffrential_func_adv_horizontal(u, y, 1, broadX=True)
    dudz = diffrential_func_adv_vertical(u, z, 0, broadX=True)
    tau = turbulent_fluxes_Deardorff1975(U=[w, v, u], X=[z, y, x], i=2, rho=rho, diffrential_func = diffrential_func_turb)
    rho = broadcast_to_any(rho, shape=U[0].shape, n=0)
    theta_rho_bar = broadcast_to_any(theta_rho_bar, shape=U[0].shape, n=0)
    turb = 0#diffrential_func_adv_horizontal(tau, x, 2, broadX=True) / rho

    result = -1 / Cp / theta_rho_bar * (u * dudx + v * dudy + w * dudz - f * v - turb)
    return result

@broadcast_var("U", varnamelist=["theta_rho_bar", "rho", "lat"])
def dpidy(U = [], X = [], theta_rho_bar = np.array([]), rho = np.array([]), lat = np.array([]), diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    B (with turbulence term) in eq11 in Foerster and Bell 2017 (actually minus B)\n
    U: [w(nz, ny, nx), v(nz, ny, nx), u(nz, ny, nx)]\n
    X: [z(nz), y(ny), x(nx)]\n
    theta_rho_bar(nz) or (nz, ny, nx)\n
    rho(nz) or (nz, ny, nx)\n
    lat(ny, nx)\n
    diffrential_func(Y, X)
    """
    w, v, u = U
    z, y, x = X
    Cp = constants.Cp
    f = constants.f(lat)
    dvdx = diffrential_func_adv_horizontal(v, x, 2, broadX=True)
    dvdy = diffrential_func_adv_horizontal(v, y, 1, broadX=True)
    dvdz = diffrential_func_adv_vertical(v, z, 0, broadX=True)
    tau = turbulent_fluxes_Deardorff1975(U=[w, v, u], X=[z, y, x], i=1, rho=rho, diffrential_func = diffrential_func_turb)
    rho = broadcast_to_any(rho, shape=U[0].shape, n=0)
    turb = 0#diffrential_func_adv_horizontal(tau, y, 1, broadX=True) / rho
    result = -1 / Cp / theta_rho_bar * (u * dvdx + v * dvdy + w * dvdz + f * u - turb)
    return result

@broadcast_var("U", varnamelist=["theta_rho_bar", "rho", "qr"])
def dpidz_m_temp(U = [], X = [], theta_rho_bar = np.array([]), rho = np.array([]), qr = np.array([]), diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    C in eq11 in Foerster and Bell 2017 (actually minus C) and consider q term + turbulence term\n
    U: [w(nz, ny, nx), v(nz, ny, nx), u(nz, ny, nx)]\n
    X: [z(nz), y(ny), x(nx)]\n
    theta_rho_bar(nz) or (nz, ny, nx)\n
    rho(nz) or (nz, ny, nx)\n
    qr(nz, ny, nx)\n
    diffrential_func(Y, X)
    """
    w, v, u = U
    z, y, x = X
    Cp = constants.Cp
    g = constants.g
    dwdx = diffrential_func_adv_horizontal(w, x, 2, broadX=True)
    dwdy = diffrential_func_adv_horizontal(w, y, 1, broadX=True)
    dwdz = diffrential_func_adv_vertical(w, z, 0, broadX=True)
    tau = turbulent_fluxes_Deardorff1975(U=[w, v, u], X=[z, y, x], i=1, rho=rho, diffrential_func = diffrential_func_turb)
    rho = broadcast_to_any(rho, shape=U[0].shape, n=0)
    turb = 0#diffrential_func_adv_vertical(tau, z, 0, broadX=True) / rho
    result = -1 / Cp / theta_rho_bar * (u * dwdx + v * dwdy + w * dwdz + qr * g - turb)
    return result

@broadcast_var("U", varnamelist=["theta_rho_bar", "rho", "lat", "qr"])
def theta_gradient(U, X, theta_rho_bar, rho, lat, qr, diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    D & E in eq13 in Foerster and Bell 2017 and consider q term + turbulence term\n
    U: [w(nz, ny, nx), v(nz, ny, nx), u(nz, ny, nx)]\n
    X: [z(nz), y(ny), x(nx)]\n
    theta_rho_bar(nz)\n
    rho(nz)\n
    lat(ny, nx)\n
    qr(nz, ny, nx)\n
    diffrential_func(Y, X)\n
    return result_dthetadx, result_dthetady
    """
    z, y, x = X
    Cp = constants.Cp
    g = constants.g
    args = dict(U=U, X=X, theta_rho_bar=theta_rho_bar, rho=rho, diffrential_func_adv_horizontal=diffrential_func_adv_horizontal, diffrential_func_adv_vertical=diffrential_func_adv_vertical, diffrential_func_turb=diffrential_func_turb)
    A = -dpidx(lat=lat, **args)
    B = -dpidy(lat=lat, **args)
    C = -dpidz_m_temp(qr=qr, **args)
    dAdz = diffrential_func_adv_vertical(A, z, n=0, broadX=True)
    dCdx = diffrential_func_adv_horizontal(C, x, n=2, broadX=True)
    result_dthetadx = -Cp * theta_rho_bar**2 / g * (dAdz - dCdx)
    dBdz = diffrential_func_adv_vertical(B, z, n=0, broadX=True)
    dCdy = diffrential_func_adv_horizontal(C, y, n=1, broadX=True)
    result_dthetady = -Cp * theta_rho_bar**2 / g * (dBdz - dCdy)
    
    return result_dthetadx, result_dthetady
