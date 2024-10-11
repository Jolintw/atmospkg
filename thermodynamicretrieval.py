from atmospkg.constant import constants
from atmospkg.parameterization import turbulent_fluxes_Deardorff1975
from mypkgs.processor.numericalmethod import central_diff_4thorder, central_diff

# https://journals.ametsoc.org/view/journals/atot/34/11/jtech-d-17-0073.1.xml
# https://journals.ametsoc.org/view/journals/mwre/113/12/1520-0493_1985_113_2142_rotffm_2_0_co_2.xml

def dpidx(U, X, theta_rho_bar, rho, lat, diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    A (with turbulence term) in eq10 in Foerster and Bell 2017 (actually minus A)\n
    var(nz, ny, nx)\n
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
    turb = tau / rho
    result = -1 / Cp / theta_rho_bar * (u * dudx + v * dudy + w * dudz - f * v - turb)
    return result

def dpidy(U, X, theta_rho_bar, rho, lat, diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    B (with turbulence term) in eq11 in Foerster and Bell 2017 (actually minus B)\n
    var(nz, ny, nx)\n
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
    turb = tau / rho
    result = -1 / Cp / theta_rho_bar * (u * dvdx + v * dvdy + w * dvdz + f * u - turb)
    return result

def dpidz_m_temp(U, X, theta_rho_bar, rho, qr, diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    C in eq11 in Foerster and Bell 2017 (actually minus C) and consider q term + turbulence term\n
    var(nz, ny, nx)\n
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
    turb = tau / rho
    result = -1 / Cp / theta_rho_bar * (u * dwdx + v * dwdy + w * dwdz + qr * g - turb)
    return result

def theta_gradient(U, X, theta_rho_bar, rho, lat, qr, diffrential_func_adv_horizontal = central_diff_4thorder, diffrential_func_adv_vertical = central_diff, diffrential_func_turb = central_diff):
    """
    D in eq13 in Foerster and Bell 2017 and consider q term + turbulence term\n
    var(nz, ny, nx)\n
    diffrential_func(Y, X)\n
    return result_dthetadx, result_dthetady
    """
    z, y, x = X
    Cp = constants.Cp
    g = constants.g
    A = -dpidx(U, X, theta_rho_bar, rho, lat, diffrential_func_adv_horizontal, diffrential_func_adv_vertical, diffrential_func_turb)
    B = -dpidy(U, X, theta_rho_bar, rho, lat, diffrential_func_adv_horizontal, diffrential_func_adv_vertical, diffrential_func_turb)
    C = -dpidz_m_temp(U, X, theta_rho_bar, rho, qr, diffrential_func_adv_horizontal, diffrential_func_adv_vertical, diffrential_func_turb)
    dAdz = diffrential_func_adv_vertical(A, z, broadX=True)
    dCdx = diffrential_func_adv_horizontal(C, x, broadX=True)
    result_dthetadx = -Cp * theta_rho_bar**2 / g * (dAdz - dCdx)
    dBdz = diffrential_func_adv_vertical(B, z, broadX=True)
    dCdy = diffrential_func_adv_horizontal(C, y, broadX=True)
    result_dthetady = -Cp * theta_rho_bar**2 / g * (dBdz - dCdy)
    
    return result_dthetadx, result_dthetady
