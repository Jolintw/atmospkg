import numpy as np
from mypkgs.processor.numericalmethod import central_diff

def water_mixingratio_AG1979(Z, rho):
    """
    Raindrop Sizes and Related Parameters for GATE (Pauline M. Austin and Spiros G. Geotis) https://journals.ametsoc.org/view/journals/apme/18/4/1520-0450_1979_018_0569_rsarpf_2_0_co_2.xml\n
    Z: reflectivity (dB)\n
    rho: density (kg/m^3)\n
    qr: kg/kg\n
    """
    qr = 1.15 * 1e-6 * Z**(0.76) / rho
    return qr

def turbulent_fluxes_Deardorff1975(U, X, i, rho, diffrential_func = central_diff):
    """
    U: [w(nz, ny, nx), v(nz, ny, nx), u(nz, ny, nx)]\n
    X: [z(nz), y(ny), x(nx)]\n
    i: w is 0, v is 1, u is 2
    origin source: The development of boundary-layer turbulence models for use in studying the severe storm environment (Deardorff 1975)\n
    https://books.google.com.tw/books?hl=en&lr=&id=XSku2W9stMoC&oi=fnd&pg=PA251&dq=the+development+of+boundary+layer+turbulence+models+for+use+in+studying+the+severe&ots=gWwbBOF9Aa&sig=UnyrhD_RHwE_dJ71bO2WlGkADpQ&redir_esc=y#v=onepage&q=the%20development%20of%20boundary%20layer%20turbulence%20models%20for%20use%20in%20studying%20the%20severe&f=false\n
    eq 2.2 and eq 2.5 of following paper could easier to read: A Method for the Initialization of the Anelastic Equations: Implications for Matching Models with Observations (Tzvi Gal-Chen 1978)\n
    https://journals.ametsoc.org/view/journals/mwre/106/5/1520-0493_1978_106_0587_amftio_2_0_co_2.xml\n
    """
    std = [np.nanmean(np.power(ui - np.nanmean(ui), 2)) for ui in U]
    DeltaX = [np.mean(xi[1:]-xi[:-1]) for xi in X]
    Delta = np.power(DeltaX[0] * DeltaX[1] * DeltaX[2], 1.0/3.0)
    E = sum(std) / 2.0
    Km = 0.12*Delta*np.power(E, 0.5)
    duidxj = np.array([diffrential_func(Y=U[i], X=X[j], n=j, broadX=True) for j in range(3)])
    dujdxi = np.array([diffrential_func(Y=U[j], X=X[i], n=i, broadX=True) for j in range(3)])
    tau = rho*Km*(np.sum(duidxj, n=0) + np.sum(dujdxi, n=0))
    return tau
