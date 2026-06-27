import numpy as np

def miki_impedance(f, sigma, l):
    """
    sigma: flow resistivity (Pa·s/m²)
    l: thickness (m)
    """
    c = 343.0
    rho0 = 1.21
    Z0 = rho0 * c

    x = 1e3 * f / sigma

    # Propagation constant (eq. 35)
    kc = (2 * np.pi * f / c) * (
        (1 + 7.81 * x**(-0.618)) - 1j * 11.41 * x**(-0.618)
    )

    # Characteristic impedance (eq. 36)
    Zc = Z0 * (
        (1 + 5.50 * x**(-0.632)) - 1j * 8.43 * x**(-0.632)
    )

    # Normal incidence surface impedance with rigid backing (eq. 37)
    Z_DBM = -1j * Zc * (1 / np.tan(kc * l))

    return Z_DBM

# Material 2 parameters from Table 1
sigma2 = 63e3   # Pa·s/m²
l2     = 37e-3  # m

def carpet_impedance(f, sigma, l, d, m, r):
    """
    d: equivalent air layer thickness (m)
    m: mass per unit area (kg/m²)
    r: damping factor (Pa·s/m)
    """
    c = 343.0
    rho0 = 1.21

    # DBM part
    Z_DBM = miki_impedance(f, sigma, l)

    # Membrane resonator part (eq. 38)
    Z_res = (r / (rho0 * c)) + (1j / (rho0 * c)) * (
        2 * np.pi * f * m - rho0 * c * (1 / np.tan(2 * np.pi * f * d / c))
    )

    # Combined carpet impedance (eq. 39)
    Z_car = (Z_res * Z_DBM) / (Z_res + Z_DBM)

    return Z_car

# Material 3 parameters from Table 1
sigma3 = 6.5e3   # Pa·s/m²
l3     = 8.5e-3  # m
d3     = 2.6e-3  # m
m3     = 2.3     # kg/m²
r3     = 5e3     # Pa·s/m