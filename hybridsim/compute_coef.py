import numpy as np

rho0 = 1.213
c0   = 343.0

# Table 1 parameters
sigma_2 = 63e3;  l_2 = 37e-3                          # Panel
sigma_3 = 6.5e3; l_3 = 8.5e-3; d_3 = 2.6e-3; m_3 = 2.3; r_3 = 5e3  # Carpet

def miki(f, sigma, l):
    x  = 1e3 * f / sigma
    kc = (2*np.pi*f/c0) * (1 + 7.81*x**-0.618 - 1j*11.41*x**-0.618)
    Zc = 1 + 5.50*x**-0.632 - 1j*8.43*x**-0.632
    return -1j * Zc / np.tan(kc * l)

def carpet(f):
    Z_DBM = miki(f, sigma_3, l_3)
    Z_res = (r_3/(rho0*c0) + 1j/(rho0*c0) *
             (2*np.pi*f*m_3 - rho0*c0/np.tan(2*np.pi*f*d_3/c0)))
    return (Z_res * Z_DBM) / (Z_res + Z_DBM)

def paris(Z, n=1000):
    theta = np.linspace(0, np.pi/2, n)
    integ = np.array([(1 - abs(((Z/np.cos(t))-1)/((Z/np.cos(t))+1))**2)
                      * np.sin(t)*np.cos(t) for t in theta])
    return float(np.clip(2*np.trapezoid(integ, theta), 0, 1))

bands      = [63, 125, 250, 500, 1000, 2000, 4000]
edges      = [44,  88, 177, 354,  707, 1414, 2828, 5656]
freqs      = np.linspace(20, 6000, 5000)

# Continuous
a_panel_c  = [paris(miki(f, sigma_2, l_2)) for f in freqs]
a_carpet_c = [paris(carpet(f))             for f in freqs]

# Band average
a_panel_b  = []
a_carpet_b = []
for i in range(len(bands)):
    mask = (freqs >= edges[i]) & (freqs < edges[i+1])
    a_panel_b.append( float(np.mean(np.array(a_panel_c)[mask])))
    a_carpet_b.append(float(np.mean(np.array(a_carpet_c)[mask])))

print("=== RESULTS ===")
print(f"{'Band':>6}  {'Wall':>6}  {'Panel':>6}  {'Carpet':>6}")
for i,f in enumerate(bands):
    print(f"{f:>6}Hz  {'0.10':>6}  {a_panel_b[i]:>6.4f}  {a_carpet_b[i]:>6.4f}")