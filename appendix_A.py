import numpy as np
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


L = 0.02
D = 0.02
R = D / 2
rho0 = 1.2
t_c = 0.0803
c = 0.025
B=2
Omega = 8000/60*2*np.pi

radius, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

phi = np.linspace(- 1 * np.pi, 1 * np.pi, 1000)
z0 = 1j * L + phi[:, None] * radius[None, :] # Nphi, Nr
z0prime = np.conj(z0)

Lprime = Fz # in N per meter
Gamma = Lprime / rho0 / Omega / radius # in m^2/s
# Gamma=0
mu = t_c * c**2 / 2 / np.pi * Omega * radius
# mu = 0
Uinf = np.sqrt(Lprime / 4 / np.pi / rho0 / radius)

# TODO: do this numerically

def u_baseflow(k):
    if k==0:
        return np.conj(Uinf)
    elif k==2:
        return -Uinf
    else:
        return 0

# def u_vortex_array


A = -1j * Uinf / Omega / radius
B = Gamma / Omega / radius / R
C = mu / Omega / radius / R**2

p_acoustic =   1j * Gamma * Omega * radius / 2 / np.pi / R * (R / z0prime)**2 - 2 * mu * Omega * radius / R**2 * (R / z0prime)**3
# TODO: need higher orders?
# p_dynamic = (
#         Uinf * (-1j * Gamma / 2 / np.pi / R * (R / z0)**2 - 2 * mu / R**2 * (R / z0)**3)
#         + Gamma**2 / 4 / np.pi**2 / R**2 * (R / z0)**2 * (R / z0prime)
#         - 2 * 1j * mu * Gamma / np.pi / R**3 * (R/z0)**3 * (R/z0prime)
#         + 1j * mu * Gamma / 2 / np.pi / R**3 * (R/z0)**2 * (R/z0prime)**2
        # -2 * mu ** 2 / R**4 * (R/z0)**3 * (R/z0prime)**2
#         )


u0 = np.conj(np.conj(Uinf) - 1j * Gamma / 2 / np.pi / z0 - mu / z0**2)
um1 = np.conj(-1j * Gamma * R / 2 / np.pi / z0**2 - 2 * mu * R / z0**3)
u2 = np.conj(-Uinf - 1j * Gamma / 2 / np.pi / z0prime - mu / z0prime**2)
u3 = np.conj(-1j * Gamma * R / 2 / np.pi / z0prime**2 + 2 * mu * R / z0prime**3)
p_dynamic = 0.5 * (
    u0 * np.conj(um1) 
    + u3 * np.conj(u2)
                   )

p_acoustic *= rho0
p_dynamic *= -rho0

# p_acoustic *= R**2 * B**2 / 4 / radius[None, :] ** 2 * np.sin(B/2 * (phi[:, None] + 1j * L / radius[None, :]))**(-2)

ratio = np.abs(p_dynamic / p_acoustic)

# ratio = p_dynamic
# ratio = p_acoustic

# Actual model
from PotentialInteraction.PIN import PotentialInteraction, DistributedPIN
r_inner = radius
dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40
ms = np.arange(1, 16, 1)
pin = PotentialInteraction(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    t_c = np.ones_like(r_outer) * 0.0803,
    U0_mps=np.vstack([np.zeros_like(Uinf), -Uinf]),
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=2,
    Dcylinder_m=0.02,
    Lcylinder_m=0.02,
    Omega_rads=8000/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': 360, 'Nthetab': 72, 'include_vortex_sources':True, 'include_thickness_sources':True, 'Nvortices': 1}
)
pin._numerics['gamma_steady'] = True
pin._numerics['only_linear']  = True
F_linear = pin.getStrutLoading()
pin._numerics['only_linear'] =  False
pin._numerics['only_nonlinear']  = True
F_nonlinear = pin.getStrutLoading() # 3, Nt, Nr

F_total = F_linear + F_nonlinear

ratio_total = (F_nonlinear[1] * 1j + F_nonlinear[2]) / (F_linear[1] * 1j + F_linear[2])

# -------------------------------------------------------
# Query radii
# -------------------------------------------------------
r_query = np.array([0.1 * 0.3 ,0.1 * 0.5, 0.1 * 0.7,  0.1 * 0.9])   # example radii [m]

fig, ax = plt.subplots(figsize=(7, 4))

for rq, color in zip(r_query, ['r', 'g', 'b', 'y']):
    for element, x_array, kwargs in zip(
        # [ratio, ratio_total],
        # [-2 * np.pi * R * p_acoustic, F_linear[1] * 1j - F_linear[2]],
        [-2 * np.pi * R * p_dynamic, F_nonlinear[1] * 1j - F_nonlinear[2]],
        # [-2 * np.pi * R * (p_dynamic+p_acoustic), F_total[1] * 1j - F_total[2]],

        # [p_acoustic, p_dynamic],
        [phi, pin.phi],
        [{'linestyle':'solid', 'color' : color}, {'linestyle':'dashed', 'color' : color}]):
        if rq < radius.min() or rq > radius.max():
            print(f"Warning: r = {rq:.4f} m outside interpolation range.")
            continue

        # interpolate along the radius axis
        ratio_interp = np.array([
            np.interp(rq, radius, ratio_i)
            for ratio_i in element
        ])

        ax.plot(
            x_array,
            # np.abs(z0),
                np.abs(ratio_interp), label=fr"$r={rq:.3f}\,\mathrm{{m}}$", **kwargs)


ax.set_yscale('log')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$|p_\mathrm{dynamic}/p_\mathrm{acoustic}|$")
ax.legend(title="Radius")
ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()