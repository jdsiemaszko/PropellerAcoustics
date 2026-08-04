import numpy as np
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

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

phi = np.linspace(- 1 * np.pi, 1.5 * np.pi, 1000)
z0 = 1j * L + phi[:, None] * radius[None, :] # Nphi, Nr
z0prime = np.conj(z0)

Lprime = Fz # in N per meter
Gamma = Lprime / rho0 / Omega / radius # in m^2/s, Nr
# Gamma=0
mu = -t_c * c**2 / 2 / np.pi * Omega * radius
# mu = 0
Uinf = - 1j * np.sqrt(B * Lprime / 4 / np.pi / rho0 / radius)
# Uinf = np.zeros_like(Uinf)

def ubar_baseflow(k):
    if k==0:
        return np.conj(Uinf) + 0j
    elif k==2:
        return -Uinf  + 0j
    else:
        return 0j

def ubar_vortex(k):
    if k <=0:
        return 1j * Gamma[None, :] / 2 / np.pi * R**(-k) / z0**(-k+1)
    elif k>=2:
        return 1j * Gamma[None, :] / 2 / np.pi * R**(k-2) / z0prime**(k-1)
    else:
        return 0

def ubar_doublet(k):
    if k <=0:
        return -(-k+1) * mu * R**(-k) / z0**(-k+2)
    elif k>=2:
        return (k-1) * mu * R**(k-2) / z0prime**(k)
    else:
        return 0j

def u_total(k):
    return np.conj(ubar_baseflow(k) + ubar_vortex(k) + ubar_doublet(k))

def uu1_total(k):
    #TODO: why the conjugate?
    return np.conj(u_total(k+1) * np.conj(u_total(k)))

# def u_vortex_array


A = Uinf / Omega / radius
B = Gamma / Omega / radius / R
C = mu / Omega / radius / R**2

#TODO: signs!
p_acoustic = -1j * Gamma * Omega * radius / 2 / np.pi / R * (R / z0prime)**2 - 2 * mu * Omega * radius / R**2 * (R / z0prime)**3

# p_dynamic = (
#         Uinf * (-1j * Gamma / 2 / np.pi / R * (R / z0)**2 - 2 * mu / R**2 * (R / z0)**3)
#         + Gamma**2 / 4 / np.pi**2 / R**2 * (R / z0)**2 * (R / z0prime)
#         - 2 * 1j * mu * Gamma / np.pi / R**3 * (R/z0)**3 * (R/z0prime)
#         + 1j * mu * Gamma / 2 / np.pi / R**3 * (R/z0)**2 * (R/z0prime)**2
        # -2 * mu ** 2 / R**4 * (R/z0)**3 * (R/z0prime)**2
#         )


# ks = np.arange(-100, 100, 1)

ks = np.arange(-3, 6, 1)
uu1s = np.array([uu1_total(k) for k in ks])
p_dynamic_8_terms = 0.5 * np.sum(uu1s, axis=0) * -rho0

ks = np.arange(-10, 21, 1)
uu1s = np.array([uu1_total(k) for k in ks])
p_dynamic_20_terms = 0.5 * np.sum(uu1s, axis=0) * -rho0

p_dynamic = 0.5 * (
    uu1_total(-1) + uu1_total(2)
                   )

# u0 = np.conj(np.conj(Uinf) - 1j * Gamma / 2 / np.pi / z0 - mu / z0**2)
# um1 = np.conj(-1j * Gamma * R / 2 / np.pi / z0**2 - 2 * mu * R / z0**3)
# u2 = np.conj(-Uinf - 1j * Gamma / 2 / np.pi / z0prime - mu / z0prime**2)
# u3 = np.conj(-1j * Gamma * R / 2 / np.pi / z0prime**2 + 2 * mu * R / z0prime**3)



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
    U0_mps=np.vstack([np.zeros_like(np.imag(Uinf)), np.imag(Uinf)]),

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
    numerics={'Nphi': 360, 'Nthetab': 72,
            'include_vortex_sources':True,
            'include_thickness_sources':True,
            # 'include_thickness_sources':False,
                   'Nvortices': 1}
)
pin._numerics['gamma_steady'] = True
pin._numerics['only_linear']  = True
F_linear = pin.getStrutLoading()

pin._numerics['only_linear'] =  False
pin._numerics['only_nonlinear']  = True
F_nonlinear = pin.getStrutLoading() # 3, Nt, Nr

# pin._numerics['only_linear'] =  False
# pin._numerics['only_nonlinear']  = False
# F_total = pin.getStrutLoading() # 3, Nt, Nr

F_total = F_linear + F_nonlinear
ratio_total = (F_nonlinear[1] * 1j - F_nonlinear[2]) / (F_linear[1] * 1j - F_linear[2])

# -------------------------------------------------------
# Query radii
# -------------------------------------------------------

r_query = np.array([0.1 * 0.3 ,0.1 * 0.5, 0.1 * 0.7, 0.1 * 0.9]) # example radii [m]
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ------------------------------------------------------------------
# Specify zoom region
x0, x1 = -np.pi/8, np.pi/8    
y0, y1 = 0.055, 0.143   
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

# Create inset
axins = inset_axes(ax, width="35%", height="55%", loc="upper right")

colors = ['r', 'g', 'b', 'y']
for rq, color in zip(r_query, colors):
    for element, x_array, kwargs in zip(
        [abs(ratio),
         abs(p_dynamic_8_terms/p_acoustic),
         abs(p_dynamic_20_terms/p_acoustic),
         abs(ratio_total)],
        [phi, phi, phi, pin.phi],
        [{'linestyle':'dotted',  'color':color},
         {'linestyle':'dashdot', 'color':color},
         {'linestyle':'dashed',  'color':color},
         {'linestyle':'solid',   'color':color}]
    ):

        if rq < radius.min() or rq > radius.max():
            continue

        ratio_interp = np.array([
            np.interp(rq, radius, ratio_i)
            for ratio_i in element
        ])

        # Main axes
        ax.plot(x_array, ratio_interp, **kwargs)

        # Inset
        axins.plot(x_array, ratio_interp, **kwargs)

# Zoom limits
axins.set_xlim(x0, x1)
axins.set_ylim(y0, y1)

# Optional inset cosmetics
# axins.set_xticks([])
# axins.set_yticks([])
axins.grid()

mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

# ax.indicate_inset_zoom(axins, edgecolor="black", )

# ------------------------------------------------------------------
# Legends (unchanged)
radius_handles = [
    Line2D([0], [0], color=color, lw=2,
           label=fr"$r/r_\mathrm{{tip}}={rq/0.1:.1f}$")
    for rq, color in zip(r_query, colors)
]
legend_radius = ax.legend(handles=radius_handles,
                          title="Station",
                        #   loc="upper left",
                          loc="lower right",

                          ncols=2)
ax.add_artist(legend_radius)

dataset_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='dotted',
           label="2"),
               Line2D([0], [0], color='k', lw=2, linestyle='dashdot',
           label="8"),
               Line2D([0], [0], color='k', lw=2, linestyle='dashed',
           label="20"),
    Line2D([0], [0], color='k', lw=2, linestyle='solid',
           label=r"$\rightarrow\infty$"),
]
ax.legend(handles=dataset_handles,
          title=r"No. of Terms in $(p^\mathrm{dynamic})_1^{\theta}$",
          loc="lower left",
          ncols=2)
# ------------------------------------------------------------------

ax.set_xlabel(r"$\phi=\Omega t$")
ax.set_ylabel(r"$\left|(p^\mathrm{dynamic})^{\theta}_1/(p^\mathrm{acoustic})^{\theta}_1\right|$")
ax.set_xticks(np.arange(-np.pi, 1.6 * np.pi, np.pi/2))
ax.set_xticks(np.arange(-np.pi, 3 * np.pi, np.pi/8), minor=True)
ax.set_xticklabels(
    [
    r"$-\mathrm{\pi}$",
    r"$-\mathrm{\pi}/2$",
    r"$0$",
    r"$\mathrm{\pi}/2$",
    r"$\mathrm{\pi}$",
    r"$3\mathrm{\pi}/2$",
]
    # [f"${val/np.pi:.1f}\pi$" for val in np.arange(-np.pi, 3 * np.pi, np.pi/2)]
                   )

plt.minorticks_on()
ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)
ax.set_ylim(0.04, 0.15)
ax.set_xlim(-np.pi, 1.5 * np.pi)
plt.tight_layout()
plt.show()
fig.savefig('./Figures/Appendix_A.pdf', dpi=300, bbox_inches='tight')