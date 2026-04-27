from PotentialInteraction.ConformalPIN import HypotrochoidalPIN
import numpy as np
from Constants.helpers import read_force_file, p_to_SPL, spl_from_autopower, plot_BPF_peaks
import matplotlib.pyplot as plt
from Hanson.far_field import HansonModel
from Constants.data_assim import getGojonData
from Constants.helpers import p_to_SPL

r_inner, Fz, Fphi = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40
ms = np.arange(1, 16, 1)

B=2
RPM = 8000
RPM_ref = 8000
# CT = 0.0625 # approximate only

NPHI = 360
NTHETA = 360
# rescale loading to approximately match loading at this RPM
Fz *= (RPM/RPM_ref)**2
Fphi = (RPM/RPM_ref)**2

pin_triangle = HypotrochoidalPIN(
    Nsides=3, theta0=np.deg2rad(210), rho_corner=0.5,
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=B,
    Dcylinder_m=0.01,
    Lcylinder_m=0.02,
    Omega_rads=RPM/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': NPHI, 'Nthetab':NTHETA}
)

pin_square = HypotrochoidalPIN(
    Nsides=4, theta0=np.deg2rad(45), rho_corner=0.3,
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=B,
    Dcylinder_m=0.01 * 1.163 * 0.94,
    Lcylinder_m=0.02,
    Omega_rads=RPM/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': NPHI, 'Nthetab':NTHETA}
)

pin_circle = HypotrochoidalPIN(
    Nsides=100, theta0=np.deg2rad(0), rho_corner=0.01,
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=B,
    Dcylinder_m=0.01,
    Lcylinder_m=0.02,
    Omega_rads = RPM / 60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': NPHI, 'Nthetab':NTHETA}
)

hanson = HansonModel(
    radius_m=r_outer,
    axis= np.array([0.0, 0.0, 1.0]), # z-direction propeller...
    origin= np.array([0.0, 0.0, 0.0]), # ... at z=0
    B=B,
    nb=1,
    Omega_rads= RPM / 60 * 2 * np.pi
)


fig, ax = pin_triangle.plotZ()
fig, ax = pin_square.plotZ(fig, ax)
fig, ax = pin_circle.plotZ(fig, ax)
plt.show()

pin_triangle.plotDownwashInRotorPlane()
plt.show()

pin_square.plotDownwashInRotorPlane()
plt.show()

pin_circle.plotDownwashInRotorPlane()
plt.show()

ind_theta = 6
ind_phi = 9
D = 0.01
L = 0.02
dir  = './Experimental/dataverse_files'

# data_T, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp = getGojonData(ind_theta, ind_phi, dir, D, L, shape='T', B=2, RPM = 8000)
# data_S, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp  = getGojonData(ind_theta, ind_phi, dir, D, L, shape='S', B=2, RPM = 8000)
# data_D, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp = getGojonData(ind_theta, ind_phi, dir, D, L, shape='D', B=2, RPM = 8000)

fig, ax = plt.subplots(figsize=(7, 4))

for shape, color, marker, pin_model in zip(['T', 'S', 'D'], ['r', 'g', 'b'], ['^', 's', 'o'], [pin_triangle, pin_square, pin_circle]):
    data, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp, casefile = getGojonData(ind_theta, ind_phi, dir, D, L, shape=shape, B=B, RPM = RPM)


    ax.plot(freq[0] / BPF, spl_from_autopower(data), 
            # label=f"{shape}{1000*L:.0f}",
            label = casefile,
              color=color, alpha=0.75)
    fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                            plot_kwargs={
                                'color':color,
                                'linestyle':'dashed',
                                'alpha':0.75,
                                'marker' : marker
                            })
    
    strut_loading = pin_model.getStrutLoadingHarmonics()
    p_model, _ = hanson.getPressureStator(x_cart, ms * B, Fstator=strut_loading)

    ax.plot(ms, p_to_SPL(p_model)[0, :], color=color, marker=marker, label=f'model ({shape})')

ax.legend(ncol=1)
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
plt.xlim(0.1, 100)
plt.ylim(0, 70)
plt.tight_layout()
plt.show()