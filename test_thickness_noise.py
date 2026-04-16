"""
comparing the Hanson and source-mode implementations of thickness noise,
here for:
NACA0012 rotor, , twist=10deg, B=2
"""
import h5py

from SourceMode.SourceMode import SourceModeArray, NACA0012_T10_SOURCEMODE_HALFCYLINDER, NACA0012_T10_SOURCEMODE_FF
from PotentialInteraction.beam_to_blade import BladeLoadings, NACA0012_T10_PIN
from TailoredGreen.HalfCylinderGreen import CG_NACA0012_T10
from Hanson.far_field import NACA0012_T10_HANSON
import matplotlib.pyplot as plt
import numpy as np
from Constants.helpers import plot_BPF_peaks, spl_from_autopower, p_to_SPL
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotSelf()
# plt.show()

# BLH = NACA0012_T10_PIN.getBladeLoadingHarmonics()
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.BLH = BLH # overwrite BLH's with computed data
# print(BLH[:, 1:, :].max())
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotFarFieldPressure(m=2, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='st')
# plt.show()
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotFarFieldPressure(m=2, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='dt')
# plt.show()


NPHI = 36*2
NTHETA = 18*2


fig, ax = NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotSelf()
ax.set_axis_off()

plt.show()

# NACA0012_T10_SOURCEMODE_FF.plotFarFieldPressure(m=1, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='tl')
# plt.show()

# NACA0012_T10_HANSON.plot3Ddirectivity(m=1, mode='rotor', loadings = NACA0012_T10_PIN.getBladeLoadingHarmonics(),
#                                       Nphi=NPHI, Ntheta=NTHETA, R=1.62)
# plt.show()


# NACA0012_T10_SOURCEMODE_FF.plotFarFieldPressure(m=1, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='dt')
# plt.show()

# NACA0012_T10_HANSON.plot3Ddirectivity(m=1, mode='thickness', chord=0.025 * np.ones(20), t_c = 0.122 * np.ones(20),
#                                       Nphi=NPHI, Ntheta=NTHETA, R=1.62)
# plt.show()


# PARSE EXPERIMENTAL

ind_theta = 6        # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
D_bras = 0.02
g = 0.02
B=2
RPM = 8000
casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

def load_h5(filename):
    return h5py.File(filename, "r")
with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
    g = f[casefile]
    freq = np.array(g["frequency_Hz"])
    ap = g["Autopower"]

    phi_exp = np.array(g["phi_deg"])[0][ind_phi] # azimuth
    theta_exp = np.array(g["theta_deg"])[0][ind_theta] # polar
    radius = g["radius_m"][0][0]# float

    BPF = B * RPM / 60
    data = ap[f"Autopower_RPM_{RPM}_Pa2"][:, ind_theta, ind_phi] # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
theta = 90 - theta_exp
phi = 180 - phi_exp
print(f'Theta_exp = {theta_exp} deg, Phi_exp = {phi_exp} deg')
print(f'Theta = {theta} deg, Phi = {phi} deg')
x_cart = np.array([
    radius * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.cos(np.deg2rad(theta)),
]).reshape((3, 1))

ms = np.arange(1, 11, 1)

p_t_sm = NACA0012_T10_SOURCEMODE_FF.getThicknessPressureDirect(x_cart, ms)
p_t_han, _ = NACA0012_T10_HANSON.getThicknessNoiseRotor(x_cart, ms, chord=0.025 * np.ones(20), thickness_to_chord = 0.0809 * np.ones(20))

# ------------------------------------------------ scattered pressure (reuse solutions)
# extract and rearrange
MODE = 'half'
SUFFIX = '_HIGHRES'
NDIPOLES = 72
Nr = 20
Omega = RPM * 2 * np.pi / 60
c0 = 340
rho0 = 1.

# for index, sm in enumerate(NACA0012_T10_SOURCEMODE_HALFCYLINDER.children):
#     print(f'pre-computing surface Greens functions: {index+1} of {Nr}')
#     source_positions = sm.dipole_positions
    # G_surface = sm.green.getGreenAtSurface(source_positions, ms*B * Omega / c0) # shape (Nm, Nz, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy', G_surface)

# save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(NACA0012_T10_SOURCEMODE_HALFCYLINDER.children):

    G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
    print(f'pre-computing far-field gradients {index+1}')

    G = sm.getScatteringGreen(x_cart, ms*B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}.npy', G)


G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
for index, sm in enumerate(NACA0012_T10_SOURCEMODE_HALFCYLINDER.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}.npy')


p_scattered = NACA0012_T10_SOURCEMODE_HALFCYLINDER.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
# np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered)
# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental", color='k', alpha=0.75)
fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'k',
                             'linestyle':'dashed',
                             'alpha':0.75
                         })
# ax.plot(ms, SPL_rotor_S, label=f"Model (rotor, steady loading)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Model (rotor, unsteady loading)", color='g', marker='^')
# ax.plot(ms, SPL_rotor_thickness, label=f"Model (rotor, thickness)", color='b', marker='+')
# ax.plot(ms, SPL_rotor_total, label=f"Model (rotor, total)", color='y', marker='*', linestyle='dashed')
# ax.plot(ms, SPL_beam, label=f"Model (beam, loading)", color='m', marker='o')
# ax.plot(ms, SPL_total, label=f"Model (total)", color='k', marker='s', linestyle='dashed')
ax.plot(ms, p_to_SPL(p_t_sm[0]), label=f"Source-Mode (Direct)", color='r', marker='^')
ax.plot(ms, p_to_SPL(p_scattered), label=f"Source-Mode (Scattered)", color='g', marker='o')

ax.plot(ms, p_to_SPL(p_t_han[0]), label=f"Hanson", color='b', marker='s')



ax.legend()
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)


plt.xlim(0.1, 100)
plt.ylim(0, 70)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ms, np.rad2deg(np.angle(p_t_sm[0])), label=f"Source-Mode (Direct)", color='r', marker='^')
# ax.plot(ms, p_to_SPL(p_t_sm_scat[0]), label=f"Source-Mode (Scattered)", color='g', marker='o')
ax.plot(ms, np.rad2deg(np.angle(p_t_han[0])), label=f"Hanson", color='b', marker='s')



ax.legend()
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("Angle (deg)")
ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)


plt.xlim(0.1, 100)
plt.ylim(-180, 180)
plt.tight_layout()
plt.show()
