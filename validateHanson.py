from PotentialInteraction.main import HansonModel
from PotentialInteraction.near_field import NearFieldHansonModel
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DATAPATH = './Vella2026Model/Validation/harmonics_ISAE2_D20_L20-1.h5'

NSEG = 20

datafile = h5py.File(DATAPATH, 'r')

R = np.array(datafile['R'][0]) # 1.62
SPL = np.array(datafile['SPL']) #
phi = np.array(datafile['phi'][0]) # pi/2
theta = np.array(datafile['theta'][:, 0]) # 0 -> pi
r = np.array(datafile['r'][:, 0]) # discretization in the radial dir
ddr = r[5] - r[4]
r_bounds = np.concatenate(([r[0]-ddr], r))
dr = np.diff(r_bounds)
BPF = np.array(datafile['BPF'][:, 0]) # in Hz?
BOmega = BPF[0] #HZ !!!!

B = 2 # ?

Omega = BOmega /B * 2 * np.pi

Fk_phi_i = np.array(datafile["Fk_phi_imag"])
Fk_phi_r = np.array(datafile["Fk_phi_real"])

Fk_z_i = np.array(datafile["Fk_z_imag"])
Fk_z_r = np.array(datafile["Fk_z_real"])

Nr, Nk = Fk_phi_i.shape # (20, 40)

Fk_z = Fk_z_r + 1j * Fk_z_i
Fk_phi = Fk_phi_r + 1j * Fk_phi_i
# Fk_phi = np.zeros_like(Fk_z, dtype=np.complex128)
Fk_r = np.zeros_like(Fk_z, dtype=np.complex128)

loading = np.stack(
    (Fk_r.T, Fk_z.T, Fk_phi.T)
) # (3, Nk, Nr)
loading = np.concatenate((np.zeros((3, 1, Nr)), loading), axis=1)
loading_per_unit_span =loading/ dr[None, None, :] # force PER UNIT LENGTH

p_i = np.array(datafile['p_imag'])
p_r = np.array(datafile['p_real'])

p = p_r + 1j * p_i

# TEST:
# loading = np.conjugate(loading)
RADIUS_MULT = 1.0
x = np.array([
    R * np.ones_like(theta) * RADIUS_MULT,
    theta, 
    phi * np.ones_like(theta)
])



m = np.round(BPF / BOmega)
MAXK = Nk

hm = HansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), chord_m = 0.025 * np.ones(NSEG+1),
                            radius_m= r_bounds, B=B, # B=2 seems to be "most correct!"
                            Dcylinder_m=0.02, Lcylinder_m=0.02, Omega_rads=Omega, rho_kgm3=1.2, c_ms=340., kmax=MAXK)


# hm_near = NearFieldHansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), chord_m = 0.025 * np.ones(NSEG+1),
#                             radius_m= r_bounds, B=B, # B=2 seems to be "most correct!"
#                             Dcylinder_m=0.02, Lcylinder_m=0.02, Omega_rads=Omega, rho_kgm3=1.2, c_ms=340., kmax=MAXK)

# hm.seg_radius= r
# hm.dr = np.ones_like(r) # overwrite the radial computations, take loading without normalizing
# p_model, xmodel =  hm.getHansonPressure(x, m, B=hm.B, Omega=hm.Omega, loading=loading[:, :MAXK+1, :], nb=1, multiplier=hm.B)

p_model, xmodel =  hm.getHansonPressure(x, m, B=hm.B, Omega=hm.Omega, loading=loading_per_unit_span[:, :MAXK+1, :], nb=1, multiplier=hm.B)
# p_model_nf, xmodel_nf =  hm_near.getHansonPressure(x, m, B=hm.B, Omega=hm.Omega, loading=
#                                                    loading_per_unit_span[:, :MAXK+1, :],
#                                                 #    np.conjugate(loading_per_unit_span[:, :MAXK+1, :]),
#                                                      nb=1, multiplier=hm.B)


# p_test, xtest =getHansonPressure_POINTWISE(x[:, 0], m[0], radius)
pref = 20e-6  # reference pressure (20 µPa)
def spl_from_p(pp):
    return 10 * np.log10(np.abs(pp)**2 / (pref * pref)) + 10 * np.log10(2) # note: using one side of a two-sided spectrum?

def plotPressureSpectrum(fig, ax, mm, pmcurrent, pmprevious, pmnearfield=None):
    if pmnearfield is not None:
        ax.plot(mm, spl_from_p(pmnearfield), label='current (General)', marker='^', color='g')
    ax.plot(mm, spl_from_p(pmcurrent), label='current (Far-field)', marker='s', color='r')
    ax.plot(mm, spl_from_p(pmprevious), label='validation (Far-field)', marker='o', color='b')


    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major')
    ax.legend()

    ax.set_xscale('log')
    ax.set_xlabel('$f^+$')
    ax.set_ylabel('SPL w.r.t 20e-6 Pa')
    plt.tight_layout()


fig, ax = plt.subplots(figsize=(4, 3))
# plot_directivity(fig, ax, x, spl_from_p(p_model)[:, m_to_plot-1])
# plot_directivity(fig, ax2, x, spl_from_p(p)[:, m_to_plot-1])

for color, mode in zip(['r', 'b', 'g'], [1, 5, 10]):
    # ax.plot(np.rad2deg(theta), spl_from_p(p_model_nf)[:, mode-1] , color=color, label=f'm={mode}', marker='s', linestyle='dashed')
    ax.plot(np.rad2deg(theta), spl_from_p(p_model)[:, mode-1] , color=color, marker='o', linestyle='dotted')
    ax.plot(np.rad2deg(theta), spl_from_p(p)[:, mode-1], color=color, marker='x', markersize=10)
    # ax.plot(np.rad2deg(theta), SPL[:, mode-1], color=color)

ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Modal SPL [dB]')
ax.grid()
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 3))
# plot_directivity(fig, ax, x, spl_from_p(p_model)[:, m_to_plot-1])
# plot_directivity(fig, ax2, x, spl_from_p(p)[:, m_to_plot-1])

for color, mode in zip(['r', 'b', 'g'], [1, 5, 10]):
    # ax.plot(np.rad2deg(theta), (-np.rad2deg(np.angle(p_model))[:, mode-1]+180) % 360, color=color, label=f'm={mode}', marker='s', linestyle='dashed')
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_model))[:, mode-1] % 360, color=color, label=f'm={mode}', marker='o', linestyle='dotted')
    # ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_model_nf))[:, mode-1] % 360, color=color, marker='s', linestyle='dashed')

    ax.plot(np.rad2deg(theta), (np.rad2deg(np.angle(p))[:, mode-1] % 360), color=color, marker='x', markersize=10)

ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Phase [deg]')
ax.grid()
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 3))
# plot_directivity(fig, ax, x, spl_from_p(p_model)[:, m_to_plot-1])
# plot_directivity(fig, ax2, x, spl_from_p(p)[:, m_to_plot-1])

for color, mode in zip(['r', 'b', 'g'], [1, 5, 10]):
    # ax.plot(np.rad2deg(theta), np.real(p_model)[:, mode-1] - np.real(p)[:, mode-1], color=color, label=f'm={mode}', marker='s')
    # ax.plot(np.rad2deg(theta), np.imag(p_model)[:, mode-1] - np.imag(p)[:, mode-1], color=color, marker='^', linestyle='dashed')

    ax.plot(np.rad2deg(theta), spl_from_p(p_model)[:, mode-1] - spl_from_p(p)[:, mode-1], color=color, label=f'm={mode}', marker='s')

ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Error [dB]')
ax.grid()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(4, 3))
# plot_directivity(fig, ax, x, spl_from_p(p_model)[:, m_to_plot-1])
# plot_directivity(fig, ax2, x, spl_from_p(p)[:, m_to_plot-1])

for color, mode in zip(['r', 'b', 'g'], [1, 5, 10]):
    # ax.plot(np.rad2deg(theta), np.real(p_model)[:, mode-1] - np.real(p)[:, mode-1], color=color, label=f'm={mode}', marker='s')
    # ax.plot(np.rad2deg(theta), np.imag(p_model)[:, mode-1] - np.imag(p)[:, mode-1], color=color, marker='^', linestyle='dashed')

    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_model))[:, mode-1] % 180 - np.rad2deg(np.angle(p))[:, mode-1] % 180, color=color, label=f'm={mode}', marker='s')

ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Phase Error [deg]')
ax.grid()
plt.tight_layout()
plt.show()

# for index_x in [0, 5, 12, 18, 24, 36]:
#     print(xmodel[:, index_x])
#     fig, ax = plt.subplots(figsize=(4, 3))
#     plotPressureSpectrum(fig, ax, m, p_model[index_x, :], p[index_x, :], pmnearfield=p_model_nf[index_x, :])
#     ax.set_xlabel('$f^+$')
#     ax.set_ylabel('SPL [dB]')
#     plt.tight_layout()
#     plt.show()
#     plt.close()


ROBS = 1.62 * RADIUS_MULT
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
hm.plot3Ddirectivity(fig, ax, m=5, R=ROBS,
                        # valmax=30, valmin=-10,
                        Nphi=36*2, Ntheta=18*2,
                        # mode='beam',
                            # mode='total',
                        mode='blade'
                        )
plt.show()
plt.close(fig)

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# hm_near.plot3Ddirectivity(fig, ax, m=5, R=ROBS,
#                         # valmax=30, valmin=-10,
#                         Nphi=36*2, Ntheta=18*2,
#                         # mode='beam',
#                             # mode='total',
#                         mode='blade'
#                         )
# plt.show()
# plt.close(fig)

