from scattering_vs_PIN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


ind_theta = 6        # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

def load_h5(filename):
    return h5py.File(filename, "r")
with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
    g = f[casefile]
    freq = np.array(g["frequency_Hz"])
    ap = g["Autopower"]

    phi_exp = np.array(g["phi_deg"])[0][ind_phi] # azimuth
    theta_exp = np.array(g["theta_deg"])[0][ind_theta] # polar
    radius = g["radius_m"][0][0] # float

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

Nr = len(r0)



ms = np.arange(1, 13, 1) # harmonics to extract
pSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH_S
                                       )[0][0]

pUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH_US
                                       )[0][0]

ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r0), 0.122 * np.ones_like(r0))[0][0] # NACA0012
BL  =  beam_l.getBeamLoadingHarmonics()
pmB_model_beam = han.getPressureStator(x_cart, ms*B, BL)[0][0]

pmB_model_rotor_total = pSmB_model_rotor + pUSmB_model_rotor + ptmB_model_rotor
# pmB_model_total = pSmB_model_rotor + pUSmB_model_rotor + ptmB_model_rotor + pmB_model_beam # assuming coherent
pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent


p_scattered = sourceArray.getScatteredPressure(x_cart, ms)[0]
np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered)

p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy')

SPL_rotor_S = p_to_SPL(pSmB_model_rotor)
SPL_rotor_US = p_to_SPL(pUSmB_model_rotor)
SPL_rotor_total = p_to_SPL(pmB_model_rotor_total)
SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
SPL_beam = p_to_SPL(pmB_model_beam)

SPL_scattered = p_to_SPL(p_scattered)



SPL_total = p_to_SPL(pmB_model_total)
# SPL_total = p_to_SPL(p_rms_total) # same computation

fig, ax = plt.subplots(figsize=(7, 4))
# ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental", color='k', alpha=0.75)
# fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
#                          plot_kwargs={
#                              'color':'k',
#                              'linestyle':'dashed',
#                              'alpha':0.75
#                          })
# ax.plot(ms, SPL_rotor_S, label=f"Model (rotor, steady loading)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Model (rotor, unsteady loading)", color='g', marker='^')
# ax.plot(ms, SPL_rotor_thickness, label=f"Model (rotor, thickness)", color='b', marker='+')
# ax.plot(ms, SPL_rotor_total, label=f"Model (rotor, total)", color='y', marker='*', linestyle='dashed')
# ax.plot(ms, SPL_beam, label=f"Model (beam, loading)", color='m', marker='o')
# ax.plot(ms, SPL_total, label=f"Model (total)", color='k', marker='s', linestyle='dashed')
ax.plot(ms, SPL_beam, label=f"Beam PIN", color='r', marker='o')
ax.plot(ms, SPL_scattered, label=f"Blade Scattering", color='b', marker='s')




ax.legend()
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
plt.xlim(0.1, 100)
plt.ylim(0, 70)
plt.tight_layout()
plt.show()
