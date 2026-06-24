import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Hanson.far_field import HansonModel
from Constants.helpers import p_to_SPL

source = './Data/current/surface_pressure/'
Fz_iLES = np.load(source + 'Fzfreq_iLESTEST.npy')[:, :] # Nfreq, Nrq
Fphi_iLES = np.load(source + 'Fphifreq_iLESTEST.npy')[:, :] # Nfreq, Nrq
freqs_iLES = np.load(source + 'freqs_iLESTEST.npy') # Nfreq

rs = np.load(source + 'rquery_iLESTEST.npy') # Nrquery
# ms = np.load(source + 'm_query.npy') #Nm
ms = np.arange(1, 41, 1)
B = 2
# ks = ms * B
# ks = ms
ks = np.arange(1, 41, 1)
Omega = 8000/60 * 2 * np.pi

# indices of frequencies matching ks
idx = np.array([
    np.argmin(np.abs(freqs_iLES / Omega * 2 * np.pi - k))
    for k in ks
])

# optional sanity check
if not np.allclose(freqs_iLES[idx]/ Omega * 2 * np.pi, ks):
    print("Warning: some ks do not exactly exist in freqs_iLES")

Fz_iLES = Fz_iLES[idx, :]
Fphi_iLES = Fphi_iLES[idx, :]
freqs_iLES = freqs_iLES[idx]

dr = rs[1] - rs[0]
r_outer = np.concatenate((rs-dr/2, np.array([rs[-1] + dr/2])))
han = HansonModel(
    radius_m=r_outer,
    Omega_rads=Omega,
    B=2,
)

x = np.array([[0], [1.62], [0]]).reshape((3, 1))
m = np.arange(1, 21, 1)

Nk, Nr = Fz_iLES.shape
Fstator = np.zeros((3, Nk+1, Nr), dtype=np.complex128)
Fstator[1, 1:, :] = Fz_iLES
Fstator[2, 1:, :] = Fphi_iLES


# KK = 20
# Fstator[1, 1:KK, :] = Fz_iLES[:KK-1, :]
# Fstator[2, 1:KK, :] = Fphi_iLES[:KK-1, :]

p_strut_iLES, _ = han.getPressureStator(x, m*B, Fstator)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(m, p_to_SPL(p_strut_iLES[0]), color='r', marker='s', label='current: iLES+Lowson')

x = [541.0601440034313, 806.6805936947535, 1072.0707442861774, 1341.1082933032599, 1608.079689786384, 1882.0817418495642, 2137.1210082627385, 2397.5260991125942, 2657.305340691887, 2927.4637137071377, 3205.628832163832]
y = [42.83813871998747, 54.32372682301254, 58.82483625027744, 60.06652087452509, 59.13525420909874, 57.583147363042315, 55.4102003363558, 52.46119775514734, 49.51219517393888, 44.54545667694829, 39.57871391697022]
# ax.plot(m[1:12], y, color='b', marker='*', label='Vella et al. 2026: iLES+Farrasat-1A')
from Constants.const import SPLSHIFT
# ax.plot(m[1:12], y - SPLSHIFT, color='g', marker='^', label='Vella et al. 2026: iLES+Farrasat-1A - 10log(2)')

x = [263.42378591440087, 528.3502536834308, 797.5598405263813, 1065.8787938559105, 1328.6958409493413, 1590.4178546951227, 1849.2791918366468, 2125.4739501057784, 2373.093994245315, 2680.473247583605, 2941.1275820604737, 3189.9091488418653]
y = [46.71772444533767, 44.5733019481741, 53.45733019481741, 57.43982431010006, 59.12472679922461, 58.6652067911856, 57.28665097408705, 54.989059347928965, 52.38512104974487, 48.708969399469744, 44.5733019481741, 39.978114488839445]
ax.plot(m[0:12], y, color='b', marker='*', label='Vella et al. 2026: iLES+Farrasat-1A')
ax.plot(m[0:12], y-SPLSHIFT, color='g', marker='p', label='Vella et al. 2026: iLES+Farrasat-1A-10log(2)')

ax.grid(which='both')

ax.set_xscale('log')
ax.set_xlabel(f'f/BPF')
ax.set_ylabel(f'SPL[dB]')
ax.legend()
plt.show()