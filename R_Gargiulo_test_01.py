import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def read_tecplot_block_dat(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # --- Parse variable names ---
    var_line = next(l for l in lines if l.startswith("VARIABLES"))
    variables = re.findall(r'"([^"]+)"', var_line)

    # --- Parse zone size (I dimension) ---
    zone_line = next(l for l in lines if l.startswith("ZONE"))
    I = int(re.search(r'I\s*=\s*(\d+)', zone_line).group(1))

    nvar = len(variables)

    # --- Collect all numeric tokens after ZONE ---
    start_idx = lines.index(zone_line) + 1
    tokens = []
    for l in lines[start_idx:]:
        tokens.extend(l.split())

    tokens = iter(tokens)

    # --- Read block data ---
    data = {}
    for v in variables:
        vals = []
        while len(vals) < I:
            try:
                vals.append(float(next(tokens)))
            except StopIteration:
                raise RuntimeError("Unexpected end of file while reading data")
        data[v] = np.array(vals)

    return data


data = read_tecplot_block_dat("Data/Gargiulo2026/raw_01.dat")

print(data.keys())

r = data["R_middle"] # segment centers, in units meter

dr = np.diff(r)
dr = np.append(dr, dr[-1]) # segment lengths

r_outer = np.append(r-dr/2, r[-1]+dr[-1]/2) # segment edges

F_radial = data["Fr_fR"] # in units Newton
F_tan = data["Ft_fR"]
F_ax = data["Tx_fR"]
F_amplitude = data["F1P_fR"]

F_radial[-5] = 0
F_tan[-5] = 0 # erreneous datapoint
F_ax[-5] = 0 
F_amplitude[-5] = 0

print(np.sum(F_ax))

fig, ax = plt.subplots()
ax.plot(r, F_radial, color='r', marker='x', label='$F_r$')
ax.plot(r, F_tan, color='g', marker='x', label='$F_t$')
ax.plot(r, F_ax, color='b', marker='x', label='$F_x$')
ax.plot(r, F_amplitude, color='k', marker='x', label='$|F|$',linestyle='dashed')

ax.set_ylabel('loading [N/m]')
ax.set_xlabel('radius [m]')


ax.grid()
ax.legend()
plt.show()


# fill the loading-per-unit-span array (steady loading only)
Fblade = np.zeros((3, 1, len(r)), dtype=np.complex128)
Fblade[0, 0, :] = F_radial / dr # not used
Fblade[1, 0, :] = F_ax / dr # unit Newton per meter
Fblade[2, 0, :] = -F_tan / dr # force oriented backwards -> positive in our sign convention

twist = np.arctan2(-F_tan, F_ax) # not exact, but should be good enough, replace by actual geometry if possible
twist = np.append(twist, twist[-1])

chord = 0.025 * np.ones_like(r_outer) # REPLACE BY ACTUAL CHORD LENGTH PER SEGMENT!

from Hanson.far_field import HansonModel

ROBS = 10. # observation radius, in meters

# Initialize Module
hm = HansonModel(twist_rad = twist,
                chord_m = chord,
                radius_m = r_outer, # blade radius stations [m] of size Nr + 1\
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
                B=8, # number of blades
                Omega_rads = 9505 / 60 * 2 * np.pi, # rotation speed [rad/s]
                rho_kgm3 = 1.2, # fluid density [kg/m^3]
                c_mps = 340., # speed of sound [m/s]
                nb = 0 # number of beams (irrelevant)
                )


# 1) Plot directivity (3D)

for m in [1, 2]:
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111, projection="3d")
    hm.plot3Ddirectivity(
        fig=fig,
        ax=ax1,
        m=m, # harmonic to plot
        R=ROBS, # observation radius
        Nphi=36*2, # plotting params
        Ntheta=18*2,
        valmin=40,
        valmax=65,
        title='far-field',
        mode='rotor', # 'rotor'
        loadings=Fblade # blade loading harmonics
    )
    plt.tight_layout()
    plt.show()

# 2) Plot directivity (2D contour)
# not yet implemented!

# 3) Plot spectrum at a point
fig, ax = plt.subplots()
hm.plotPressureSpectrum(fig=fig, ax=ax, 
                        x =np.array([1.0, 0.0, 0.0]).T, # position to plot the spectrum at
                        m = np.arange(1, 11, 1), # modes to compute
                        loadings=Fblade # blade loading harmonics
                            )
plt.tight_layout()
plt.show()