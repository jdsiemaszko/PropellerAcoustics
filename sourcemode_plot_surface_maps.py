from scattering_vs_PIN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from Constants.helpers import plot_directivity_contour

mplot = 1

# save gradients in the far-field (run once per observer and m)
ind_m = np.where(m_surface == mplot)[0][0]
gradG_surface_arr = np.zeros((sourceArray.seg_radius.shape[0], 3, 1,
                               sourceArray.green.getBoundaryEvaluationPoints().shape[1], NDIPOLES), dtype=np.complex128)
for index, sm in enumerate(sourceArray.children):
    gradG_surface_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy')[:, ind_m, :, :].reshape(
        3, 1, sourceArray.green.getBoundaryEvaluationPoints().shape[1], NDIPOLES
    )
     # shape (3, Nm, Nz, Ny)
print(f'parsed surface data')

# manual computation!
pmB = sourceArray.getPressure(sourceArray.green.getBoundaryPoints(), mplot, gradG_surface_arr)
z_edges = sourceArray.green.panel_z_edges
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
th_edges = sourceArray.green.panel_th_edges
th_centers = (th_edges[:-1] + th_edges[1:]) / 2
TH, PHI = np.meshgrid(th_centers, z_centers, indexing='ij')

fig, ax = plot_directivity_contour(Theta=np.rad2deg(TH), Phi=PHI, magnitudes=pmB, ylabel=r'$\theta$ [deg]', xlabel='$z$ [m]', title=f'Surface Pressure $p_{{{mplot*B}}}$ (dB)')

ax.scatter(PHI, np.rad2deg(TH), color='k', marker='x',alpha=0.25)
print(f'maximum surface SPL: {np.max(p_to_SPL(pmB))} dB')
plt.show()


# sourceArray.plotSurfacePressure(m=mplot, gradG_surface=gradG_surface_arr, extent_z=[0.016, 0.1])
# plt.show()


