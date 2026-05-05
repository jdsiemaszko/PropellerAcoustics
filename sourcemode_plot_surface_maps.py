from scattering_vs_PIN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData

mplot = 5

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


sourceArray.plotSurfacePressure(m=mplot, gradG_surface=gradG_surface_arr, extent_z=[0.016, 0.1])
plt.show()


