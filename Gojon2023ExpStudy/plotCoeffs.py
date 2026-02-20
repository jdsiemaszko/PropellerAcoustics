import numpy as np
import matplotlib.pyplot as plt

# read file (skip header row)
datadir = './dataverse_files'
M = np.loadtxt(f'{datadir}/ISAE_2_D10_L20_static.txt', delimiter=',', skiprows=1)

# assign variables
rpm = M[:, 0]
ct  = M[:, 1]
cp  = M[:, 2]

# plot
plt.plot(rpm, ct, 'bs', markersize=10, label='Ct')
plt.plot(rpm, cp, 'rs', markersize=10, label='Cp')

# tune plot
plt.axis([2000, 10000, 0.0, 0.08])
plt.tick_params(labelsize=20)
plt.xlabel('RPM', fontsize=20)
plt.ylabel('Ct, Cp', fontsize=20)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
