from PotentialInteraction.placeholder import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors

fig, ax = plt.subplots(figsize=(4, 3))
plt.plot(R_RT_UDW, -UDW_EXACT, label='code',color='r')
plt.plot(R_RT_UDNS_V2, UZ_UDNS_V2, label='paper', color='b')
ax.grid()
ax.set_xlabel('$r/r_t$')
ax.set_ylabel('$U_z$')
ax.legend()
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 3))
plt.plot(R_RT_EXACT, T_EXACT, label='code',color='r')
plt.plot(R_RT_THRUST, THRUST, label='paper', color='b')
ax.grid()
ax.set_xlabel('$r/r_t$')
ax.set_ylabel('$T [N]$')
ax.legend()
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 3))
plt.plot(R_RT_EXACT, Q_EXACT, label='code',color='r')
plt.plot(R_RT_TAN, TAN, label='paper', color='b')
ax.grid()
ax.set_xlabel('$r/r_t$')
ax.set_ylabel('$Q [N]$')
ax.legend()
plt.tight_layout()
plt.show()
plt.close()