"""
comparing the Hanson and source-mode implementations of thickness noise,
here for:
NACA0012 rotor, , twist=10deg, B=2
"""
from SourceMode.SourceMode import SourceModeArray, NACA0012_T10_SOURCEMODE_HALFCYLINDER, NACA0012_T10_SOURCEMODE_FF
from PotentialInteraction.beam_to_blade import BladeLoadings, NACA0012_T10_PIN
from TailoredGreen.HalfCylinderGreen import CG_NACA0012_T10
from Hanson.far_field import NACA0012_T10_HANSON
import matplotlib.pyplot as plt
import numpy as np
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotSelf()
# plt.show()

# BLH = NACA0012_T10_PIN.getBladeLoadingHarmonics()
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.BLH = BLH # overwrite BLH's with computed data
# print(BLH[:, 1:, :].max())
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotFarFieldPressure(m=2, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='st')
# plt.show()
# NACA0012_T10_SOURCEMODE_HALFCYLINDER.plotFarFieldPressure(m=2, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='dt')
# plt.show()


NPHI = 72
NTHETA = 36


NACA0012_T10_SOURCEMODE_FF.plotSelf()
plt.show()

NACA0012_T10_SOURCEMODE_FF.plotFarFieldPressure(m=1, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='tl')
plt.show()

NACA0012_T10_HANSON.plot3Ddirectivity(m=1, mode='rotor', loadings = NACA0012_T10_PIN.getBladeLoadingHarmonics(),
                                      Nphi=NPHI, Ntheta=NTHETA,)
plt.show()


NACA0012_T10_SOURCEMODE_FF.plotFarFieldPressure(m=1, R=1.62, Nphi=NPHI, Ntheta=NTHETA, mode='dt')
plt.show()

NACA0012_T10_HANSON.plot3Ddirectivity(m=1, mode='thickness', chord=0.025 * np.ones(20), t_c = 0.122 * np.ones(20),
                                      Nphi=NPHI, Ntheta=NTHETA,)
plt.show()


