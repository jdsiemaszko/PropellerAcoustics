import numpy as np
import h5py

def getGojonData(ind_theta, ind_phi, datadir, D, L, shape='D', B=2, RPM=8000):
    if shape != 'A':
        casefile = f'ISAE_2_{shape}{int(1000*D)}_L{int(1000*L)}'
    else:
        casefile = f'ISAE_2_airfoil_8000'
    def load_h5(filename):
        return h5py.File(filename, "r")

    with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
        g = f[casefile]
        freq = np.array(g["frequency_Hz"])
        ap = g["Autopower"]

        theta_exp = np.array(g["theta_deg"])[0][ind_theta] # polar
        radius = g["radius_m"][0][0] # float

        BPF = B * RPM / 60
        if shape != 'A':
            phi_exp = np.array(g["phi_deg"])[0][ind_phi] # azimuth
            data = ap[f"Autopower_RPM_{RPM}_Pa2"][:, ind_theta, ind_phi] # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
        else: # different labels for airfoil data :(
            phi_exp = np.array(g["phi_L20_deg"])[0][ind_phi] # azimuth
            data = ap[f"Autopower_arfoil20_Pa2"][:, ind_theta, ind_phi] #


    theta = 90 - theta_exp
    phi = 180 - phi_exp
    print(f'Theta_exp = {theta_exp} deg, Phi_exp = {phi_exp} deg')
    print(f'Theta = {theta} deg, Phi = {phi} deg')
    x_cart = np.array([
        radius * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
        radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
        radius * np.cos(np.deg2rad(theta)),
    ]).reshape((3, 1))

    return data, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp, casefile