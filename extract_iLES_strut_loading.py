#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib.pyplot as plt


Nr = 40
Ntheta = 36


def read_surface_file(filename):
    """
    Read a surface data file.

    Returns
    -------
    x : (N,)
    y : (N,)
    z : (N,)
    pressure : (N,)
    area : (N,)
    """

    data = []

    with open(filename, "r") as f:
        for line in f:

            cols = line.split()

            # Skip header lines
            if len(cols) < 12:
                continue

            try:
                row = [float(v) for v in cols]
                data.append(row)
            except ValueError:
                continue

    data = np.asarray(data)

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    pressure = data[:, 4]
    area = data[:, 8]


    return x, y, z, pressure, area


def compute_pressure_map(filename,
                         Nr=40,
                         Ntheta=36,
                         interpolation="linear"):
    """
    Compute p(r,theta) on a structured grid.

    Returns
    -------
    p_grid : (Nr, Ntheta)
    r_grid : (Nr,)
    theta_grid : (Ntheta,)
    """

    x, y, z, pressure, area = read_surface_file(filename)
    D = z.max() - z.min()
    z -= z.min() + D/2
    # Surface coordinates
    r = x
    theta = np.arctan2(z, y)

    # wrap the theta axis to avoid extrapolating
    r = np.concatenate((r, r, r))
    theta = np.concatenate((theta, theta-2*np.pi, theta+2*np.pi))
    pressure = np.concatenate((pressure, pressure, pressure))

    # r_u = np.unique(r)
    # theta_u = np.unique(theta)

    # pressure: quantity to interpolate!
    pA = pressure

    # Structured mesh
    r_grid = np.linspace(r.min()+1e-12, r.max()-1e-12, Nr)
    theta_grid = np.linspace(-np.pi, np.pi, Ntheta)

    R, TH = np.meshgrid(r_grid, theta_grid, indexing="ij")

    # Interpolate scattered data
    p_grid = griddata(
        points=(r, theta),
        values=pA,
        xi=(R.ravel(), TH.ravel()),
        method=interpolation,
        fill_value=np.nan,
    )

    return p_grid, r_grid, theta_grid


def process_timesteps(file_list,
                      Nr=40,
                      Ntheta=36):
    """
    Process all timestep files.

    Parameters
    ----------
    file_list : list[str]

    Returns
    -------
    pressure_maps : ndarray
        Shape (Nt, Nr, Ntheta)
    """

    pressure_maps = []
    rs = []
    thetas = []

    for f in file_list:
        print(f'processing file: {f}')
        p_grid, r_grid, theta_grid = compute_pressure_map(
            f,
            Nr=Nr,
            Ntheta=Ntheta
        )
        pressure_maps.append(p_grid)
        rs.append(r_grid)
        thetas.append(theta_grid)

    return np.asarray(pressure_maps), np.asarray(rs), np.asarray(thetas)


if __name__ == "__main__":

    # Example:
    files = sorted(Path("./Data/Vella2026/strut_loading").glob("*"))

    pressure_maps, r, theta = process_timesteps(
        files,
        Nr=40,
        Ntheta=36
    )

    print("Output shape:", pressure_maps.shape)
    # (Nt, 40, 36)

    np.save("./Data/current/pressure_maps_strut_iLES.npy", pressure_maps)
    np.save("./Data/current/rs_strut_iLES.npy", r)
    np.save("./Data/current/thetas_maps_strut_iLES.npy", theta)
