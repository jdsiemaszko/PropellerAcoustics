#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib.pyplot as plt

def get_p_grid_hist(r_query, th_query, r, theta, values, areas):
    """
    Extract the area-weighted mean value on a regular (r, theta) grid.

    Parameters
    ----------
    r_query : (Nr,) array
        Radial grid centers.
    th_query : (Nth,) array
        Angular grid centers.
    r : (N,) array
        Radial coordinates of scattered points.
    theta : (N,) array
        Angular coordinates of scattered points.
    values : (N,) array
        Values at scattered points.
    areas : (N,) array
        Area associated with each scattered point.

    Returns
    -------
    values_grid : (Nth, Nr) array
        Area-weighted mean value on the query grid.
    """

    dr = r_query[1] - r_query[0]
    dth = th_query[1] - th_query[0]

    # Bin edges corresponding to the cell boundaries
    r_edges = np.concatenate((
        [r_query[0] - dr / 2],
        0.5 * (r_query[:-1] + r_query[1:]),
        [r_query[-1] + dr / 2]
    ))

    th_edges = np.concatenate((
        [th_query[0] - dth / 2],
        0.5 * (th_query[:-1] + th_query[1:]),
        [th_query[-1] + dth / 2]
    ))

    # Sum of value * area in each bin
    value_area_sum, _, _ = np.histogram2d(
        r, theta,
        bins=[r_edges, th_edges],
        weights=values * areas
    )

    # Divide by cell area
    values_grid = value_area_sum / (dr * dth)

    return values_grid

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
    area = np.concatenate((area, area, area))

    # r_u = np.unique(r)
    # theta_u = np.unique(theta)

    # pressure: quantity to interpolate!
    pA = pressure

    # Structured mesh
    r_grid_outer = np.linspace(r.min(), r.max(), Nr+1)
    theta_grid_outer = np.linspace(-np.pi, np.pi, Ntheta+1)
    r_grid = (r_grid_outer[1:] + r_grid_outer[:-1]) / 2
    theta_grid = (theta_grid_outer[1:] + theta_grid_outer[:-1]) / 2

    R, TH = np.meshgrid(r_grid, theta_grid, indexing="ij")

    # Interpolate scattered data
    p_grid = griddata(
        points=(r, theta),
        values=pA,
        xi=(R.ravel(), TH.ravel()),
        method=interpolation,
        fill_value=np.nan,
    )

    # p_hist = get_p_grid_hist(r_grid, theta_grid, r, theta, pA, area)


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

    np.save("./Data/current/pressure_maps_strut_iLES.npy", 2 * pressure_maps) # 2 times because each cell contains half the pressure and is counted TWICE
    np.save("./Data/current/rs_strut_iLES.npy", r)
    np.save("./Data/current/thetas_maps_strut_iLES.npy", theta)
