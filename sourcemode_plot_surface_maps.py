
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from Constants.helpers import plot_directivity_contour, plot_phase_directivity_contour, p_to_SPL, read_force_file
from SourceMode.Configurations_NACA0012 import m_surface



# from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray
# SUFFIX = '_D20L20W20_D180'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray
# SUFFIX = '_D360_HR'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D180_v2 as sourceArray
# SUFFIX = 'D20L20_D180_v2'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180 as sourceArray
# SUFFIX = 'PARROT_D20L20_D180'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import D15L20W00_D180 as sourceArray # pick configuration
# SUFFIX = 'D15L20_D180'
# shape='D'

# from SourceMode.Configurations_NACA0012 import D15L20W00_D180 as sourceArray # pick configuration
# SUFFIX = 'D15L20_D180_R40'
# shape='D'

from SourceMode.Configurations_NACA0012 import D10L20W00_D180 as sourceArray # pick configuration
SUFFIX = 'D10L20_D180_R80'
shape='D'



# sourceArray.numerics['CompactnessCorrection'] = True
sourceArray.numerics['CompactnessCorrection'] = False


MODE = 'half'
FILE = 'TOTAL_DIR'

mss = np.arange(1, 11, 1)
# mss = np.array([5])


Nm = len(mss)
Ncomp = 8+3
Nr = 37
r_query = np.array([0.5, 0.8, 0.9]) * 0.1
Nr_query = len(r_query)
Fms = np.zeros((3, Nm, Nr_query, Ncomp), dtype=np.complex128) # store loading harmonics!
Fsectional = np.zeros((3, Nm, Nr, Ncomp), dtype=np.complex128) # store loading harmonics!


for index_m, m in enumerate(mss):
    print(f'saving diagrams for m={m}')
    ms = np.array([m])

    B = sourceArray.B
    NDIPOLES = sourceArray.Nsources

    r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

    if shape == "PARROT":

        rt, t =  np.loadtxt('./Data/Parrot2024/thrust_Npm.csv', skiprows=1, delimiter=',').T # radius/r1, thrust in Npm
        rq, q =  np.loadtxt('./Data/Parrot2024/torque_Nmpm.csv', skiprows=1, delimiter=',').T # radius/r1, torque in Nmpm

        q /= 1.125

        r_inner = sourceArray.seg_radius
        r1 = sourceArray.r1
        Fz = np.interp(r_inner/r1, rt, t) # same radial array
        Q = np.interp(r_inner/r1, rq, q) 
        Fphi = Q / r_inner

        TTARGET = 2.15 / sourceArray.B # Newtons
        QTARGET = 25 / 1000 / sourceArray.B # Newton-radian-meters
        Fz *= TTARGET / np.trapezoid(Fz, r_inner)  # rescale to target
        Fphi *= QTARGET / np.trapezoid(Fphi * r_inner, r_inner) # rescale to target

    D_bras = sourceArray.green.radius * 2
    g = -1 * sourceArray.green.origin[2]
    BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D_bras, g, steady_only=False) # compute loading on the fly, return PIN for reuse
    PIN = sourceArray.PIN


    # save gradients in the far-field (run once per observer and m)

    mplot = ms[0]
    ind_m = np.where(m_surface == ms[0])[0][0]


    gradG_surface = np.zeros((sourceArray.Nchildren, 3, m_surface.shape[0], sourceArray.green.getBoundaryPoints().shape[1] ,NDIPOLES,), dtype=np.complex128)
    G_surface = np.zeros((sourceArray.Nchildren, m_surface.shape[0], sourceArray.green.getBoundaryPoints().shape[1] ,NDIPOLES,), dtype=np.complex128)

    for index, sm in enumerate(sourceArray.children):

        gradG_surface[index, :, :, :, :] = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Nyi)
        print(f'pre-computing far-field gradients {index+1}')

        G_surface[index, :, :, :] = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Nyi)
        print(f'pre-computing far-field G {index+1}')


    print(f'parsed surface data')



    # manual computation!
    sourceArray.numerics['CompactnessCorrection'] = False

    pmB_loading = sourceArray.getPressure(sourceArray.green.getBoundaryPoints(), mplot, gradG_surface[:, :, ind_m:ind_m+1, :, :])
    pmB_thickness = sourceArray.getThicknessPressure(sourceArray.green.getBoundaryPoints(), mplot, G=G_surface[:, ind_m:ind_m+1, :, :])

    pmB_total = pmB_loading + pmB_thickness

    sourceArray.numerics['CompactnessCorrection'] = True
    pmB_loading_cp = sourceArray.getPressure(sourceArray.green.getBoundaryPoints(), mplot, gradG_surface[:, :, ind_m:ind_m+1, :, :])
    pmB_thickness_cp = sourceArray.getThicknessPressure(sourceArray.green.getBoundaryPoints(), mplot, G=G_surface[:, ind_m:ind_m+1, :, :])

    pmB_total_cp = pmB_loading_cp + pmB_thickness_cp


    index = np.where(PIN.k == mplot * B)[0][0] # index of the desired mode 

    PIN._numerics['only_linear'] = True 
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = False
    p_PIN_loading = PIN.getStrutPressureHarmonics()[:, index, :]

    PIN._numerics['only_linear'] = True 
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = False
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_thickness = PIN.getStrutPressureHarmonics()[:, index, :]

    PIN._numerics['only_linear'] = False 
    PIN._numerics['only_nonlinear'] = True
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_nonlinear = PIN.getStrutPressureHarmonics()[:, index, :]

    PIN._numerics['only_linear'] = True
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_total = PIN.getStrutPressureHarmonics()[:, index, :]


    PIN._numerics['only_linear'] = False
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_total_incl_nonlinear = PIN.getStrutPressureHarmonics()[:, index, :]

    # if does not exist, will throw an error, which is fin
    # pmB_total = pmB_loading

    z_edges = sourceArray.green.panel_z_edges
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    th_edges = sourceArray.green.panel_th_edges
    th_centers = (th_edges[:-1] + th_edges[1:]) / 2
    # TH, PHI = np.meshgrid(th_centers, z_centers, indexing='ij')
    PHI, TH = np.meshgrid(z_centers, th_centers, indexing='ij')

    # pin range
    thetab = PIN.theta_beam
    radius = PIN.seg_radius
    PHI_PIN, TH_PIN = np.meshgrid(radius, thetab, indexing='ij')

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    folder_name = f"./Figures/SurfacePressureComponents_{SUFFIX}_M{mplot}_RdBu"
    os.makedirs(folder_name, exist_ok=True)


    components = [
        # ==========================================================
        # 1) Loading contributions
        # ==========================================================
        {
            "name": "loading_scattering",
            "title": "Loading contribution",
            "data": pmB_loading[:, 0],
            'theta' : TH,
            'phi' : PHI,
            'color' : 'r',
            'linestyle': 'dashed',
            'marker' : 's',
            # 'label' : ''
        },

        # ==========================================================
        # 2) Thickness contributions
        # ==========================================================
        {
            "name": "thickness_scattering",
            "title": "Thickness contribution",
            "data": pmB_thickness[:, 0],
            'theta' : TH,
            'phi' : PHI,
            'color' : 'b',
            'linestyle': 'dashed',
            'marker' : 's',
        },
        {
            "name": "total_scattering",
            "data": pmB_total[:, 0],
            'theta' : TH,
            'phi' : PHI,
            'color' : 'k',
            'linestyle': 'dashed',
            'marker' : 's',
        },

                # ==========================================================
        # 1) Loading contributions
        # ==========================================================
        {
            "name": "loading_scattering_cp",
            "data": pmB_loading_cp[:, 0],
            'theta' : TH,
            'phi' : PHI,
            'color' : 'r',
            'linestyle': 'dotted',
            'marker' : 's',
            # 'label' : ''
        },

        # ==========================================================
        # 2) Thickness contributions
        # ==========================================================
        {
            "name": "thickness_scattering_cp",
            "data": pmB_thickness_cp[:, 0],
            'theta' : TH,
            'phi' : PHI,
            'color' : 'b',
            'linestyle': 'dotted',
            'marker' : 's',
        },

        {
            "name": "total_scattering_cp",
            "data": pmB_total_cp[:, 0],
            'theta' : TH,
            'phi' : PHI,
            'color' : 'k',
            'linestyle': 'dotted',
            'marker' : 's',
        },



            {
            "name": "loading_PIN",
            "data": p_PIN_loading.T,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
        'color' : 'r',
            'linestyle': 'solid',
            'marker' : '^',
        },        {
            "name": "thickness_PIN",
            "data": p_PIN_thickness.T,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
                    'color' : 'b',
            'linestyle': 'solid',
            'marker' : '^',
        },
        {
            "name": "nonlinear_PIN",
            "data": p_PIN_nonlinear.T,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
                'color' : 'm',
            'linestyle': 'solid',
            'marker' : '^',
        },
        {
            "name": "total_PIN",
            "data": p_PIN_total.T,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
                    'color' : 'k',
            'linestyle': 'solid',
            'marker' : '^',
        },

            {
            "name": "total_plus_nolinear_PIN",
            "data": p_PIN_total_incl_nonlinear.T,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
            'color' : 'c',
            'linestyle': 'solid',
            'marker' : '^',
        },



    ]

    rtip, rroot = 0.1, 0.1*0.16

    VMIN, VMAX = 80, 130
    levels = np.linspace(VMIN, VMAX, 21)
    levels_phase = np.linspace(-np.pi, np.pi, 21)


    mappables = {}
    reference_xrange = rtip - rroot
    reference_width = 6.0
    reference_height = 4.0
    reference_width_long = reference_width * 2 * rtip / reference_xrange
    RADIUS = 0.01
    fig3, ax3 = plt.subplots(figsize=(reference_width_long, reference_height)) # common radial plot!
    fig4, ax4 = plt.subplots(figsize=(reference_width_long, reference_height)) # common radial plot!

    for index_comp, comp in enumerate(components):

        TH = comp['theta'] # NPhi, NTh
        PHI = comp['phi'] # NPhi, NTh
        Z = comp['data'].reshape(TH.shape) # NPhi, NTh
        period = 2*np.pi

        TH = np.concatenate(
            [TH[:, -1:] - period, TH, TH[:, :1] + period],
            axis=1
        )

        PHI = np.concatenate(
            [PHI[:, -1:], PHI, PHI[:, :1]],
            axis=1
        )

        Z = np.concatenate(
            [Z[:, -1:], Z, Z[:, :1]],
            axis=1
        )

        print(f'component {comp['name']}, max SPL: {p_to_SPL(Z).max()}dB')


        HEIGHT = reference_height

        xrange = PHI.max() - PHI.min()

        WIDTH = reference_width * xrange / reference_xrange

        fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

        fig, ax, mappable = plot_directivity_contour(
            Phi=PHI,
            Theta=np.rad2deg(TH),
            magnitudes=Z,
            xlabel="z [m]",
            ylabel=r"$\theta$ [deg]",
            # title=comp["title"],
            levels=levels,
            fig=fig,
            ax=ax,
            cmap='jet'
        )
        ax.set_ylim(0, 360)

        # ax.set_xlim(rroot, rtip)
        ax.set_aspect((360 / (rtip - rroot))**(-1))
        if PHI.max() > rtip:
            ax.axvline(rroot, color='white', linestyle='dashed')
            ax.axvline(rtip, color='white', linestyle='dashed')

        # store mappable for colorbar later
        mappables[comp["name"]] = mappable

        fig.savefig(
            os.path.join(folder_name, f"p_surface_{comp['name']}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(WIDTH, HEIGHT))

        fig2, ax2, mappable_phase = plot_phase_directivity_contour(
            Phi=PHI,
            Theta=np.rad2deg(TH),
            magnitudes=Z,
            xlabel="z [m]",
            ylabel=r"$\theta$ [deg]",
            # title=comp["title"],
            levels=levels_phase,
            fig=fig2,
            ax=ax2
        )
        ax2.set_ylim(0, 360)


        # ax.set_xlim(rroot, rtip)
        ax2.set_aspect((360 / (rtip - rroot))**(-1))
        if PHI.max() > rtip:
            ax.axvline(rroot, color='white', linestyle='dashed')
            ax.axvline(rtip, color='white', linestyle='dashed')

        fig2.savefig(
            os.path.join(folder_name, f"p_surface_{comp['name']}_phase.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2)

        Z = Z.reshape(TH.shape) # of shape Nphi, Ntheta
        dtheta = TH[0, 1] - TH[0, 0]
        F_sectional_z = -RADIUS * np.sum(Z[:, 1:-1] * np.sin(TH[:, 1:-1]) * dtheta, axis=1, dtype=np.complex128) # of shape Nphi
        F_sectional_phi = RADIUS * np.sum(Z[:, 1:-1] * np.cos(TH[:, 1:-1]) * dtheta, axis=1, dtype=np.complex128) # of shape Nphi

        print(f'saving sectional loading at r_query={r_query}')
        Fms[1, index_m, :, index_comp] = (
            np.interp(r_query, PHI[:, -1], F_sectional_z.real)
            + 1j * np.interp(r_query, PHI[:, -1], F_sectional_z.imag)
        )

        Fms[2, index_m, :, index_comp] = (
            np.interp(r_query, PHI[:, -1], F_sectional_phi.real)
            + 1j * np.interp(r_query, PHI[:, -1], F_sectional_phi.imag)
        )
        print(f'saving net loading harmonics')
        # Fsectional[1, index_m, :, index_comp] = F_sectional_z
        # Fsectional[1, index_m, :, index_comp] = F_sectional_z

        np.save(f'./Data/current/surface_pressure/R_REF_{index_m}_{index_comp}.npy', PHI)
        np.save(f'./Data/current/surface_pressure/F_SECTIONAL_{index_m}_{index_comp}_Z.npy', F_sectional_z)
        np.save(f'./Data/current/surface_pressure/F_SECTIONAL_{index_m}_{index_comp}_PHI.npy', F_sectional_phi)



        COLOR, MARKER, LINESTYLE = comp['color'], comp['marker'], comp['linestyle']
        MARKER=None
        ax3.plot(PHI[:, 0], np.abs(F_sectional_z), color=COLOR, marker=MARKER, linestyle=LINESTYLE)
        ax4.plot(PHI[:, 0], np.abs(F_sectional_phi), color=COLOR, marker=MARKER, linestyle=LINESTYLE)


    fig = plt.figure(figsize=(1.5, 5))

    cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

    cbar = fig.colorbar(
        mappables["loading_scattering"],
        cax=cax,
    )

    cbar.set_label("SPL [dB]")

    fig.savefig(
        os.path.join(folder_name, "colorbar_spl.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plt.figure(figsize=(1.5, 5))

    cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

    cbar = fig.colorbar(
        mappable_phase,
        cax=cax,
    )

    cbar.set_label("SPL [dB]")

    fig.savefig(
        os.path.join(folder_name, "colorbar_phase.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    from matplotlib.lines import Line2D
    component_handles = [
        Line2D([0], [0], color='r', lw=2, label='Rotor Loading'),
        # Line2D([0], [0], color='g', lw=2, label='Unsteady Loading'),
        Line2D([0], [0], color='b', lw=2, label='Rotor Thickness'),
        # Line2D([0], [0], color='y', lw=2, label='Rotor Total'),
        Line2D([0], [0], color='m', lw=2, label='Non-linear'),
        Line2D([0], [0], color='c', lw=2, label='Loading+Thickness+Non-linear'),

        # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        Line2D([0], [0], color='k', lw=2, label='Loading + Thickness'),
    ]

    model_handles = [
    Line2D([0], [0], color='k',  linestyle='-',
           label='PIN'),
    Line2D([0], [0], color='k',  linestyle='--',
           label='SM'),
    ]

    ax3.grid()
    ax3.set_xlim((0, 2*rtip))
    
    ax3.axvline(rroot, color='k', alpha=0.7)
    ax3.axvline(rtip, color='k', alpha=0.7)

    leg = ax3.legend(handles=component_handles, loc='upper right')
    leg2 = ax3.legend(handles=model_handles,
                #  title='Model',
                 loc='upper left')
    ax3.add_artist(leg)
    ax3.add_artist(leg2)

    ax3.set_xlabel(fr'Radial Position $r$ $[m]$')
    ax3.set_ylabel(fr'Net Axial Loading $|\hat{{\boldsymbol{{F}}^z_k}}| [m]$')

    fig3.savefig(
        os.path.join(folder_name, "sectional_loading.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig3)

    ax4.grid()
    ax4.set_xlim((0, 2*rtip))
    
    ax4.axvline(rroot, color='k', alpha=0.7)
    ax4.axvline(rtip, color='k', alpha=0.7)

    leg = ax4.legend(handles=component_handles, loc='upper right')
    leg2 = ax4.legend(handles=model_handles,
                #  title='Model',
                 loc='upper left')
    ax4.add_artist(leg)
    ax4.add_artist(leg2)

    ax4.set_xlabel(fr'Radial Position $r$ $[m]$')
    ax4.set_ylabel(fr'Net Tangential Loading $|\hat{{\boldsymbol{{F}}^\phi_k}}| [m]$')

    fig4.savefig(
        os.path.join(folder_name, "sectional_loading_tan.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig4)

destination = './Data/current/surface_pressure/loading_harmonics_ALL.npy'
destination_r = './Data/current/surface_pressure/r_query_ALL.npy'
destination_m = './Data/current/surface_pressure/m_query_ALL.npy'

print(f'saving results of sectional loading at r_query={r_query} to {destination}')
np.save(destination, Fms)
np.save(destination_r, r_query)
np.save(destination_m, mss)

