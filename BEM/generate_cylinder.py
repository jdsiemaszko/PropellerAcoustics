import numpy as np


def generate_cylinder_geo(
        filename="cylinder.geo",
        radius=1.0,
        Rmax_factor=5.0,
        Nazim=18,
        Nax=64,
        mode="uniform",
        geom_ratio=1.01
):
    """
    Generate a Gmsh .geo file for a structured cylindrical surface mesh.
    """

    Rmax = Rmax_factor * radius

    # -------------------------
    # Axial discretization
    # -------------------------
    if mode == "geometric":
        g = geom_ratio
        if np.isclose(g, 1.0):
            dz = np.full(Nax, Rmax / Nax)
        else:
            a = Rmax * (g - 1.0) / (g**Nax - 1.0)
            dz = a * g**np.arange(Nax)

        z_edges = np.concatenate(([0.0], np.cumsum(dz)))

    elif mode == "uniform":
        z_edges = np.linspace(0.0, Rmax, Nax + 1)

    else:
        raise ValueError(f"mode {mode} not recognized")

    # -------------------------
    # Azimuthal discretization
    # -------------------------
    th_edges = np.linspace(0.0, 2.0 * np.pi, Nazim + 1)

    # -------------------------
    # Write GEO file
    # -------------------------
    with open(filename, "w") as f:

        f.write('// Structured cylindrical surface\n')
        f.write('SetFactory("Built-in");\n\n')

        point_id = 1
        line_id = 1
        line_loop_id = 1
        surface_id = 1

        point_map = {}

        # ---- Create points ----
        for i, z in enumerate(z_edges):
            for j, th in enumerate(th_edges):
                x = radius * np.cos(th)
                y = radius * np.sin(th)
                f.write(
                    f"Point({point_id}) = {{{x}, {y}, {z}, 1.0}};\n"
                )
                point_map[(i, j)] = point_id
                point_id += 1

        f.write("\n")

        # ---- Create quad surfaces ----
        for i in range(Nax):
            for j in range(Nazim):

                p1 = point_map[(i, j)]
                p2 = point_map[(i+1, j)]
                p3 = point_map[(i+1, j+1)]
                p4 = point_map[(i, j+1)]

                l1 = line_id
                f.write(f"Line({l1}) = {{{p1}, {p2}}};\n")
                line_id += 1

                l2 = line_id
                f.write(f"Line({l2}) = {{{p2}, {p3}}};\n")
                line_id += 1

                l3 = line_id
                f.write(f"Line({l3}) = {{{p3}, {p4}}};\n")
                line_id += 1

                l4 = line_id
                f.write(f"Line({l4}) = {{{p4}, {p1}}};\n")
                line_id += 1

                f.write(
                    f"Line Loop({line_loop_id}) = {{{l1}, {l2}, {l3}, {l4}}};\n"
                )

                f.write(
                    f"Plane Surface({surface_id}) = {{{line_loop_id}}};\n"
                )

                line_loop_id += 1
                surface_id += 1

        f.write("\n")
        f.write(f'Physical Surface("CylinderSurface") = {{1:{surface_id-1}}};\n')

    print(f"GEO file written to: {filename}")


if __name__ == "__main__":
    generate_cylinder_geo(
        filename="cylinder.geo",
        radius=1.0,
        Rmax_factor=5.0,
        Nazim=18,
        Nax=64,
        mode="uniform",        # or "geometric"
        geom_ratio=1.01
    )
