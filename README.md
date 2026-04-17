Code implementation of several models in Propeller Acoustics

Modules include:

1) Hanson - implementing the Hanson model for propagating loading harmonics to far-field pressure. Both the standard far-field formulation from Hanson & Patrzych (1993) as well as the general extension from Roger & Moreau (2008) are included.

2) Tailored Green - module responsible for computing the Tailored and free-field Green's functions. instances for Free-field, Infinite Cylinder, and Semi-infinite Cylinder included.

3) Source Mode - a general propagator for rotating (dipole and monopole) sources from Roger & Moreau 2008. Equivalent to the general model from above in the free-field, but allows for the inclusion of scattering effects if interfaced with a Tailored Green's function. This model is interfaced with a TailoredGreen instance, allowing for interchanging of the scattering geometry.

4) PotentialInteraction - analytical model for computing propeller and cylinder surface loadings based on Vella et al. 2026, Wu et al. 2022

Full "blind" model is then constructed by interfacing the modules:

4 -> 1 for a free-field model

4, 2 -> 3 for a tailored model

For usage, see scripts "Hanson_example.py". Mind that other scripts may be deprecated.
