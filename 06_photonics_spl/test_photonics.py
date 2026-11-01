#!/usr/bin/env python3
"""
Photonic Inverse Design, GDSII Layout & Metasurface Simulation Suite
1. GDSFactory + gdstk + KLayout: Generates a silicon photonic ring resonator GDSII layout.
2. Ceviche: 2D FDFD Maxwell solver + automatic differentiation sensitivity gradient for inverse design.
3. GRCWA: Rigorous Coupled-Wave Analysis for a periodic photonic/plasmonic grating.
"""
import numpy as np
import gdsfactory as gf
import gdstk
import ceviche
from ceviche import fdfd_ez, jacobian
import autograd.numpy as anp
import grcwa

def build_photonic_gds_layout(out_gds="pic_ring_resonator.gds"):
    gf.gpdk.PDK.activate()
    c = gf.Component("silicon_photonic_ring")
    ring = c << gf.components.ring_single(gap=0.2, radius=10.0, length_x=4.0, length_y=0.6)
    c.write_gds(out_gds)
    lib = gdstk.read_gds(out_gds)
    print(f"[Photonics / GDSFactory] v{gf.__version__} + gdstk v{gdstk.__version__} generated {out_gds} ({len(lib.cells)} cells)")

def run_ceviche_inverse_design():
    # 2D FDFD grid (60 x 60 pixels, dl = 40 nm, lambda = 1550 nm)
    Nx, Ny = 60, 60
    dl = 40e-9
    omega = 2 * np.pi * 299792458.0 / 1.55e-6
    npml = 10

    # Silicon waveguide core (eps_r = 12.0) in SiO2 cladding (eps_r = 2.1)
    eps_r = np.full((Nx, Ny), 2.1)
    eps_r[:, 26:34] = 12.0

    # Line source at left waveguide entrance
    source = np.zeros((Nx, Ny), dtype=complex)
    source[npml + 3, 26:34] = 1.0

    simulation = fdfd_ez(omega, dl, eps_r, [npml, npml])

    # Objective function: maximize electric field intensity |Ez|^2 at right output port
    def objective(eps_flat):
        eps_2d = eps_flat.reshape((Nx, Ny))
        sim = fdfd_ez(omega, dl, eps_2d, [npml, npml])
        _, _, Ez = sim.solve(source)
        return anp.sum(anp.abs(Ez[Nx - npml - 4, 26:34])**2)

    val = objective(eps_r.flatten())
    grad_fn = jacobian(objective, mode="reverse")
    grad = grad_fn(eps_r.flatten())
    print(f"[Photonics / Ceviche] FDFD inverse-design solved: Output intensity = {val:.3e}, max |dJ/deps| = {np.max(np.abs(grad)):.3e}")

def run_grcwa_metasurface():
    # 2D silicon pillar metasurface on glass at lambda = 1.55 um
    nG = 101  # 11 kept only 9 harmonics and overestimated R ~12x
    L1 = [0.8, 0.0]
    L2 = [0.0, 0.8]
    freq = 1.0 / 1.55
    Nx, Ny = 50, 50
    obj = grcwa.obj(nG, L1, L2, freq, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(0.1, 1.0)      # Superstrate (Air)
    obj.Add_LayerGrid(0.3, Nx, Ny)      # Patterned Metasurface Layer
    obj.Add_LayerUniform(0.1, 2.1)      # Substrate (SiO2)
    obj.Init_Setup()

    ep_grid = np.full((Nx, Ny), 1.0)
    ep_grid[15:35, 15:35] = 12.0        # Silicon nanopillar
    obj.GridLayer_geteps(ep_grid.flatten())
    obj.MakeExcitationPlanewave(p_amp=1.0, p_phase=0.0, s_amp=0.0, s_phase=0.0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    print(f"[Photonics / GRCWA] Periodic metasurface RCWA solved: R = {R:.4f}, T = {T:.4f}, R+T = {R+T:.4f}")

if __name__ == "__main__":
    build_photonic_gds_layout()
    run_ceviche_inverse_design()
    run_grcwa_metasurface()
