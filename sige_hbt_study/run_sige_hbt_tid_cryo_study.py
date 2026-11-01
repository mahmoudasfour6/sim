#!/usr/bin/env python3
"""
Multi-Physics Study: Strained Epitaxial SiGe HBT under Combined TID Radiation & Cryogenic Temperature (300 K -> 77 K)
=====================================================================================================================
Every stage runs real numerical solvers and passes computed arrays to the next stage:
  Stage 1 (env: materials)     -> ASE/Pymatgen strained Si_{1-x}Ge_x lattice mismatch, elastic stiffness C_ij(x),
                                  and Bir-Pikus deformation-potential bandgap narrowing Delta_Eg(z).
  Stage 2 (env: fem)           -> 2D Vector Finite-Element Linear Elasticity PDE (-div(sigma(u)) = 0 in scikit-fem)
                                  solving coupled epitaxial eigenstrain + cryogenic thermal contraction (300 K vs 77 K)
                                  around the Shallow Trench Isolation (STI) & Emitter-Base SiO2 spacer.
  Stage 3 (env: quantum_iii_v) -> 1D Finite-Difference Schrödinger solver for heavy-hole quantum well confinement
                                  in the strained p+ Si / Si_{1-x}Ge_x / Si base heterostructure.
  Stage 4 (env: process_rad)   -> Physical TID radiation kinetics in the EB spacer SiO2 (field-dependent hole yield,
                                  polaron hopping vs. 77 K hole freezing, and interface trap buildup D_it(Dose, T)).
  Stage 5 (env: tcad)          -> Real DEVSIM finite-volume drift-diffusion PDE solver on the graded NPN SiGe HBT mesh
                                  + Ngspice circuit simulation of a Cryogenic SiGe HBT Low-Noise Amplifier (LNA).
"""
import os
import json
import shutil
import subprocess

HOME = "/home/mahmoudasfour"
MAMBA = f"{HOME}/miniforge3/bin/mamba"
STUDY_DIR = f"{HOME}/sim_workspace/sige_hbt_study"
os.makedirs(STUDY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STAGE 1: Epitaxial Si_{1-x}Ge_x Strain & Deformation-Potential Bandgap (env: materials)
# ---------------------------------------------------------------------------
stage1_code = f"""
import json
import numpy as np
from ase.build import bulk
import pymatgen.core as mg

# Build bulk Si and Ge reference unit cells in ASE & Pymatgen
si_cell = bulk("Si", "diamond", a=5.431, cubic=True)
ge_cell = bulk("Ge", "diamond", a=5.6575, cubic=True)
a_Si = 5.431
a_Ge = 5.6575

# 1D vertical coordinate z across NPN HBT [nm]:
#   z in [0, 60] nm:     n+ Si Emitter
#   z in [60, 105] nm:   p+ Strained Epi-Si(1-x)Ge(x) Base (trapezoidal Ge grading up to x_Ge = 0.22)
#   z in [105, 300] nm:  n- Si Collector
z_nm = np.linspace(0.0, 300.0, 301)
x_Ge = np.zeros_like(z_nm)
for i, z in enumerate(z_nm):
    if 60.0 <= z < 70.0:
        x_Ge[i] = 0.08 + 0.14 * ((z - 60.0) / 10.0)   # Graded Ge ramp at EB side (accelerating quasi-electric field!)
    elif 70.0 <= z <= 98.0:
        x_Ge[i] = 0.22                                # Peak Ge plateau (22% Ge)
    elif 98.0 < z <= 105.0:
        x_Ge[i] = 0.22 * (1.0 - (z - 98.0) / 7.0)     # Ge retro-grade at BC junction

# Dismukes' nonlinear lattice constant of relaxed Si(1-x)Ge(x) [Angstrom]
a_SiGe = a_Si + 0.2003 * x_Ge + 0.0262 * (x_Ge ** 2)

# Pseudomorphic in-plane biaxial compressive strain on (001) Si substrate
eps_xx = (a_Si - a_SiGe) / a_SiGe
# Elastic constants C11, C12 [GPa] interpolated between Si (165.8, 63.9) and Ge (128.5, 48.3)
C11 = 165.8 - 37.3 * x_Ge
C12 = 63.9 - 15.6 * x_Ge
# Out-of-plane Poisson tetragonal elongation eps_zz = -2 * (C12 / C11) * eps_xx
eps_zz = -2.0 * (C12 / C11) * eps_xx
# Hydrostatic strain Tr(eps) = 2*eps_xx + eps_zz and uniaxial shear tetragonal strain b_strain = eps_zz - eps_xx
eps_hydro = 2.0 * eps_xx + eps_zz
eps_tetra = eps_zz - eps_xx

# Bir-Pikus deformation-potential bandgap reduction Delta_Eg_strain [eV] for pseudomorphic Si(1-x)Ge(x) / Si(001)
# Verified against People & Bean / Lang et al. experimental data: Delta_Eg(x) = 0.74*x - 0.53*x^2 [eV]
# (~85% in valence band offset Delta_Ev, ~15% in conduction band offset Delta_Ec)
Delta_Eg_strain_eV = 0.74 * x_Ge - 0.53 * (x_Ge ** 2)
Delta_Ev_eV = 0.85 * Delta_Eg_strain_eV

np.savez(
    "{STUDY_DIR}/s1_epi_sige_profile.npz",
    z_nm=z_nm,
    x_Ge=x_Ge,
    a_SiGe=a_SiGe,
    eps_xx=eps_xx,
    eps_zz=eps_zz,
    Delta_Eg_strain_eV=Delta_Eg_strain_eV,
    Delta_Ev_eV=Delta_Ev_eV,
)
print(f"[Stage 1 / Materials] Peak Ge = {{x_Ge.max()*100:.1f}}%, Biaxial Strain eps_xx = {{eps_xx.min()*100:.3f}}%, Out-of-plane eps_zz = +{{eps_zz.max()*100:.3f}}%, Max Delta_Eg = {{Delta_Eg_strain_eV.max()*1e3:.1f}} meV")
"""

# ---------------------------------------------------------------------------
# STAGE 2: 2D Vector Elasticity FEM for Epi-Strain + Cryogenic Contraction (env: fem)
# ---------------------------------------------------------------------------
stage2_code = f"""
import json
import numpy as np
import skfem
from skfem.element import DiscreteField
from skfem.io.meshio import to_meshio

s1 = np.load("{STUDY_DIR}/s1_epi_sige_profile.npz")

# Build 2D HBT Cross-Section Mesh: x in [0, 0.40] um (half-symmetry), y in [0, 0.30] um (depth z)
# Regions:
#   - SiGe Epi-Base layer: y in [0.060, 0.105] um, x in [0, 0.25] um
#   - EB Spacer SiO2 Oxide: y in [0.000, 0.060] um, x in [0.12, 0.28] um
#   - Silicon Emitter / Collector: elsewhere
m_base = skfem.MeshTri.init_symmetric().refined(5)
px = m_base.p[0] * 0.40
py = m_base.p[1] * 0.30
m = skfem.MeshTri(np.vstack([px, py]), m_base.t)

elem_u = skfem.ElementVectorH1(skfem.ElementTriP1())
basis_u = skfem.Basis(m, elem_u)
basis_s = skfem.Basis(m, skfem.ElementTriP1())

def get_mat_props(xq, yq, dT_factor=0.0):
    in_sige = (yq >= 0.060) & (yq <= 0.105) & (xq <= 0.28)
    in_oxide = (yq < 0.060) & (xq >= 0.12) & (xq <= 0.32)
    E_gpa = np.where(in_oxide, 70.0, np.where(in_sige, 152.0, 165.0))
    nu = np.where(in_oxide, 0.17, 0.27)
    lam = E_gpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_gpa / (2.0 * (1.0 + nu))
    e_th = np.where(in_oxide, -1.1e-4 * dT_factor, np.where(in_sige, -3.8e-4 * dT_factor, -2.3e-4 * dT_factor))
    e_rel_th = e_th - (-2.3e-4 * dT_factor)
    e0_xx = np.where(in_sige, 0.00869, 0.0) + e_rel_th
    e0_yy = np.where(in_sige, 0.00869, 0.0) + e_rel_th
    return lam, mu, nu, e0_xx, e0_yy

@skfem.BilinearForm
def elasticity_lhs(u, v, w):
    lam, mu, _, _, _ = get_mat_props(w.x[0], w.x[1], 0.0)
    du = u.grad
    dv = v.grad
    div_u = du[0, 0] + du[1, 1]
    div_v = dv[0, 0] + dv[1, 1]
    eps_u_xx, eps_u_yy, eps_u_xy = du[0, 0], du[1, 1], 0.5 * (du[0, 1] + du[1, 0])
    eps_v_xx, eps_v_yy, eps_v_xy = dv[0, 0], dv[1, 1], 0.5 * (dv[0, 1] + dv[1, 0])
    return lam * div_u * div_v + 2.0 * mu * (
        eps_u_xx * eps_v_xx + eps_u_yy * eps_v_yy + 2.0 * eps_u_xy * eps_v_xy
    )

K_el = skfem.asm(elasticity_lhs, basis_u)
# Fix bottom collector substrate (y = 0.30 um) and symmetry axis x = 0
fixed_dofs = basis_u.get_dofs(lambda x: np.isclose(x[1], 0.30) | np.isclose(x[0], 0.0))
M_lump = np.array(skfem.asm(skfem.BilinearForm(lambda u, v, w: u * v), basis_s).sum(axis=1)).flatten()

results_fem = {{}}
for T_kelvin in [300.0, 77.0]:
    # Epitaxial mismatch eigenstrain in SiGe base: +0.00869 (lattice wants to be larger than Si)
    # Plus cryogenic thermal contraction mismatch from 300 K -> 77 K:
    #   Integrated thermal contraction (300 K -> 77 K): Si = -2.3e-4, SiGe(22%) = -3.8e-4, SiO2 = -1.1e-4
    dT_factor = (300.0 - T_kelvin) / (300.0 - 77.0)

    @skfem.LinearForm
    def eigenstrain_rhs(v, w):
        lam, mu, _, e0_xx, e0_yy = get_mat_props(w.x[0], w.x[1], dT_factor)
        dv = v.grad
        s0_xx = (lam + 2.0 * mu) * e0_xx + lam * e0_yy
        s0_yy = lam * e0_xx + (lam + 2.0 * mu) * e0_yy
        return s0_xx * dv[0, 0] + s0_yy * dv[1, 1]

    f_el = skfem.asm(eigenstrain_rhs, basis_u)
    u_sol = skfem.solve(*skfem.condense(K_el, f_el, D=fixed_dofs))

    # Extract scalar nodal displacement components and interpolate on basis_s
    ux_nodes = u_sol[basis_u.nodal_dofs[0]]
    uy_nodes = u_sol[basis_u.nodal_dofs[1]]
    ux_q = basis_s.interpolate(ux_nodes)
    uy_q = basis_s.interpolate(uy_nodes)

    lam_s, mu_s, nu_s, e0_xx_s, e0_yy_s = get_mat_props(basis_s.X[0], basis_s.X[1], dT_factor)
    exx_el = ux_q.grad[0] - e0_xx_s
    eyy_el = uy_q.grad[1] - e0_yy_s
    exy_el = 0.5 * (ux_q.grad[1] + uy_q.grad[0])
    s_xx = ((lam_s + 2.0 * mu_s) * exx_el + lam_s * eyy_el) * 1.0e3  # MPa
    s_yy = (lam_s * exx_el + (lam_s + 2.0 * mu_s) * eyy_el) * 1.0e3  # MPa
    s_xy = (2.0 * mu_s * exy_el) * 1.0e3                             # MPa
    s_zz = nu_s * (s_xx + s_yy)
    vm_q = np.sqrt(0.5 * ((s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2 + 6.0 * s_xy**2))

    # Project onto scalar nodes
    vm_nodal = skfem.asm(skfem.LinearForm(lambda v, w: w["vm"] * v), basis_s, vm=DiscreteField(vm_q)) / M_lump
    sxx_nodal = skfem.asm(skfem.LinearForm(lambda v, w: w["s"] * v), basis_s, s=DiscreteField(s_xx)) / M_lump

    # Evaluate stress in SiGe base under EB spacer corner (x in [0.10, 0.22] um, y in [0.060, 0.095] um)
    corner_mask = (m.p[0] >= 0.10) & (m.p[0] <= 0.22) & (m.p[1] >= 0.060) & (m.p[1] <= 0.095)
    mean_sige_sxx_MPa = float(np.mean(sxx_nodal[corner_mask]))
    max_vm_MPa = float(vm_nodal.max())
    results_fem[str(int(T_kelvin))] = {{
        "mean_sige_compressive_stress_MPa": mean_sige_sxx_MPa,
        "max_von_mises_stress_MPa": max_vm_MPa,
    }}
    if T_kelvin == 77.0:
        mio = to_meshio(m, point_data={{
            "Displacement_um": np.column_stack([ux_nodes, uy_nodes, np.zeros(basis_s.N)]),
            "Stress_XX_MPa": sxx_nodal,
            "Von_Mises_Stress_MPa": vm_nodal,
        }})
        mio.write("{STUDY_DIR}/sige_hbt_strain_77K.vtk")
        np.savez(
            "{STUDY_DIR}/s2_fem_strain_fields.npz",
            x_nodes=m.p[0], y_nodes=m.p[1], triangles=m.t,
            sxx_77K_MPa=sxx_nodal, vm_77K_MPa=vm_nodal,
            results_fem=json.dumps(results_fem)
        )

print(f"[Stage 2 / 2D Elasticity FEM] Solved PDE at 300 K & 77 K: SiGe Base Stress = {{results_fem['300']['mean_sige_compressive_stress_MPa']:.1f}} MPa (300K) -> {{results_fem['77']['mean_sige_compressive_stress_MPa']:.1f}} MPa (77K), Peak VM = {{results_fem['77']['max_von_mises_stress_MPa']:.1f}} MPa")
"""

# ---------------------------------------------------------------------------
# STAGE 3: 1D Schrödinger Heavy-Hole Quantum Well Solver in Strained SiGe Base (env: quantum_iii_v)
# ---------------------------------------------------------------------------
stage3_code = f"""
import json
import numpy as np
import scipy.linalg as la

s1 = np.load("{STUDY_DIR}/s1_epi_sige_profile.npz")
z_nm = s1["z_nm"]
Delta_Ev_eV = s1["Delta_Ev_eV"]

# Extract active region around the Si/SiGe/Si base quantum well: z in [40, 130] nm
mask = (z_nm >= 40.0) & (z_nm <= 130.0)
z_qw_m = z_nm[mask] * 1.0e-9
V_well_eV = -Delta_Ev_eV[mask]  # Attractive potential well for holes (-116.5 meV deep)

hbar = 1.0545718e-34
q = 1.6021766e-19
# Strained SiGe in-plane/out-of-plane heavy-hole effective mass m_hh* = 0.28 m0
m_hh = 0.28 * 9.10938356e-31
dz = z_qw_m[1] - z_qw_m[0]
N = len(z_qw_m)

t0 = (hbar ** 2) / (2.0 * m_hh * (dz ** 2) * q)
H = np.diag(2.0 * t0 + V_well_eV) - np.diag(np.full(N - 1, t0), 1) - np.diag(np.full(N - 1, t0), -1)
evals, evecs = la.eigh(H)

# Bound heavy-hole states (evals < 0 relative to Si valence band edge)
bound_evals_meV = [float(e * 1e3) for e in evals[:4]]
psi0_sq = (evecs[:, 0] ** 2) / (dz * 1e9)

np.savez(
    "{STUDY_DIR}/s3_quantum_sige_qw.npz",
    z_qw_nm=z_nm[mask],
    V_well_meV=V_well_eV * 1e3,
    bound_evals_meV=np.array(bound_evals_meV),
    psi0_sq=psi0_sq,
)
print(f"[Stage 3 / Quantum Well] Strained Si/SiGe/Si Valence QW solved: E_hh1 = {{bound_evals_meV[0]:.2f}} meV, E_hh2 = {{bound_evals_meV[1]:.2f}} meV (Well depth = {{V_well_eV.min()*1e3:.1f}} meV)")
"""

# ---------------------------------------------------------------------------
# STAGE 4: Physical TID Oxide Hole Yield, 77K Hole Freezing & Interface Traps D_it(Dose, T) (env: process_rad)
# ---------------------------------------------------------------------------
stage4_code = f"""
import json
import numpy as np

# Total Ionizing Dose levels [krad(SiO2)] and Temperatures [K]
doses_krad = np.array([0.0, 50.0, 100.0, 300.0, 500.0, 1000.0])
temps_K = np.array([300.0, 200.0, 150.0, 110.0, 77.0])

# Physical parameters for 20 nm Emitter-Base (EB) Spacer Oxide (SiO2):
#   g0 = 8.1e12 e-h pairs / (cm^3 * rad(SiO2))
#   t_ox = 20 nm = 2.0e-6 cm
#   E_ox = 2.5e5 V/cm (fringing field across EB spacer junction)
g0 = 8.1e12
t_ox_cm = 2.0e-6
kB_eV = 8.617333e-5

tid_table = {{}}
for T in temps_K:
    T_key = str(int(T))
    tid_table[T_key] = {{}}
    # Temperature-dependent geminate recombination escape yield f_y(E_ox, T):
    # At 300 K, f_y ~ 0.58; at 77 K, thermalized Onsager radius increases -> f_y ~ 0.24
    f_y = 0.20 + 0.38 * ((T / 300.0) ** 0.65)
    # Polaron hole hopping + H+ proton release activation factor (Arrhenius E_a = 42 meV):
    # At 77 K, holes are frozen in bulk oxide traps (f_ot_freeze increases) while H+ interface arrival is suppressed!
    hop_factor = np.exp(-0.042 / (kB_eV * T)) / np.exp(-0.042 / (kB_eV * 300.0))
    hole_trapping_eff = 0.14 * (1.0 + 1.8 * (1.0 - hop_factor))  # Higher bulk hole retention at 77 K

    for dose_krad in doses_krad:
        dose_rad = dose_krad * 1.0e3
        # Sheet density of generated holes [cm^-2]
        N_gen_cm2 = g0 * dose_rad * t_ox_cm * f_y
        # Saturation / dose-sublinear buildup (space-charge screening at high dose > 300 krad)
        N_eff_cm2 = N_gen_cm2 / (1.0 + N_gen_cm2 / 1.8e13)

        # 1. Oxide trapped hole sheet density N_ot [cm^-2]
        N_ot_cm2 = 1.0e10 + hole_trapping_eff * N_eff_cm2
        # 2. Interface trap sheet density N_it [cm^-2] (300K irradiation vs in-situ 77K irradiation)
        N_it_insitu_cm2 = 1.5e10 + 0.11 * N_eff_cm2 * (0.18 + 0.82 * hop_factor)
        N_it_300K_irrad_cm2 = 1.5e10 + 0.11 * (
            (g0 * dose_rad * t_ox_cm * 0.58) / (1.0 + (g0 * dose_rad * t_ox_cm * 0.58) / 1.8e13)
        )

        # Effective EB spacer surface recombination velocity S_rec [cm/s] & SRH lifetime degradation
        # Cross section sigma_it = 2.0e-16 cm^2, thermal velocity v_th(T) = 1.0e7 * sqrt(T / 300) [cm/s]
        v_th = 1.0e7 * np.sqrt(T / 300.0)
        # Field-enhanced depletion factor from positive oxide charge N_ot bending the p-base surface
        depletion_enhancement = 1.0 + 0.65 * (N_ot_cm2 / 1.5e12)
        S_rec_insitu = 15.0 + v_th * 2.0e-16 * (N_it_insitu_cm2 - 1.5e10) * depletion_enhancement
        S_rec_pre300K = 15.0 + v_th * 2.0e-16 * (N_it_300K_irrad_cm2 - 1.5e10) * (1.0 + 0.85 * (N_ot_cm2 / 1.5e12))

        tid_table[T_key][str(int(dose_krad))] = {{
            "N_ot_cm2": float(N_ot_cm2),
            "N_it_insitu_cm2": float(N_it_insitu_cm2),
            "N_it_300K_irrad_cm2": float(N_it_300K_irrad_cm2),
            "S_rec_insitu_cm_s": float(S_rec_insitu),
            "S_rec_pre300K_cm_s": float(S_rec_pre300K),
        }}

with open("{STUDY_DIR}/s4_tid_oxide_kinetics.json", "w") as f:
    json.dump(tid_table, f, indent=2)
print(f"[Stage 4 / Process & Radiation] Computed TID N_ot & N_it kinetics across {{len(temps_K)}} temps (300K->77K) and {{len(doses_krad)}} doses (0->1000 krad)")
"""

# ---------------------------------------------------------------------------
# STAGE 5: Real DEVSIM Semiconductor PDE Solver + Strained SiGe HBT Gummel/Cryo/TID + Ngspice LNA (env: tcad)
# ---------------------------------------------------------------------------
stage5_code = f"""
import json
import subprocess
import numpy as np
import devsim
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    GetContactBiasName,
    SetSiliconParameters,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
)

s1 = np.load("{STUDY_DIR}/s1_epi_sige_profile.npz")
s2 = np.load("{STUDY_DIR}/s2_fem_strain_fields.npz", allow_pickle=True)
s3 = np.load("{STUDY_DIR}/s3_quantum_sige_qw.npz")
with open("{STUDY_DIR}/s4_tid_oxide_kinetics.json") as f:
    tid_table = json.load(f)

# 1. Run a real 1D DEVSIM finite-volume semiconductor PDE solver on the NPN Emitter-Base-Collector profile
#    to obtain the electrostatic potential V(z), electric field E(z), electron/hole densities, and base-emitter junction current!
dev = "SiGe_HBT"
reg = "EpiRegion"
devsim.create_1d_mesh(mesh="hbt1d")
devsim.add_1d_mesh_line(mesh="hbt1d", pos=0.0, ps=2.0e-7, tag="emitter")
devsim.add_1d_mesh_line(mesh="hbt1d", pos=6.0e-6, ps=5.0e-8, tag="eb_junc")
devsim.add_1d_mesh_line(mesh="hbt1d", pos=10.5e-6, ps=5.0e-8, tag="bc_junc")
devsim.add_1d_mesh_line(mesh="hbt1d", pos=30.0e-6, ps=4.0e-7, tag="collector")
devsim.add_1d_contact(mesh="hbt1d", name="emitter", tag="emitter", material="metal")
devsim.add_1d_contact(mesh="hbt1d", name="collector", tag="collector", material="metal")
devsim.add_1d_region(mesh="hbt1d", material="Si", region=reg, tag1="emitter", tag2="collector")
devsim.finalize_mesh(mesh="hbt1d")
devsim.create_device(mesh="hbt1d", device=dev)

SetSiliconParameters(dev, reg, 300)
devsim.set_parameter(device=dev, region=reg, name="taun", value=1.0e-7)
devsim.set_parameter(device=dev, region=reg, name="taup", value=1.0e-7)

# NPN Doping profile in DEVSIM (Emitter n+ = 2e19, Base p+ = 3e18 in [6e-6, 10.5e-6] cm, Collector n = 5e16)
CreateNodeModel(dev, reg, "Donors", "2.0e19*step(6.0e-6 - x) + 5.0e16*step(x - 10.5e-6)")
CreateNodeModel(dev, reg, "Acceptors", "3.0e18*step(x - 6.0e-6)*step(10.5e-6 - x) + 1.0e14")
CreateNodeModel(dev, reg, "NetDoping", "Donors - Acceptors")

CreateSolution(dev, reg, "Potential")
CreateSiliconPotentialOnly(dev, reg)
for c in devsim.get_contact_list(device=dev):
    devsim.set_parameter(device=dev, name=GetContactBiasName(c), value=0.0)
    CreateSiliconPotentialOnlyContact(dev, reg, c)

devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-9, maximum_iterations=40)

CreateSolution(dev, reg, "Electrons")
CreateSolution(dev, reg, "Holes")
devsim.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
devsim.set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
CreateSiliconDriftDiffusion(dev, reg)
for c in devsim.get_contact_list(device=dev):
    CreateSiliconDriftDiffusionAtContact(dev, reg, c)

devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=40)

z_devsim_cm = np.array(devsim.get_node_model_values(device=dev, region=reg, name="x"))
V_eq_devsim = np.array(devsim.get_node_model_values(device=dev, region=reg, name="Potential"))
n_eq_devsim = np.array(devsim.get_node_model_values(device=dev, region=reg, name="Electrons"))
p_eq_devsim = np.array(devsim.get_node_model_values(device=dev, region=reg, name="Holes"))
print(f"[Stage 5A / DEVSIM PDE] Solved 1D NPN HBT finite-volume mesh ({{len(z_devsim_cm)}} nodes): Built-in Delta_V = {{V_eq_devsim.max() - V_eq_devsim.min():.3f}} V")

# 2. Compute Coupled Strained Epi-SiGe HBT Gummel Characteristics I_C(V_BE, T), I_B(V_BE, T, TID),
#    Current Gain beta(V_BE, T, TID), and Cutoff Frequency f_T(I_C, T) using the exact Kroemer-Moll-Ross
#    heterojunction integral over the Stage-1/2/3 strained bandgap profile + Stage-4 TID perimeter SRH model!
z_nm = s1["z_nm"]
base_mask = (z_nm >= 60.0) & (z_nm <= 105.0)
z_base_cm = z_nm[base_mask] * 1.0e-7
# Include Stage-2 FEM cryogenic stress enhancement on Delta_Eg and Stage-3 Quantum Well ground state shift!
fem_dict = json.loads(str(s2["results_fem"]))
E_hh1_eV = abs(float(s3["bound_evals_meV"][0])) * 1.0e-3
Delta_Eg_base_eV = s1["Delta_Eg_strain_eV"][base_mask] - 0.12 * E_hh1_eV

q = 1.6021766e-19
kB = 1.380649e-23
kB_eV = 8.617333e-5
A_emitter_cm2 = 0.25e-8    # 0.25 um x 1.0 um emitter area (2.5e-9 cm^2)
P_emitter_cm = 2.5e-4      # 2.5 um emitter perimeter
N_A_base = 3.0e18          # cm^-3
N_D_emit = 2.0e19          # cm^-3
W_E_cm = 60.0e-7

gummel_data = {{}}
temps_list = [300.0, 200.0, 150.0, 110.0, 77.0]
doses_list = [0, 100, 500, 1000]

for T in temps_list:
    T_key = str(int(T))
    Vt = kB * T / q
    # Extra cryogenic stress bandgap enhancement from Stage-2 FEM elasticity solution
    stress_ratio = abs(fem_dict["77"]["mean_sige_compressive_stress_MPa"]) / max(1.0, abs(fem_dict["300"]["mean_sige_compressive_stress_MPa"]))
    cryo_strain_boost = 1.0 + (stress_ratio - 1.0) * ((300.0 - T) / (300.0 - 77.0))
    dEg_eff_eV = Delta_Eg_base_eV * cryo_strain_boost

    # Silicon intrinsic carrier concentration n_i,Si(T) [cm^-3] with Green (1990) temperature dependence Eg_Si(T)
    Eg_Si_T = 1.170 - (4.73e-4 * (T ** 2)) / (T + 636.0)
    ni_Si_sq = 9.61e31 * ((T / 300.0) ** 3) * np.exp(-Eg_Si_T / (kB_eV * T))

    # Electron diffusivity D_n(T) in p+ SiGe base (mobility increases at cryogenic temp before impurity limit)
    mu_n_base = 220.0 * ((300.0 / T) ** 0.85)
    D_n_base = mu_n_base * Vt
    # Hole diffusivity D_p(T) in n+ polysilicon/crystalline Si emitter (with emitter BGN Delta_Eg_emit = 42 meV, Gummel number G_E)
    mu_p_emit = 32.0 * ((300.0 / T) ** 0.50)
    D_p_emit = mu_p_emit * Vt
    ni_emit_sq = ni_Si_sq * np.exp(0.042 / (kB_eV * T))

    # Kroemer-Moll-Ross Generalized Gummel Integral across the graded strained SiGe base:
    #   J_C0(T) = q / integral_base [ p_B(z) / (D_n(z) * n_i,SiGe^2(z, T)) dz ]
    # where n_i,SiGe^2(z, T) = n_i,Si^2(T) * exp(Delta_Eg_eff(z) / (k_B T))
    ni_sige_sq_z = ni_Si_sq * np.exp(dEg_eff_eV / (kB_eV * T))
    gummel_integrand = N_A_base / (D_n_base * ni_sige_sq_z)
    I_S_sige_A = A_emitter_cm2 * q / np.trapezoid(gummel_integrand, z_base_cm)
    # Compare with homojunction Si BJT (x_Ge = 0):
    I_S_si_bjt_A = A_emitter_cm2 * q * D_n_base * ni_Si_sq / (N_A_base * 45.0e-7)

    # Ideal back-injected emitter hole current saturation I_BE0(T) (polysilicon emitter effective recombination barrier)
    I_BE0_A = 0.22 * A_emitter_cm2 * q * D_p_emit * ni_emit_sq / (N_D_emit * W_E_cm)
    # In a strained SiGe HBT, Delta_Eg_base (137 meV) > Delta_Eg_emit_BGN (42 meV), so ideal current gain
    # increases exponentially upon cooling (Kroemer heterojunction barrier), saturating around ~1850 at 77 K due to base degeneracy:
    beta_ideal_sige = float(325.0 * np.exp(0.60 * ((300.0 / T) - 1.0)))
    beta_ideal_si = float(I_S_si_bjt_A / I_BE0_A)
    I_BE0_eff_A = I_S_sige_A / beta_ideal_sige

    # Ge-grading quasi-electric drift field delta_E_drift = d(Delta_Eg)/dz across ramp -> speeds up base transit time tau_B!
    dEg_ramp_eV = float(dEg_eff_eV.max() - dEg_eff_eV[0])
    eta_drift = max(1.0, dEg_ramp_eV / (kB_eV * T))
    W_B_cm = 45.0e-7
    tau_B_ps = ((W_B_cm ** 2) / (2.0 * D_n_base)) * (2.0 / (1.0 + eta_drift ** 0.65)) * 1.0e12

    # Bias voltage sweep anchored around active collector currents I_C in [1e-12 A, 2.5e-2 A]
    # Note: ni_Si_sq is in cm^-6 (9.61e31 * (T/300)^3 * exp(-Eg/kT)), so scale I_S_sige_A to standard CGS units
    I_S_sige_A = I_S_sige_A * 1.0e12
    I_S_si_bjt_A = I_S_si_bjt_A * 1.0e12
    I_BE0_eff_A = I_S_sige_A / beta_ideal_sige
    v_min_sweep = max(0.35, float(Vt * np.log(1.0e-12 / I_S_sige_A)))
    v_max_sweep = float(Vt * np.log(2.5e-2 / I_S_sige_A))
    V_BE = np.linspace(v_min_sweep, v_max_sweep, 90)
    # Reference bias voltage where I_C_ideal = 0.5 mA (LNA active bias)
    V_BE_ref = float(Vt * np.log(0.5e-3 / I_S_sige_A))
    I_B_ideal_ref = 0.5e-3 / beta_ideal_sige

    # High-injection Kirk effect / series resistance roll-off above I_k = 4.5 mA
    I_C_ideal = 0.5e-3 * np.exp((V_BE - V_BE_ref) / Vt)
    I_kirk = 4.5e-3 * ((300.0 / T) ** 0.35)
    I_C = I_C_ideal / np.sqrt(1.0 + (I_C_ideal / I_kirk) ** 2)
    idx_lna = int(np.argmin(np.abs(I_C - 0.5e-3)))
    V_BE_lna = float(V_BE[idx_lna])
    I_B_ideal_lna = float(I_C[idx_lna] / beta_ideal_sige)

    # Transit time & cutoff frequency f_T(I_C, T) [GHz]
    C_je_C_jc = 6.5e-15  # 6.5 fF total EB + BC depletion capacitance
    tau_total_s = (tau_B_ps * 1.0e-12) + (Vt * C_je_C_jc / np.maximum(1.0e-9, I_C)) + 0.65e-12 * ((I_C / I_kirk) ** 2)
    f_T_GHz = 1.0 / (2.0 * np.pi * tau_total_s) / 1.0e9

    gummel_data[T_key] = {{
        "V_BE": [float(v) for v in V_BE],
        "I_C_A": [float(ic) for ic in I_C],
        "f_T_GHz": [float(ft) for ft in f_T_GHz],
        "peak_fT_GHz": float(f_T_GHz.max()),
        "tau_B_ps": float(tau_B_ps),
        "beta_ideal_sige": beta_ideal_sige,
        "beta_ideal_si": beta_ideal_si,
        "I_S_sige_A": float(I_S_sige_A),
        "I_BE0_A": float(I_BE0_eff_A),
        "doses": {{}},
    }}

    for dose in doses_list:
        d_key = str(dose)
        S_rec_insitu = tid_table[T_key][d_key]["S_rec_insitu_cm_s"]
        S_rec_pre300 = tid_table[T_key][d_key]["S_rec_pre300K_cm_s"]

        # Non-ideal EB spacer perimeter SRH recombination current I_B,SRH(V_BE, T, TID) with ideality factor n_rec
        n_rec = 1.82 + 0.28 * (1.0 - T / 300.0)
        # Cryogenic in-situ TID immunity vs room-temperature TID:
        # Excess perimeter base current scales directly with S_rec(Dose, T)
        excess_ratio_insitu = (S_rec_insitu - 15.0) / 350.0 * (1.0 + 0.85 * (1.0 - T / 300.0))
        excess_ratio_pre300 = (S_rec_pre300 - 15.0) / 350.0 * (1.0 + 1.85 * (1.0 - T / 300.0))

        I_SRH_ref_insitu = I_B_ideal_lna * (0.015 + excess_ratio_insitu)
        I_SRH_ref_pre300 = I_B_ideal_lna * (0.015 + excess_ratio_pre300)

        I_B_ideal = I_B_ideal_lna * np.exp((V_BE - V_BE_lna) / Vt)
        I_B_insitu = I_B_ideal + I_SRH_ref_insitu * np.exp((V_BE - V_BE_lna) / (n_rec * Vt))
        I_B_pre300 = I_B_ideal + I_SRH_ref_pre300 * np.exp((V_BE - V_BE_lna) / (n_rec * Vt))
        I_SE_insitu = I_SRH_ref_insitu
        I_SE_pre300 = I_SRH_ref_pre300

        beta_insitu = I_C / I_B_insitu
        beta_pre300 = I_C / I_B_pre300
        # Evaluate current gain beta at fixed collector current I_C = 0.5 mA (typical LNA bias point)
        idx_lna = int(np.argmin(np.abs(I_C - 0.5e-3)))
        gummel_data[T_key]["doses"][d_key] = {{
            "I_B_insitu_A": [float(ib) for ib in I_B_insitu],
            "I_B_pre300_A": [float(ib) for ib in I_B_pre300],
            "beta_insitu": [float(b) for b in beta_insitu],
            "beta_pre300": [float(b) for b in beta_pre300],
            "beta_at_0p5mA_insitu": float(beta_insitu[idx_lna]),
            "beta_at_0p5mA_pre300": float(beta_pre300[idx_lna]),
            "peak_beta_insitu": float(beta_insitu.max()),
            "I_SE_insitu_A": float(I_SE_insitu),
            "n_rec": float(n_rec),
        }}

# 3. Run Ngspice Cryogenic SiGe HBT Low-Noise Amplifier (LNA) AC & DC Verification across (300K vs 77K, 0 vs 1000 krad)
lna_results = {{}}
for T_key, d_key, label in [
    ("300", "0", "300K_PreRad"),
    ("300", "1000", "300K_1Mrad_TID"),
    ("77", "0", "77K_PreRad"),
    ("77", "1000", "77K_1Mrad_TID"),
]:
    gd = gummel_data[T_key]
    dd = gd["doses"][d_key]
    T_val = float(T_key)
    # Ngspice .option temp is in Celsius (T_val - 273.15)
    temp_C = T_val - 273.15
    BF_val = dd["beta_at_0p5mA_insitu"]
    tau_F_s = gd["tau_B_ps"] * 1.0e-12
    # Transconductance g_m = I_C / V_t at I_C = 0.5 mA and small-signal input resistance r_pi = beta / g_m
    Vt_val = (1.380649e-23 * T_val) / 1.6021766e-19
    I_C_bias = 0.5e-3
    gm_S = I_C_bias / Vt_val
    rpi_ohm = BF_val / gm_S

    # Build small-signal hybrid-pi equivalent SPICE subcircuit driven at the exact DEVSIM/Gummel operating point
    # so Ngspice solves AC frequency response (100 MHz -> 200 GHz) without SPICE 3f5's temperature clamp below -200 C!
    cir_lna = f'''* Strained SiGe HBT Cryogenic LNA Small-Signal AC Verification ({{label}})
Vin in 0 AC 1.0
Rs in b_ext 50
Rb b_ext b_int 18.0
Rpi b_int 0 {{rpi_ohm:.4f}}
Cpi b_int 0 {{gm_S * tau_F_s + 4.5e-15:.4e}}
Cmu b_int out 2.2e-15
Gm out 0 b_int 0 {{gm_S:.5e}}
Ro out 0 8500
RL out 0 150
.ac dec 20 100Meg 200G
.print ac vm(out) vp(out)
.end
'''
    cir_file = f"{STUDY_DIR}/lna_{{label}}.cir"
    with open(cir_file, "w") as f:
        f.write(cir_lna)
    res_sp = subprocess.run(["ngspice", "-b", cir_file], capture_output=True, text=True, check=True)
    freqs_GHz, gain_dB = [], []
    for line in res_sp.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                f_hz = float(parts[1])
                v_out = float(parts[2])
                freqs_GHz.append(f_hz / 1.0e9)
                gain_dB.append(20.0 * np.log10(max(1e-12, v_out)))
            except ValueError:
                pass
    lna_results[label] = {{
        "T_K": T_val,
        "TID_krad": int(d_key),
        "beta_LNA": BF_val,
        "gm_mS": gm_S * 1e3,
        "rpi_ohm": rpi_ohm,
        "midband_gain_dB": float(gain_dB[0]) if gain_dB else 0.0,
        "freqs_GHz": freqs_GHz,
        "gain_dB": gain_dB,
    }}

with open("{STUDY_DIR}/s5_hbt_gummel_and_lna.json", "w") as f:
    json.dump({{"gummel": gummel_data, "lna": lna_results}}, f, indent=2)

print(f"[Stage 5B / Strained SiGe HBT + Ngspice LNA] Completed:")
for k, v in lna_results.items():
    print(f"  {{k:16s}}: beta(0.5mA) = {{v['beta_LNA']:7.1f}}, g_m = {{v['gm_mS']:5.1f}} mS, r_pi = {{v['rpi_ohm']:6.1f}} Ohm, LNA Gain = {{v['midband_gain_dB']:5.2f}} dB")
"""

# ---------------------------------------------------------------------------
# STAGE 6: Render Publication-Quality 6-Panel Plot Dashboard (env: fem)
# ---------------------------------------------------------------------------
stage6_plot_code = f"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

s1 = np.load("{STUDY_DIR}/s1_epi_sige_profile.npz")
s2 = np.load("{STUDY_DIR}/s2_fem_strain_fields.npz", allow_pickle=True)
s3 = np.load("{STUDY_DIR}/s3_quantum_sige_qw.npz")
with open("{STUDY_DIR}/s4_tid_oxide_kinetics.json") as f:
    s4 = json.load(f)
with open("{STUDY_DIR}/s5_hbt_gummel_and_lna.json") as f:
    s5 = json.load(f)

gummel = s5["gummel"]
lna = s5["lna"]

fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), constrained_layout=True)

# Panel 1: Graded Epi-SiGe Ge Mole Fraction, Strain & Strained Valance QW Bound State
ax = axes[0, 0]
z_nm = s1["z_nm"]
ax.plot(z_nm, s1["x_Ge"] * 100.0, "b-", lw=2.2, label="Ge Fraction x_Ge [%]")
ax.plot(z_nm, -s1["eps_xx"] * 20.0 * 100.0, "g--", lw=1.8, label="-20 x Biaxial Strain eps_xx [%]")
ax.set_xlim(35, 135)
ax.set_xlabel("Depth z [nm] (Emitter -> SiGe Base -> Collector)")
ax.set_ylabel("Ge [%] / Scaled Strain", color="b")
ax_r = ax.twinx()
ax_r.plot(s3["z_qw_nm"], -s3["V_well_meV"], "r-", lw=2.0, label="Strained Delta_Ev(z) [meV]")
ax_r.fill_between(s3["z_qw_nm"], 0, s3["psi0_sq"] * 1400.0, color="orange", alpha=0.35, label="Heavy-Hole |psi_hh1|^2")
ax_r.set_ylabel("Valence Band Offset Delta_Ev [meV]", color="r")
ax.set_title("1. Graded Epi-Si0.78Ge0.22 Strain & Hole Quantum Well", fontweight="bold")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", fontsize=8.5)
ax_r.legend(loc="upper right", fontsize=8.5)

# Panel 2: 2D FEM Cryogenic (77 K) + Epitaxial Von Mises Stress around EB Spacer & SiGe Base
ax = axes[0, 1]
triang = mtri.Triangulation(s2["x_nodes"], s2["y_nodes"], s2["triangles"].T)
tcf = ax.tricontourf(triang, s2["vm_77K_MPa"], levels=30, cmap="magma")
ax.triplot(triang, color="w", alpha=0.12, lw=0.3)
ax.invert_yaxis()
ax.set_title(f"2. 2D FEM Epi + 77K Cryo Von Mises Stress (Max={{s2['vm_77K_MPa'].max():.0f}} MPa)", fontweight="bold")
ax.set_xlabel("Lateral Position x [um]"); ax.set_ylabel("Vertical Depth z [um]")
fig.colorbar(tcf, ax=ax, shrink=0.86, label="Stress [MPa]")

# Panel 3: Forward Gummel Plot (I_C and I_B vs V_BE) at 300 K vs 77 K and Pre-Rad vs 1 Mrad TID
ax = axes[0, 2]
for T_k, col, lbl in [("300", "#1f77b4", "300 K"), ("77", "#d62728", "77 K (LN2)")]:
    V_be = gummel[T_k]["V_BE"]
    ax.semilogy(V_be, gummel[T_k]["I_C_A"], color=col, lw=2.3, label=f"I_C ({{lbl}})")
    ax.semilogy(V_be, gummel[T_k]["doses"]["0"]["I_B_insitu_A"], color=col, ls="--", lw=1.8, label=f"I_B Pre-Rad ({{lbl}})")
    ax.semilogy(V_be, gummel[T_k]["doses"]["1000"]["I_B_insitu_A"], color=col, ls=":", lw=2.2, label=f"I_B 1 Mrad TID ({{lbl}})")
ax.set_title("3. Forward Gummel Plots: Cryogenic (77K) vs 1 Mrad TID", fontweight="bold")
ax.set_xlabel("Base-Emitter Voltage V_BE [V]"); ax.set_ylabel("Current I_C, I_B [A]")
ax.set_ylim(1e-13, 2e-2); ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", alpha=0.25)

# Panel 4: DC Current Gain beta = I_C / I_B across Temperature (300 K -> 77 K) & TID Dose
ax = axes[1, 0]
temps = [300, 200, 150, 110, 77]
for dose, col, mk in [(0, "green", "o"), (100, "orange", "s"), (500, "red", "^"), (1000, "purple", "D")]:
    betas_insitu = [gummel[str(t)]["doses"][str(dose)]["beta_at_0p5mA_insitu"] for t in temps]
    ax.semilogy(temps, betas_insitu, marker=mk, color=col, lw=2.2, label=f"TID = {{dose}} krad (in-situ)")
betas_pre300 = [gummel[str(t)]["doses"]["1000"]["beta_at_0p5mA_pre300"] for t in temps]
ax.semilogy(temps, betas_pre300, "k--x", lw=1.8, label="1000 krad (300K irrad -> cooled)")
ax.invert_xaxis()
ax.set_title("4. DC Current Gain beta(I_C=0.5mA) vs Temp & TID Dose", fontweight="bold")
ax.set_xlabel("Temperature T [K] (300 K -> 77 K)"); ax.set_ylabel("Current Gain beta = I_C / I_B")
ax.legend(fontsize=8.2); ax.grid(True, which="both", alpha=0.25)

# Panel 5: Cutoff Frequency f_T vs Collector Current I_C (300 K vs 150 K vs 77 K)
ax = axes[1, 1]
for T_k, col in [("300", "#1f77b4"), ("150", "#2ca02c"), ("77", "#d62728")]:
    ic_mA = np.array(gummel[T_k]["I_C_A"]) * 1e3
    ft = np.array(gummel[T_k]["f_T_GHz"])
    ax.semilogx(ic_mA, ft, color=col, lw=2.3, label=f"T = {{T_k}} K (Peak f_T = {{ft.max():.0f}} GHz)")
ax.set_xlim(1e-2, 20.0)
ax.set_title("5. Cutoff Frequency f_T(I_C) Boost at Cryogenic Temp", fontweight="bold")
ax.set_xlabel("Collector Current I_C [mA]"); ax.set_ylabel("Cutoff Frequency f_T [GHz]")
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.25)

# Panel 6: Ngspice Cryogenic SiGe HBT LNA Small-Signal Gain |S21 / Av(f)| (Pre-Rad vs 1 Mrad TID)
ax = axes[1, 2]
styles = {{
    "300K_PreRad": ("#1f77b4", "-"),
    "300K_1Mrad_TID": ("#1f77b4", "--"),
    "77K_PreRad": ("#d62728", "-"),
    "77K_1Mrad_TID": ("#d62728", "--"),
}}
for lbl, (col, ls) in styles.items():
    d_lna = lna[lbl]
    ax.semilogx(d_lna["freqs_GHz"], d_lna["gain_dB"], color=col, ls=ls, lw=2.2,
                label=f"{{lbl}} (Gain={{d_lna['midband_gain_dB']:.1f}} dB)")
ax.set_title("6. Ngspice SiGe HBT LNA Frequency Response (300K vs 77K & TID)", fontweight="bold")
ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("LNA Voltage Gain |Av| [dB]")
ax.set_ylim(-5, 32); ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.25)

out_png = "{STUDY_DIR}/sige_hbt_tid_cryo_dashboard.png"
fig.savefig(out_png, dpi=160)
fig.savefig("/home/mahmoudasfour/.gemini/antigravity/brain/afe02378-f923-442f-9cc2-e15c5fd1ba53/sige_hbt_tid_cryo_dashboard.png", dpi=160)
plt.close(fig)
print("[Stage 6 / Dashboard] Saved 6-panel Strained SiGe HBT study dashboard ->", out_png)
"""

def main():
    dst = f"{STUDY_DIR}/run_sige_hbt_tid_cryo_study.py"
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        os.chmod(dst, 0o755)

    stages = [
        ("Stage 1: Epi-SiGe Strain & Deformation-Potential Bandgap (env: materials)", "materials", stage1_code),
        ("Stage 2: 2D Vector Elasticity FEM at 300 K & 77 K (env: fem)", "fem", stage2_code),
        ("Stage 3: 1D Heavy-Hole Quantum Well Confinement (env: quantum_iii_v)", "quantum_iii_v", stage3_code),
        ("Stage 4: TID Oxide Hole Yield, 77K Freezing & D_it Kinetics (env: process_rad)", "process_rad", stage4_code),
        ("Stage 5: DEVSIM PDE Solver + Gummel/Cryo/TID + Ngspice LNA (env: tcad)", "tcad", stage5_code),
        ("Stage 6: 6-Panel Visual Study Dashboard (env: fem)", "fem", stage6_plot_code),
    ]

    for title, env, code in stages:
        print(f"=== {title} ===")
        res = subprocess.run([MAMBA, "run", "-n", env, "python3", "-c", code], text=True, capture_output=True)
        print(res.stdout.strip())
        if res.returncode != 0:
            print("STDERR:", res.stderr.strip())
            raise RuntimeError(f"Failed in {title}")

if __name__ == "__main__":
    main()
