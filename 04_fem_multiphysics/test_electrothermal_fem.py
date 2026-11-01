#!/usr/bin/env python3
"""
Coupled Electro-Thermal (Joule Heating) FEM Simulation
Solves:
  1. Electrical conduction: -div(sigma * grad(V)) = 0
  2. Joule heat source:     Q = sigma * |grad(V)|^2
  3. Heat transfer:         -div(k * grad(T)) = Q
Exports VTK multiphysics fields for ParaView / PyVista.
"""
import numpy as np
import skfem
from skfem.models.poisson import laplace, unit_load
from skfem.io.meshio import to_meshio
import dolfinx
from mpi4py import MPI
import pyvista as pv

def run_electrothermal_sim(output_vtk="electrothermal_joule.vtk"):
    # Create 2D interconnect/device mesh (10 um x 10 um)
    m = skfem.MeshTri.init_symmetric().scaled(10.0).refined(4)
    e = skfem.ElementTriP1()
    basis = skfem.Basis(m, e)

    # Material properties (doped Si / metal heater region)
    sigma = 1.0e4  # S/m
    k_th = 148.0   # W/(m*K) for Silicon

    # 1. Solve electrical potential: V = 1.5 V on left, 0 V on right
    left_dofs = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))
    right_dofs = basis.get_dofs(lambda x: np.isclose(x[0], 10.0))
    
    A_elec = skfem.asm(laplace, basis)
    b_elec = np.zeros(basis.N)
    V = np.zeros(basis.N)
    V[left_dofs] = 1.5
    V[right_dofs] = 0.0
    D_elec = left_dofs + right_dofs
    V = skfem.solve(*skfem.condense(A_elec, b_elec, x=V, D=D_elec))

    # 2. Compute Joule heating Q = sigma * |grad(V)|^2 at quadrature points
    grad_V = basis.interpolate(V).grad
    E_sq = grad_V[0]**2 + grad_V[1]**2

    @skfem.LinearForm
    def joule_load(v, w):
        return (sigma / k_th) * w['E_sq'] * v

    A_th = skfem.asm(laplace, basis)
    b_th = skfem.asm(joule_load, basis, E_sq=E_sq)

    # Heat sink at bottom boundary (y = 0) at T = 300 K
    bottom_dofs = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))
    T = np.full(basis.N, 300.0)
    T = skfem.solve(*skfem.condense(A_th, b_th, x=T, D=bottom_dofs))

    # Export via meshio / PyVista
    mio = to_meshio(m, point_data={"Potential_V": V, "Temperature_K": T})
    mio.write(output_vtk)

    # Also verify FEniCSx (dolfinx) MPI mesh creation
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local

    print(f"[FEM] FEniCSx (dolfinx v{dolfinx.__version__}) verified ({num_cells} MPI cells).")
    print(f"[FEM] PyVista v{pv.__version__} & scikit-fem v{skfem.__version__} verified.")
    print(f"[FEM] Electro-Thermal results: V_max = {V.max():.2f} V, T_max = {T.max():.2f} K -> saved to {output_vtk}")

if __name__ == "__main__":
    run_electrothermal_sim()
