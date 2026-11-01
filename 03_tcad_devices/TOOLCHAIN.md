# TCAD toolchain manifest (generated 2026-11-01)

Toolchain root: `~/tcad` (symlinked as `03_tcad_devices/toolchain`, not in git)

| Component | Version / identity | Location |
|---|---|---|
| Charon distrib v2.2 | sha256 `2743f39fb14166091f1e38581f9d85379a7db178b4b2d4ce5c8411fdec727073` | ~/tcad/charon-distrib-v2_2.tar.gz |
| Trilinos | `81e9581a3c5f701d41e6828d0f0e5affcef5373a` | ~/tcad/tcad-charon/Trilinos |
| TriBITS | `769f615fafbaac66be8caa47d2916b1e9b078f5d` | ~/tcad/TriBITS |
| Charon executable | source build | ~/tcad/build/src/charon_mp.exe |
| Dakota | 6.24.0 RHEL8 CLI binary | ~/tcad/dakota-6.24.0-public-rhel8.Linux.x86_64-cli |
| OpenVAF-Reloaded | v24.0.2mob, sha256 `` | ~/opt/openvaf (symlinked as tools/openvaf) |

## Charon build options (~/tcad/conda.opts)
```
CMAKE_CXX_COMPILER:FILEPATH="/home/mahmoudasfour/miniforge3/envs/charon/bin/mpicxx"
CMAKE_C_COMPILER:FILEPATH="/home/mahmoudasfour/miniforge3/envs/charon/bin/mpicc"
CMAKE_Fortran_COMPILER:FILEPATH="/home/mahmoudasfour/miniforge3/envs/charon/bin/mpifort"
CMAKE_CXX_FLAGS_RELEASE_OVERRIDE:STRING="-O2"
CMAKE_C_FLAGS_RELEASE_OVERRIDE:STRING="-O2"
CMAKE_Fortran_FLAGS_RELEASE_OVERRIDE:STRING="-O2"
CMAKE_CXX_FLAGS:STRING="-Wno-deprecated-declarations"
TPL_ENABLE_BLAS:BOOL=ON
TPL_ENABLE_LAPACK:BOOL=ON
Boost_INCLUDE_DIRS:PATH=/home/mahmoudasfour/miniforge3/envs/charon/include
BoostLib_LIBRARY_DIRS:PATH=/home/mahmoudasfour/miniforge3/envs/charon/lib
BoostLib_INCLUDE_DIRS:PATH=/home/mahmoudasfour/miniforge3/envs/charon/include
Netcdf_LIBRARY_DIRS:PATH=/home/mahmoudasfour/miniforge3/envs/charon/lib
Netcdf_INCLUDE_DIRS:PATH=/home/mahmoudasfour/miniforge3/envs/charon/include
tcad-charon_ENABLE_DAKOTA_DRIVERS:BOOL=OFF
Charon_ENABLE_DEBUG:BOOL=OFF
CMAKE_INSTALL_PREFIX:PATH=/home/mahmoudasfour/tcad/install
TPL_BLAS_LIBRARIES:FILEPATH=/home/mahmoudasfour/miniforge3/envs/charon/lib/libblas.so
TPL_LAPACK_LIBRARIES:FILEPATH=/home/mahmoudasfour/miniforge3/envs/charon/lib/liblapack.so
```
