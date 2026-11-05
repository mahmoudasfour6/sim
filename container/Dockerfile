# Docker equivalent of container/simws.def (same steps). From the repository root:
#   docker build -f container/Dockerfile -t simws:0.9c .
#   docker run --rm -v $PWD/results:/opt/sim_workspace/results simws:0.9c snakemake --cores 8 studies
FROM condaforge/miniforge3:24.9.2-0
WORKDIR /opt/sim_workspace
COPY envs envs
COPY tools/container_envs.py tools/container_envs.py
RUN /opt/conda/bin/python tools/container_envs.py --envs-dir envs --prefix /opt/conda/envs \
        --only workflow,tcad,materials,process_rad,quantum_iii_v,fenicsx && /opt/conda/bin/conda clean -afy
COPY simws simws
COPY workflow workflow
COPY config config
COPY tools tools
COPY tests tests
COPY pyproject.toml pyproject.toml
RUN sed -i 's#^envs_root:.*#envs_root: /opt/conda/envs#' config/config.yaml
ENV DEVSIM_MATH_LIBS=/opt/conda/envs/tcad/lib/libopenblas.so.0 PYTHONNOUSERSITE=1 SIMWS_ROOT=/opt/sim_workspace PATH=/opt/conda/envs/workflow/bin:/opt/conda/bin:$PATH OMP_NUM_THREADS=1
