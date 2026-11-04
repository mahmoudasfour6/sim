# Docker equivalent of container/simws.def (same steps). From the repository root:
#   docker build -f container/Dockerfile -t simws:0.9c .
#   docker run --rm -v $PWD/results:/opt/sim_workspace/results simws:0.9c snakemake --cores 8 studies
FROM condaforge/miniforge3:24.9.2-0
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gfortran git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/sim_workspace
COPY envs envs
COPY tools/container_envs.py tools/container_envs.py
RUN /opt/conda/bin/python tools/container_envs.py --envs-dir envs --prefix /opt/conda/envs && /opt/conda/bin/conda clean -afy
COPY simws simws
COPY workflow workflow
COPY config config
COPY tools tools
COPY tests tests
COPY pyproject.toml pyproject.toml
RUN sed -i 's#^envs_root:.*#envs_root: /opt/conda/envs#' config/config.yaml
ENV SIMWS_ROOT=/opt/sim_workspace PATH=/opt/conda/envs/workflow/bin:/opt/conda/bin:$PATH OMP_NUM_THREADS=1
