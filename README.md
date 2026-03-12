# VarDyn

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/datlas-ocean/VarDyn.git
   cd VarDyn
   ```

2. **Create a new environment**
   ```bash
   conda create -n env_vardyn python=3.10
   ```
   ```bash
   conda activate env_vardyn
   ```

3. **Install `pyinterp` with conda-forge** (very long, up to 2 hours)
   ```bash
   conda install -c conda-forge pyinterp
   ```
4. (OPTIONAL) **Install `jax` for GPU or TPU** 
\
\
Users with access to GPUs or TPUs should first install `jax` separately in order to fully benefit from its high-performance computing capacities. See [JAX instructions](https://docs.jax.dev/en/latest/installation.html). By default, a CPU-only version of JAX will be installed if no other version is already present in the Python environment. 
   
5. **Install other dependencies with pip** 
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

