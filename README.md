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

3. **Install ```pyinterp``` with conda-forge** (very long, up to 2 hours)
   ```bash
   conda install -c conda-forge pyinterp
   ```
   
3. **Install other dependencies with pip** 
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

