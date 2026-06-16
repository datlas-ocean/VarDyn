"""
Created by Florian Le Guillou on June 2026.

Global precision flag used to configure JAX float64 behavior.
"""

# config.py
import os
import jax

# Global flag: True for float64, False for float32
USE_FLOAT64 = False

jax.config.update("jax_enable_x64", USE_FLOAT64)