# VarDyn

VarDyn reconstructs ocean dynamical variables by combining a reduced control
space, dynamical propagation, observations, and variational inversion.

## Language

**Analysis Window**:
The time interval jointly optimized by one 4DVar inversion.
_Avoid_: Batch, run period

**Checkpoint**:
A time in the Analysis Window where VarDyn applies the Reduced Basis, evaluates
observation misfits, or stores state needed by the adjoint.
_Avoid_: Snapshot, save time

**Control Vector**:
The coefficients optimized by 4DVar; the Reduced Basis maps them to model
variables and parameters.
_Avoid_: Parameter vector, weights

**Reduced Basis**:
The mapping between the Control Vector and corrections to model variables or
parameters over space and time.
_Avoid_: Encoder, feature basis

**Cost-Gradient Evaluation**:
One complete background-cost, forward-model, observation-misfit, and adjoint
calculation for a Control Vector.
_Avoid_: Objective call, model pass

**4DVar Minimizer**:
The algorithm that proposes Control Vectors and accepts steps using
Cost-Gradient Evaluations.
_Avoid_: Solver, optimizer backend

**Historical SciPy Minimizer**:
The host-resident SciPy L-BFGS-B 4DVar Minimizer retained for compatibility and
reference comparisons.
_Avoid_: CPU mode, legacy solver

**Device-Resident Optax Minimizer**:
The L-BFGS 4DVar Minimizer whose Control Vector and history remain on the JAX
device while Python decides the scalar Armijo line search.
_Avoid_: Full-GPU solver, JAXopt mode
