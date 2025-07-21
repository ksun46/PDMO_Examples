"""
    AbstractADMMAdapter

Abstract base type for ADMM penalty parameter adaptation schemes.

ADMM adapters are responsible for dynamically adjusting the penalty parameter (ρ) during
ADMM iterations to improve convergence rates and numerical stability. The penalty parameter
plays a crucial role in balancing the enforcement of constraints versus the optimization
of the objective function.

**Mathematical Background**

The ADMM algorithm solves optimization problems of the form:
```math
\\min_{x,z} f(x) + g(z) \\text{ subject to } Ax + Bz = c
```

The augmented Lagrangian is:
```math
L_ρ(x,z,λ) = f(x) + g(z) + λ^T(Ax + Bz - c) + \\frac{ρ}{2}\\|Ax + Bz - c\\|_2^2
```

where ρ is the penalty parameter that affects:
- **Convergence speed**: Higher ρ can accelerate convergence but may cause numerical issues
- **Numerical stability**: Lower ρ provides better conditioning but slower convergence
- **Primal-dual balance**: Optimal ρ balances primal and dual residual reductions

**Adapter Interface**

All concrete adapter types must implement:
- `initialize!(adapter, info, admmGraph)`: Initialize adapter with problem data
- `updatePenalty(adapter, info, admmGraph, iter)`: Update penalty parameter and return whether changed

**Common Adaptation Strategies**

1. **Residual Balancing**: Adjust ρ based on primal vs dual residual magnitudes
2. **Spectral Methods**: Use spectral radius estimates to guide parameter selection
3. **Adaptive Schedules**: Predetermined schedules based on iteration counts
4. **Gradient-based**: Use gradient information to optimize parameter selection

**Performance Considerations**

- **Frequency**: Adapters should not update ρ too frequently (every 5-10 iterations)
- **Magnitude**: Changes should be moderate (factors of 2-10) to avoid instability
- **Bounds**: ρ should be constrained within reasonable numerical bounds
- **Convergence**: Adaptation should eventually stabilize as algorithm converges

**Available Implementations**

- `NullAdapter`: No adaptation (constant ρ)
- `RBAdapter`: Residual balancing adaptation
- `SRAAdapter`: Spectral radius adaptive adaptation

**See Also**
- `NullAdapter`: No-operation adapter for baseline comparison
- `RBAdapter`: Residual balancing adapter
- `SRAAdapter`: Spectral radius adaptive adapter


"""
abstract type AbstractADMMAdapter end 

"""
    ADMM_MIN_RHO

Minimum allowed value for the ADMM penalty parameter.

This constant defines the lower bound for the penalty parameter ρ to prevent
numerical instability that can occur with very small penalty values.

**Value**: 1.0e-2

**Rationale**
- Prevents division by zero or near-zero operations
- Maintains reasonable conditioning of the augmented Lagrangian
- Ensures adequate constraint enforcement
"""
const ADMM_MIN_RHO = 1.0e-2

"""
    ADMM_MAX_RHO

Maximum allowed value for the ADMM penalty parameter.

This constant defines the upper bound for the penalty parameter ρ to prevent
numerical overflow and excessive constraint penalty that can dominate the objective.

**Value**: 1.0e8

**Rationale**
- Prevents numerical overflow in penalty computations
- Avoids excessive constraint weighting that can hurt convergence
- Maintains balance between objective and constraint terms
"""
const ADMM_MAX_RHO = 1.0e8 

"""
    NullAdapter <: AbstractADMMAdapter

No-operation adapter that maintains a constant penalty parameter throughout ADMM iterations.

This adapter provides a baseline behavior where the penalty parameter ρ remains unchanged
from its initial value. It's useful for:
- Baseline performance comparisons
- Problems where optimal ρ is known a priori
- Debugging and testing ADMM implementations
- Scenarios where adaptation might be harmful

"""
mutable struct NullAdapter <: AbstractADMMAdapter 
    NullAdapter() = new()
end

"""
    initialize!(adapter::NullAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the null adapter (no-operation).

**Arguments**
- `adapter::NullAdapter`: The adapter instance
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of the problem

**Implementation**
This function performs no operations and returns immediately since the null adapter
requires no initialization.
"""
function initialize!(adapter::NullAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    return 
end 

"""
    updatePenalty(adapter::NullAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, iter::Int64)

Update penalty parameter for null adapter (no-operation).

**Arguments**
- `adapter::NullAdapter`: The adapter instance
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation
- `iter::Int64`: Current iteration number

**Returns**
- `Bool`: Always returns `false` indicating no parameter change

**Implementation**
This function always returns `false` since the null adapter never changes the penalty parameter.
"""
function updatePenalty(adapter::NullAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, iter::Int64)
    return false 
end 


include("RBAdapter.jl")
include("SRAAdapter.jl")

"""
    getADMMAdapterName(adapter::AbstractADMMAdapter)

Get a string identifier for the adapter type.

**Arguments**
- `adapter::AbstractADMMAdapter`: The adapter instance

**Returns**
- `String`: A string identifier for the adapter type:
  - `"NULL_ADAPTER"` for `NullAdapter`
  - `"RB_ADAPTER"` for `RBAdapter`
  - `"SRA_ADAPTER"` for `SRAAdapter`

**Throws**
- `ErrorException`: If the adapter type is not recognized



**Usage**
This function is commonly used for:
- Logging and debugging output
- Performance tracking and comparison
- Configuration management
- Algorithm selection in parameter studies
"""
function getADMMAdapterName(adapter::AbstractADMMAdapter)
    if isa(adapter, NullAdapter)
        return "NULL_ADAPTER"
    elseif isa(adapter, RBAdapter)
        return "RB_ADAPTER"
    elseif isa(adapter, SRAAdapter)
        return "SRA_ADAPTER"
    else 
        error("getADMMAdapterName: Unknown adapter type")
    end 
end 
