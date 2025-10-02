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
    initialize!(adapter::AbstractADMMAdapter, 
               info::ADMMIterationInfo, 
               admmGraph::ADMMBipartiteGraph)

Initialize the ADMM penalty parameter adapter with problem data and algorithm state.

This method is called once before the main ADMM iterations begin. It allows adapters
to perform preprocessing, analyze problem structure, compute initial parameters, and
set up any adapter-specific data structures or metrics.

**Arguments**
- `adapter::AbstractADMMAdapter`: The adapter instance to initialize
- `info::ADMMIterationInfo`: Initial iteration information and algorithm state
- `admmGraph::ADMMBipartiteGraph`: The ADMM bipartite graph structure

**Required Implementation**
Every concrete ADMM adapter MUST implement this method. The implementation should handle:

1. **Problem Structure Analysis**: Examine the bipartite graph structure and constraint properties
2. **Initial Parameter Computation**: Calculate baseline metrics for adaptation decisions
3. **History Initialization**: Set up tracking for residual trends and parameter changes
4. **Workspace Allocation**: Pre-allocate any buffers needed for adaptation computations
5. **Validation**: Check that the adapter can handle the given problem structure

**Example Implementation**
```julia
function initialize!(adapter::MyConcreteAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    # Analyze problem structure
    adapter.numNodes = length(admmGraph.nodes)
    adapter.numEdges = length(admmGraph.edges)
    
    # Initialize tracking metrics
    adapter.residualHistory = Float64[]
    adapter.rhoHistory = Float64[]
    adapter.lastUpdateIteration = 0
    
    # Compute initial adaptation parameters
    adapter.initialRho = info.rhoHistory[end][1]
    adapter.adaptationThreshold = computeAdaptationThreshold(admmGraph)
    
    # Allocate workspace if needed
    adapter.workspace = allocateWorkspace(admmGraph)
    
    # Set adapter state
    adapter.initialized = true
end
```

**Common Initialization Tasks**
- **Residual Balancing Adapters**: Compute initial primal/dual residual baselines
- **Spectral Adapters**: Estimate spectral properties of constraint matrices
- **Schedule-based Adapters**: Set up predetermined adaptation schedules
- **Gradient-based Adapters**: Initialize gradient estimation mechanisms

**Error Handling**
- Validate that required problem structure is present
- Check for numerical issues in initial penalty parameter
- Ensure adapter parameters are within reasonable bounds

**Performance Considerations**
- Keep initialization lightweight to avoid startup overhead
- Pre-compute expensive quantities that will be reused
- Allocate workspace to avoid allocations during adaptation

See also: `updatePenalty`, `ADMMIterationInfo`, `ADMMBipartiteGraph`
"""
function initialize!(adapter::AbstractADMMAdapter, 
                    info::ADMMIterationInfo, 
                    admmGraph::ADMMBipartiteGraph)
    error("AbstractADMMAdapter: initialize! is not implemented for $(typeof(adapter))")
end

"""
    updatePenalty(adapter::AbstractADMMAdapter, 
                 info::ADMMIterationInfo, 
                 admmGraph::ADMMBipartiteGraph, 
                 iter::Int64) -> Bool

Update the ADMM penalty parameter based on algorithm progress and adaptation strategy.

This is the core method that every concrete ADMM adapter MUST implement. It analyzes
the current algorithm state and decides whether to modify the penalty parameter ρ
to improve convergence performance.

**Mathematical Background**
The penalty parameter ρ in the augmented Lagrangian:
```math
L_ρ(x,z,λ) = f(x) + g(z) + λ^T(Ax + Bz - c) + \\frac{ρ}{2}\\|Ax + Bz - c\\|_2^2
```

affects the balance between:
- **Constraint enforcement**: Higher ρ penalizes constraint violations more heavily
- **Objective optimization**: Lower ρ allows more focus on the original objective
- **Convergence speed**: Optimal ρ balances primal and dual residual reduction rates

**Arguments**
- `adapter::AbstractADMMAdapter`: The adapter instance
- `info::ADMMIterationInfo`: Current iteration information including residuals and objective
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure (for reference)
- `iter::Int64`: Current iteration number

**Returns**
- `Bool`: `true` if the penalty parameter was updated, `false` otherwise

**Required Implementation Behavior**
1. **Analyze Current State**: Examine primal/dual residuals, objective progress, etc.
2. **Apply Adaptation Logic**: Use adapter-specific strategy to determine if update is needed
3. **Update Penalty Parameter**: If adaptation is triggered, modify ρ in `info.rhoHistory`
4. **Enforce Bounds**: Ensure new ρ is within [`ADMM_MIN_RHO`, `ADMM_MAX_RHO`]
5. **Return Update Status**: Return `true` if ρ was changed, `false` otherwise

**Example Implementation**
```julia
function updatePenalty(adapter::ResidualBalancingAdapter, info::ADMMIterationInfo, 
                      admmGraph::ADMMBipartiteGraph, iter::Int64)
    
    # Only update every N iterations to avoid instability
    if iter - adapter.lastUpdateIteration < adapter.updateInterval
        return false
    end
    
    # Get current residuals
    currentPresL2 = info.presL2[end]
    currentDresL2 = info.dresL2[end]
    currentRho = info.rhoHistory[end][1]
    
    # Residual balancing logic
    residualRatio = currentPresL2 / currentDresL2
    
    if residualRatio > adapter.increaseThreshold
        # Primal residual too large - increase ρ
        newRho = min(currentRho * adapter.increaseFactor, ADMM_MAX_RHO)
    elseif residualRatio < adapter.decreaseThreshold
        # Dual residual too large - decrease ρ
        newRho = max(currentRho / adapter.decreaseFactor, ADMM_MIN_RHO)
    else
        # Residuals are balanced - no change
        return false
    end
    
    # Update penalty parameter
    push!(info.rhoHistory, (newRho, iter))
    adapter.lastUpdateIteration = iter
    
    return true
end
```

**Common Adaptation Strategies**
1. **Residual Balancing**: Adjust ρ to balance primal and dual residual magnitudes
2. **Spectral Methods**: Use eigenvalue estimates to guide parameter selection
3. **Objective-based**: Adapt based on Lagrangian objective progress
4. **Predetermined Schedules**: Follow fixed adaptation patterns
5. **Gradient-based**: Use gradient information for optimal parameter selection

**Update Guidelines**
- **Frequency**: Don't update too often (every 5-10 iterations minimum)
- **Magnitude**: Use moderate scaling factors (2x-10x) to avoid instability
- **Bounds**: Always enforce `ADMM_MIN_RHO ≤ ρ ≤ ADMM_MAX_RHO`
- **Convergence**: Reduce adaptation frequency as algorithm converges

**Side Effects**
- MUST update `info.rhoHistory` if penalty parameter changes
- MAY update adapter internal state for tracking
- SHOULD NOT modify other fields in `info` or `admmGraph`

See also: `initialize!`, `ADMM_MIN_RHO`, `ADMM_MAX_RHO`, `ADMMIterationInfo`
"""
function updatePenalty(adapter::AbstractADMMAdapter, 
                      info::ADMMIterationInfo, 
                      admmGraph::ADMMBipartiteGraph, 
                      iter::Int64)::Bool
    error("AbstractADMMAdapter: updatePenalty is not implemented for $(typeof(adapter))")
end 

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
