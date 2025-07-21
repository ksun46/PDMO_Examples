"""
    AbstractADMMAccelerator

Abstract base type for ADMM acceleration schemes that can improve convergence rates.

ADMM accelerators implement various techniques to speed up the convergence of the
Alternating Direction Method of Multipliers (ADMM) algorithm. Common approaches include:
- Anderson acceleration (extrapolation methods)
- Halpern-type acceleration schemes
- Null acceleration (no acceleration applied)

**Required Interface Methods**

All concrete subtypes must implement:
- `initialize!(accelerator, info, admmGraph)`: Initialize the accelerator with ADMM problem data
- `accelerateBetweenPrimalUpdates!(accelerator, info, admmGraph)`: Apply acceleration between primal updates
- `accelerateAfterDualUpdates!(accelerator, info)`: Apply acceleration after dual updates

**Mathematical Background**

ADMM acceleration works by combining information from multiple previous iterations
to compute improved estimates of the solution. The general form can be written as:

```math
x_{k+1} = f(x_k, x_{k-1}, ..., x_{k-m})
```

where `f` is the acceleration function and `m` is the memory depth.

**Performance Considerations**

- Different accelerators have varying memory requirements
- Some accelerators may increase per-iteration cost while reducing iteration count
- The effectiveness depends on problem structure and conditioning



**See Also**
- `AndersonAccelerator`: Implementation of Anderson acceleration
- `AutoHalpernAccelerator`: Implementation of Halpern-type acceleration
- `NullAccelerator`: No-operation accelerator for baseline comparison
"""
abstract type AbstractADMMAccelerator end 

"""
    NullAccelerator <: AbstractADMMAccelerator

A no-operation accelerator that maintains default ADMM behavior without any acceleration.

This accelerator serves as a baseline for comparison with other acceleration schemes.
All methods are implemented as no-ops, so the ADMM algorithm proceeds with its
standard update rules.

**Use Cases**
- Baseline performance comparison
- Debugging and testing ADMM implementations
- Scenarios where acceleration is not desired or beneficial

**Performance**
- Zero computational overhead
- No memory usage beyond the accelerator object itself
- Identical convergence behavior to unaccelerated ADMM


"""
mutable struct NullAccelerator <: AbstractADMMAccelerator
    NullAccelerator() = new()
end

"""
    initialize!(accelerator::NullAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the null accelerator (no-operation).

**Arguments**
- `accelerator::NullAccelerator`: The accelerator instance
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of the ADMM problem

**Implementation**
This function performs no operations and returns immediately.
"""
function initialize!(accelerator::NullAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    return 
end 

"""
    accelerateBetweenPrimalUpdates!(accelerator::NullAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Apply acceleration between primal updates (no-operation for null accelerator).

**Arguments**
- `accelerator::NullAccelerator`: The accelerator instance
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of the ADMM problem

**Implementation**
This function performs no operations and returns immediately, allowing
the ADMM algorithm to proceed with its standard primal updates.
"""
function accelerateBetweenPrimalUpdates!(accelerator::NullAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    return 
end 

"""
    accelerateAfterDualUpdates!(accelerator::NullAccelerator, info::ADMMIterationInfo)

Apply acceleration after dual updates (no-operation for null accelerator).

**Arguments**
- `accelerator::NullAccelerator`: The accelerator instance
- `info::ADMMIterationInfo`: Current ADMM iteration information

**Implementation**
This function performs no operations and returns immediately, allowing
the ADMM algorithm to proceed with its standard dual updates.
"""
function accelerateAfterDualUpdates!(accelerator::NullAccelerator, info::ADMMIterationInfo)
    return 
end 

include("AutoHalpernAccelerator.jl")
include("AndersonAccelerator.jl")

"""
    getADMMAcceleratorName(accelerator::AbstractADMMAccelerator)

Get a string identifier for the accelerator type.

**Arguments**
- `accelerator::AbstractADMMAccelerator`: The accelerator instance

**Returns**
- `String`: A string identifier for the accelerator type:
  - `"NULL_ACCELERATOR"` for `NullAccelerator`
  - `"AUTO_HALPERN_ACCELERATOR"` for `AutoHalpernAccelerator`
  - `"ANDERSON_ACCELERATOR"` for `AndersonAccelerator`
  - `"UNKNOWN_ACCELERATOR"` for unrecognized types



**Usage**
This function is commonly used for:
- Logging and debugging
- Performance tracking and comparison
- Configuration management
- Result reporting
"""
function getADMMAcceleratorName(accelerator::AbstractADMMAccelerator)
    if typeof(accelerator) == NullAccelerator
        return "NULL_ACCELERATOR"
    elseif typeof(accelerator) == AutoHalpernAccelerator
        return "AUTO_HALPERN_ACCELERATOR"
    elseif typeof(accelerator) == AndersonAccelerator
        return "ANDERSON_ACCELERATOR"
    else 
        return "UNKNOWN_ACCELERATOR"
    end 
end 
