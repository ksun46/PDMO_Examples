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
    initialize!(accelerator::AbstractADMMAccelerator, 
               info::ADMMIterationInfo, 
               admmGraph::ADMMBipartiteGraph)

Initialize the ADMM accelerator with problem data and algorithm state.

This method is called once before the main ADMM iterations begin. It allows accelerators
to perform preprocessing, analyze problem structure, allocate acceleration workspace,
and set up any accelerator-specific data structures or parameters.

**Arguments**
- `accelerator::AbstractADMMAccelerator`: The accelerator instance to initialize
- `info::ADMMIterationInfo`: Initial iteration information and algorithm state
- `admmGraph::ADMMBipartiteGraph`: The ADMM bipartite graph structure

**Required Implementation**
Every concrete ADMM accelerator MUST implement this method. The implementation should handle:

1. **Problem Structure Analysis**: Examine the bipartite graph and variable dimensions
2. **Memory Allocation**: Allocate workspace for storing iteration history
3. **Parameter Initialization**: Set up accelerator-specific parameters and thresholds
4. **History Setup**: Initialize storage for previous iterates needed for acceleration
5. **Convergence Tracking**: Set up metrics for monitoring acceleration effectiveness

**Example Implementation**
```julia
function initialize!(accelerator::MyConcreteAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    # Analyze problem dimensions
    accelerator.numNodes = length(admmGraph.nodes)
    accelerator.variableDims = Dict{String, Int}()
    for (nodeID, node) in admmGraph.nodes
        accelerator.variableDims[nodeID] = length(node.val)
    end
    
    # Allocate acceleration workspace
    accelerator.iterateHistory = Vector{Dict{String, NumericVariable}}()
    accelerator.residualHistory = Vector{Float64}()
    
    # Initialize acceleration parameters
    accelerator.memoryDepth = min(accelerator.maxMemory, 10)  # Start conservatively
    accelerator.activationThreshold = 1e-3
    accelerator.lastAccelerationIteration = 0
    
    # Set up convergence tracking
    accelerator.accelerationMetrics = AccelerationMetrics()
    accelerator.initialized = true
end
```

**Common Initialization Tasks**
- **Anderson Accelerators**: Set up QR factorization workspace and residual matrices
- **Halpern Accelerators**: Initialize step size sequences and extrapolation parameters
- **Momentum Methods**: Set up momentum buffers and decay parameters
- **Adaptive Accelerators**: Initialize adaptation thresholds and performance tracking

**Memory Management**
- Allocate sufficient workspace for the chosen memory depth
- Consider memory vs. performance trade-offs for large problems
- Pre-allocate all buffers to avoid allocations during iterations

**Error Handling**
- Validate that accelerator parameters are reasonable
- Check for sufficient memory for the chosen acceleration scheme
- Ensure problem structure is compatible with the accelerator

See also: `accelerateBetweenPrimalUpdates!`, `accelerateAfterDualUpdates!`, `ADMMIterationInfo`
"""
function initialize!(accelerator::AbstractADMMAccelerator, 
                    info::ADMMIterationInfo, 
                    admmGraph::ADMMBipartiteGraph)
    error("AbstractADMMAccelerator: initialize! is not implemented for $(typeof(accelerator))")
end

"""
    accelerateBetweenPrimalUpdates!(accelerator::AbstractADMMAccelerator, 
                                   info::ADMMIterationInfo, 
                                   admmGraph::ADMMBipartiteGraph)

Apply acceleration between primal variable updates in the ADMM algorithm.

This method is called during ADMM iterations between primal variable updates to apply
acceleration techniques that can improve convergence rates. The timing allows accelerators
to modify primal variables or related quantities before the next set of primal updates.

**Mathematical Background**
Acceleration between primal updates typically involves extrapolation or momentum techniques:
```math
x^{k+1} = x^k + β_k (x^k - x^{k-1}) + \\text{correction terms}
```

Common approaches include:
- **Anderson Acceleration**: Linear combination of previous iterates to minimize residuals
- **Heavy Ball Methods**: Momentum-based extrapolation using previous steps
- **Nesterov-type**: Predictive updates based on gradient information

**Arguments**
- `accelerator::AbstractADMMAccelerator`: The accelerator instance
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure

**Required Implementation Behavior**
1. **Check Activation Conditions**: Determine if acceleration should be applied this iteration
2. **Update History**: Store current iterates in acceleration memory
3. **Compute Acceleration**: Apply accelerator-specific extrapolation or combination
4. **Update Variables**: Modify primal variables in `info.primalSol` if acceleration is applied
5. **Track Performance**: Monitor acceleration effectiveness for adaptive schemes

**Example Implementation**
```julia
function accelerateBetweenPrimalUpdates!(accelerator::AndersonAccelerator, 
                                        info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    
    # Check if we have enough history and should activate
    currentIter = length(info.lagrangianObj)
    if currentIter < accelerator.memoryDepth || !shouldActivate(accelerator, info)
        # Store current iterate and return
        storeIterate!(accelerator, info.primalSol, currentIter)
        return
    end
    
    # Compute residuals for Anderson combination
    residuals = computeResiduals(accelerator, info, admmGraph)
    
    # Solve least squares problem for combination weights
    weights = computeAndersonWeights(accelerator, residuals)
    
    # Apply Anderson extrapolation
    acceleratedSol = Dict{String, NumericVariable}()
    for (nodeID, _) in admmGraph.nodes
        acceleratedSol[nodeID] = computeWeightedCombination(accelerator, nodeID, weights)
    end
    
    # Update primal solution with accelerated values
    for (nodeID, acceleratedVal) in acceleratedSol
        info.primalSol[nodeID] = acceleratedVal
    end
    
    # Update acceleration metrics
    updateAccelerationMetrics!(accelerator, info)
end
```

**Timing Considerations**
- Called between left and right primal updates (or equivalent)
- Should be computationally efficient to avoid dominating iteration cost
- May skip acceleration on early iterations when insufficient history exists

**Side Effects**
- MAY modify `info.primalSol` with accelerated values
- SHOULD update accelerator internal state and history
- MUST NOT modify `admmGraph` or other algorithm state

**Performance Guidelines**
- Keep acceleration overhead low (< 10% of iteration time)
- Use efficient linear algebra operations for combinations
- Consider adaptive activation based on convergence progress

See also: `initialize!`, `accelerateAfterDualUpdates!`, `ADMMIterationInfo`
"""
function accelerateBetweenPrimalUpdates!(accelerator::AbstractADMMAccelerator, 
                                        info::ADMMIterationInfo, 
                                        admmGraph::ADMMBipartiteGraph)
    error("AbstractADMMAccelerator: accelerateBetweenPrimalUpdates! is not implemented for $(typeof(accelerator))")
end

"""
    accelerateAfterDualUpdates!(accelerator::AbstractADMMAccelerator, 
                               info::ADMMIterationInfo)

Apply acceleration after dual variable updates in the ADMM algorithm.

This method is called after dual variable updates to apply acceleration techniques
that can improve convergence rates. This timing allows accelerators to modify
dual variables or prepare for the next iteration's acceleration.

**Mathematical Background**
Acceleration after dual updates typically focuses on dual variable extrapolation:
```math
λ^{k+1} = λ^k + α_k (λ^k - λ^{k-1}) + \\text{correction terms}
```

or preparation for primal acceleration in the next iteration. Common approaches include:
- **Dual Variable Momentum**: Direct extrapolation of dual variables
- **Residual-based Acceleration**: Using dual residual information for next iteration
- **Converter Updates**: Preparing accelerated quantities for specialized solvers

**Arguments**
- `accelerator::AbstractADMMAccelerator`: The accelerator instance
- `info::ADMMIterationInfo`: Current iteration information including updated dual variables

**Required Implementation Behavior**
1. **Analyze Dual Updates**: Examine the quality and magnitude of dual variable changes
2. **Update Acceleration State**: Store dual information needed for future acceleration
3. **Apply Dual Acceleration**: Modify dual variables if the accelerator operates on them
4. **Prepare Next Iteration**: Set up any quantities needed for next primal acceleration
5. **Convergence Assessment**: Update metrics for adaptive acceleration schemes

**Example Implementation**
```julia
function accelerateAfterDualUpdates!(accelerator::HalpernAccelerator, info::ADMMIterationInfo)
    
    currentIter = length(info.lagrangianObj)
    
    # Update dual variable history
    storeDualIterate!(accelerator, info.dualSol, currentIter)
    
    # Check if dual acceleration should be applied
    if currentIter >= accelerator.minIterations && shouldAccelerateDual(accelerator, info)
        
        # Compute Halpern step size
        stepSize = computeHalpernStepSize(accelerator, currentIter)
        
        # Apply dual variable extrapolation
        for (edgeID, dualVar) in info.dualSol
            prevDual = accelerator.dualHistory[end-1][edgeID]
            extrapolatedDual = dualVar + stepSize * (dualVar - prevDual)
            info.dualSol[edgeID] = extrapolatedDual
        end
        
        # Update step size sequence for next iteration
        updateStepSizeSequence!(accelerator, currentIter)
    end
    
    # Prepare converter output for specialized solvers (e.g., Anderson)
    if hasConverter(accelerator)
        updateConverterOutput!(accelerator, info)
    end
end
```

**Dual Variable Considerations**
- Dual acceleration can be more delicate than primal acceleration
- May require different step sizes or activation criteria
- Should maintain dual feasibility when possible

**Converter Integration**
Some accelerators (like Anderson) use "converters" that prepare special dual quantities
for use by subproblem solvers. This method often handles converter updates.

**Side Effects**
- MAY modify `info.dualSol` with accelerated dual variables
- SHOULD update accelerator internal state and dual history
- MAY update converter outputs for specialized subproblem solvers

**Performance Guidelines**
- Dual acceleration should be even more lightweight than primal acceleration
- Consider the impact on subproblem solver performance
- Monitor numerical stability of dual variable modifications

See also: `initialize!`, `accelerateBetweenPrimalUpdates!`, `ADMMIterationInfo`
"""
function accelerateAfterDualUpdates!(accelerator::AbstractADMMAccelerator, 
                                    info::ADMMIterationInfo)
    error("AbstractADMMAccelerator: accelerateAfterDualUpdates! is not implemented for $(typeof(accelerator))")
end 

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
