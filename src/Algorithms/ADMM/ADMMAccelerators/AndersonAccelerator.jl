# using LinearAlgebra
# using DataStructures

"""
    QRCacheType

Custom type for storing QR factorization data with efficient update capabilities.
"""
mutable struct QRCacheType
    Q::Matrix{Float64}    # Q matrix of the QR factorization
    R::Matrix{Float64}    # R matrix of the QR factorization
    m::Int               # Number of rows
    n::Int               # Maximum number of columns
    
    function QRCacheType(m::Int, n::Int)
        new(Matrix{Float64}(I, m, m), zeros(m, n), m, n)
    end
end

"""
    AndersonConverter

Converter for transforming between ADMM iterates and Anderson acceleration fixed-point iterates.
"""
struct AndersonConverter 
    edges::Vector{String}
    isLeft::Dict{String, Bool}   
    inputBuffer::Dict{String, NumericVariable}
    outputBuffer::Dict{String, NumericVariable}

    AndersonConverter() = new(
        Vector{String}(),
        Dict{String, Bool}(),
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}()
    )
end 

"""
    AndersonAccelerator <: AbstractADMMAccelerator

Implementation of Anderson acceleration for ADMM algorithms with safeguard strategy.

This extends the original Anderson accelerator with a safeguard mechanism that detects
when the acceleration might be causing instability and reverts to previous stable iterates.

**Safeguard Strategy**

The safeguard monitors the residual norm progression and triggers when:
```math
\\|w_k\\| > δ \\|w_{k-1}\\|
```

where:
- `w_k` is the current iterate difference (residual)
- `δ` is the safeguard threshold parameter
- When triggered, the algorithm clears history and reverts to previous iterates

**Algorithm Details**

1. **History Management**: Maintains circular buffers for previous iterates
2. **Residual Monitoring**: Tracks residual norm progression for stability
3. **Safeguard Triggering**: Detects potential instability and takes corrective action
4. **State Reversion**: Reverts to previous stable state when safeguard triggers
5. **Efficient Updates**: Uses QR decomposition with Givens rotations for least squares

**Parameters**
- `historyDepth::Int64`: Number of previous iterates to store and use
- `beta::Float64`: Mixing parameter controlling acceleration strength
- `delta::Float64`: Safeguard threshold parameter (typical values: 0.1-1.0)

**Performance Characteristics**
- **Memory**: O(historyDepth × problem_size)
- **Per-iteration Cost**: O(historyDepth × problem_size)
- **Stability**: Enhanced stability through safeguard mechanism
- **Convergence**: Robust convergence even for ill-conditioned problems

**Example Usage**
```julia
accelerator = AndersonAccelerator(historyDepth=5, beta=1.0, delta=0.1)
initialize!(accelerator, info, admmGraph)

# During ADMM iterations:
accelerateBetweenPrimalUpdates!(accelerator, info, admmGraph)
```
"""
mutable struct AndersonAccelerator <: AbstractADMMAccelerator
    historyDepth::Int64
    beta::Float64
    delta::Float64  # Safeguard threshold parameter
    safeguardTriggered::Bool  # Flag to track safeguard activation
    
    previousZeta::Dict{String, NumericVariable}
    F::CircularDeque{Dict{String, NumericVariable}}
    E::CircularDeque{Dict{String, NumericVariable}}
    zetaTrag::CircularDeque{Dict{String, NumericVariable}}
    wTrag::CircularDeque{Dict{String, NumericVariable}}
    converter::AndersonConverter

    # Cache for QR factorization
    QRCache::Union{QRCacheType, Nothing}
    matrixCache::Union{Matrix{Float64}, Nothing}
    vectorCache::Union{Vector{Float64}, Nothing}
    totalSize::Int64
    vectorKeys::Vector{String}
    
    # ===== EFFICIENCY OPTIMIZATION FIELDS =====
    # Pre-allocated working buffers to avoid repeated allocations
    workingBuffer1::Dict{String, NumericVariable}  # For difference computations
    workingBuffer2::Dict{String, NumericVariable}  # For temporary calculations
    edgeKeyCache::Vector{String}                   # Cached edge keys for fast iteration
    rhoCache::Float64                              # Cached rho value to avoid repeated lookups
    
    """
        AndersonAccelerator(historyDepth::Int64=5, beta::Float64=1.0, delta::Float64=0.1)
    
    Construct a safeguarded Anderson accelerator with specified parameters.
    
    **Arguments**
    - `historyDepth::Int64=5`: Number of previous iterates to store (memory depth)
    - `beta::Float64=1.0`: Mixing parameter controlling acceleration strength
    - `delta::Float64=0.1`: Safeguard threshold parameter
    
    **Parameter Guidelines**
    - `historyDepth`: Typical values 3-10. Higher values use more memory but may improve convergence
    - `beta`: Usually 1.0. Values < 1.0 provide more conservative acceleration
    - `delta`: Safeguard threshold. Smaller values trigger safeguard more aggressively
      * 0.1: Very conservative, triggers safeguard easily
      * 0.5: Moderate safeguarding
      * 1.0: Liberal safeguarding, allows more acceleration
    
    **Memory Usage**
    Total memory scales as `O(historyDepth × problem_size)` where problem_size
    is the total number of dual variables in the ADMM problem.
    """
    AndersonAccelerator(historyDepth::Int64=5, beta::Float64=1.0, delta::Float64=0.1) = new(
        historyDepth, beta, delta, false,
        Dict{String, NumericVariable}(),
        CircularDeque{Dict{String, NumericVariable}}(historyDepth),
        CircularDeque{Dict{String, NumericVariable}}(historyDepth),
        CircularDeque{Dict{String, NumericVariable}}(historyDepth),
        CircularDeque{Dict{String, NumericVariable}}(historyDepth),
        AndersonConverter(),
        nothing, nothing, nothing, 0, String[],
        # Initialize optimization fields as empty - will be set up in initialize!
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}(),
        String[],
        0.0
    )
end

"""
    computeResidualNorm(residual::Dict{String, NumericVariable})

Compute the L2 norm of the residual across all edges.

**Arguments**
- `residual::Dict{String, NumericVariable}`: Dictionary containing residual for each edge

**Returns**
- `Float64`: L2 norm of the combined residual vector

**Mathematical Definition**
```math
\\|w\\| = \\sqrt{\\sum_{e \\in E} \\|w_e\\|_2^2}
```

where `E` is the set of edges and `w_e` is the residual for edge `e`.
"""
function computeResidualNorm(residual::Dict{String, NumericVariable})
    totalNorm = 0.0
    for (edgeID, res) in residual
        totalNorm += LinearAlgebra.norm(res)^2
    end
    return sqrt(totalNorm)
end

"""
    computeResidualNormOptimized(accelerator::AndersonAccelerator, residual::Dict{String, NumericVariable})

Optimized version of computeResidualNorm that uses cached edge keys for faster iteration.

**Performance Benefits**
- Uses cached edge keys to avoid dictionary key iteration overhead
- Faster execution in hot paths like safeguard checking

**Arguments**
- `accelerator::AndersonAccelerator`: Accelerator with cached edge keys
- `residual::Dict{String, NumericVariable}`: Dictionary containing residual for each edge

**Returns**
- `Float64`: L2 norm of the combined residual vector
"""
function computeResidualNormOptimized(accelerator::AndersonAccelerator, residual::Dict{String, NumericVariable})
    totalNorm = 0.0
    for edgeID in accelerator.edgeKeyCache
        totalNorm += LinearAlgebra.norm(residual[edgeID])^2
    end
    return sqrt(totalNorm)
end

"""
    checkSafeguard(accelerator::AndersonAccelerator)

Check if the safeguard condition is triggered.

**Safeguard Condition**
The safeguard is triggered when:
```math
\\|w_k\\| > δ \\|w_{k-1}\\|
```

This indicates that the current residual is significantly larger than the previous one,
suggesting potential instability in the acceleration process.

**Performance Optimizations**
- Uses optimized residual norm computation with cached edge keys
- Faster execution for frequent safeguard checks

**Arguments**
- `accelerator::AndersonAccelerator`: Accelerator with residual history

**Returns**
- `Bool`: `true` if safeguard should be triggered, `false` otherwise

**Implementation Notes**
- Requires at least 2 residual history entries
- Uses L2 norm for residual comparison
- Conservative approach: triggers when residual increases beyond threshold
"""
function checkSafeguard(accelerator::AndersonAccelerator)
    if length(accelerator.wTrag) <= 1
        return false  # Not enough history for safeguard check
    end
    
    # Use optimized residual norm computation for better performance
    currentResidualNorm = computeResidualNormOptimized(accelerator, accelerator.wTrag[1])
    previousResidualNorm = computeResidualNormOptimized(accelerator, accelerator.wTrag[2])
    
    return currentResidualNorm > accelerator.delta * previousResidualNorm
end

"""
    clearHistory!(accelerator::AndersonAccelerator)

Clear the acceleration history when safeguard is triggered.

This follows the notebook logic which clears w_trag, E, and F histories
but keeps zeta_trag for state reversion.

**Cleared Components**
- Residual history (`wTrag`)
- Iterate difference history (`E`) 
- Residual difference history (`F`)
- QR factorization cache

**Post-Condition**
The accelerator is reset to a state equivalent to the first iteration,
allowing for a fresh start of the acceleration process.
"""
function clearHistory!(accelerator::AndersonAccelerator)
    empty!(accelerator.wTrag)
    empty!(accelerator.E)
    empty!(accelerator.F)
    
    # Clear QR cache to force reinitialization
    accelerator.QRCache = nothing
    accelerator.matrixCache = nothing
    accelerator.vectorCache = nothing
    
    # @info "Safeguard triggered: Clearing acceleration history"
end

"""
    revertToStableState!(accelerator::AndersonAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Revert the algorithm state to the previous stable iterate when safeguard is triggered.

This follows the notebook's anderson_revision! logic adapted for the full PDMO data structures:
1. Revert primal/dual variables to previous stable states
2. Recompute the fixed-point variable from reverted state
3. Update the acceleration iterate accordingly

**Mathematical Operations**

The reversion process involves:
1. **State Reversion**: Use previous stable iterates from history
2. **Fixed-point Recomputation**: `ζ = λ + ρ(Ax - b)` from reverted state  
3. **Consistency**: Ensure primal-dual consistency

**Performance Optimizations**
- Uses cached edge keys for faster iteration
- Uses cached rho value to avoid repeated lookups
- Direct edge access instead of dictionary key iteration

**Arguments**
- `accelerator::AndersonAccelerator`: Accelerator with history
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation

**Implementation Notes**
- Adapts notebook logic to work with full PDMO data structures
- Uses available history in zetaTrag for reversion
- Maintains ADMM primal-dual relationships
"""
function revertToStableState!(accelerator::AndersonAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    if length(accelerator.zetaTrag) < 2
        @warn "Insufficient history for state reversion, using current state"
        return
    end
    
    # Revert to previous stable fixed-point iterate (following notebook logic)
    # In notebook: accelerator.converter.zeta = previous stable value
    copytoAnderson!(accelerator.previousZeta, accelerator.zetaTrag[2])
    
    # In the notebook, this would be followed by applyAccelerationToADMM!
    # which updates the dual variables based on the reverted fixed-point iterate
    # For the full PDMO implementation, we need to maintain dual variable consistency
    
    # Optimized version using cached edge keys and rho value
    for edgeID in accelerator.edgeKeyCache
        edge = admmGraph.edges[edgeID]
        rightNodeID = accelerator.converter.isLeft[edge.nodeID1] ? edge.nodeID2 : edge.nodeID1
        
        # Update dual variables to be consistent with reverted fixed-point iterate
        # Following the notebook's applyAccelerationToADMM! logic:
        # λ = ζ + ρ * B * z (where B*z represents the constraint mapping)
        edge.mappings[rightNodeID](info.primalSol[rightNodeID], info.dualSol[edgeID], false)
        axpby!(1.0, accelerator.previousZeta[edgeID], accelerator.rhoCache, info.dualSol[edgeID])
    end
    
    # @info "State reverted to previous stable iterate"
end

"""
    initialize!(accelerator::AndersonAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the Anderson accelerator with the ADMM graph structure and iteration information.
Sets up the converter and initial iterates for the acceleration process.

**Performance Optimizations**
- Pre-allocates working buffers to avoid repeated memory allocations
- Caches edge keys for faster iteration in hot paths
- Caches rho value to avoid repeated dictionary lookups
"""
function initialize!(accelerator::AndersonAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    initialize!(accelerator.converter, info, admmGraph)
    
    # ===== PERFORMANCE OPTIMIZATION SETUP =====
    # Cache edge keys for fast iteration (avoid repeated keys() calls)
    accelerator.edgeKeyCache = collect(accelerator.converter.edges)
    
    # Cache current rho value
    accelerator.rhoCache = info.rhoHistory[end][1]
    
    # Pre-allocate working buffers with correct structure
    for edgeID in accelerator.edgeKeyCache
        accelerator.previousZeta[edgeID] = similar(info.dualSol[edgeID])
        accelerator.workingBuffer1[edgeID] = similar(info.dualSol[edgeID])
        accelerator.workingBuffer2[edgeID] = similar(info.dualSol[edgeID])
    end
    # ===== END OPTIMIZATION SETUP =====

    retrieveIterateInBuffer!(accelerator.converter, info, admmGraph)
    copytoAnderson!(accelerator.previousZeta, accelerator.converter.inputBuffer)
    pushfirst!(accelerator.zetaTrag, deepcopy(accelerator.previousZeta))

    # Use cached rho value
    for edgeID in accelerator.edgeKeyCache
        edge = admmGraph.edges[edgeID]
        rightNodeID = accelerator.converter.isLeft[edge.nodeID1] ? edge.nodeID2 : edge.nodeID1
        edge.mappings[rightNodeID](info.primalSol[rightNodeID], info.dualSol[edgeID], false)
        axpby!(1.0, accelerator.previousZeta[edgeID], accelerator.rhoCache, info.dualSol[edgeID])
    end 
    
    accelerator.safeguardTriggered = false
end

"""
    accelerateBetweenPrimalUpdates!(accelerator::AndersonAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Apply safeguarded Anderson acceleration between primal variable updates.

This follows the exact logic from the intern's notebook implementation with safeguard strategy.

**Algorithm Steps from Notebook**

1. **Compute Fixed-Point Mapping**: `t_zeta = T(zeta)` via `retrieveIterateInBuffer!`
2. **Compute Residual**: `w = t_zeta - zeta`
3. **Safeguard Check**: Monitor residual norm progression
4. **Corrective Action**: Clear history and revert if safeguard triggers
5. **Anderson Acceleration**: Apply acceleration or simple update
6. **Update History**: Store new iterates for future use

**Mathematical Implementation**

The accelerated iterate (when safeguard is not triggered) is:
- First iteration: `ζ_{k+1} = ζ_k + β w_k`
- Subsequent iterations: `ζ_{k+1} = ζ_k + β w_k - (E + β F) γ`

where `γ` solves `F γ = w_k` in least squares sense.

**Performance Optimizations**
- Uses pre-allocated working buffers to eliminate memory allocations
- Caches rho value to avoid repeated lookups
- Optimized difference computations with cached edge keys
- In-place operations wherever possible

**Arguments**
- `accelerator::AndersonAccelerator`: Accelerator with safeguard
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation
"""
function accelerateBetweenPrimalUpdates!(accelerator::AndersonAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    # Update cached rho value if it changed
    currentRho = info.rhoHistory[end][1]
    if currentRho != accelerator.rhoCache
        accelerator.rhoCache = currentRho
    end
    
    # Step 1: Compute fixed-point mapping result T(ζ)
    retrieveIterateInBuffer!(accelerator.converter, info, admmGraph)
    
    # Step 2: Compute residual w = T(ζ) - ζ using optimized version
    w_buffer = computeDifferenceOptimized!(accelerator, accelerator.converter.inputBuffer, accelerator.previousZeta)
    safePushfirst!(accelerator.wTrag, deepcopy(w_buffer))  # Need to copy since we reuse buffer
    
    # Step 3: Safeguard check
    if checkSafeguard(accelerator)
        accelerator.safeguardTriggered = true
        clearHistory!(accelerator)
        revertToStableState!(accelerator, info, admmGraph)
        return  # Exit early, no acceleration this iteration
    end
    
    accelerator.safeguardTriggered = false
    
    # Step 4: Anderson acceleration logic following notebook
    if length(accelerator.E) == 0 
        # First iteration: simple update ζ = ζ + β * w
        # Use cached edge keys for faster iteration
        for edgeID in accelerator.edgeKeyCache
            axpy!(accelerator.beta, accelerator.wTrag[1][edgeID], accelerator.previousZeta[edgeID])
        end
    else 
        # Subsequent iterations: full Anderson acceleration
        
        # Compute F matrix differences: w_{k+1} - w_k using optimized version
        F_diff_buffer = computeDifferenceOptimized!(accelerator, accelerator.wTrag[1], accelerator.wTrag[2])
        safePushfirst!(accelerator.F, deepcopy(F_diff_buffer))  # Need to copy since we reuse buffer
        
        # Solve least squares problem: F γ = w_k (simplified from notebook)
        # Note: Using QR solver for numerical stability instead of direct backslash
        gamma = solveLeastSquareForGamma(accelerator)
        
        # Update ζ = ζ + β * w - (E + β * F) * γ
        # This is equivalent to: ζ = ζ + β * w - E * γ - β * F * γ
        # Optimized version using cached edge keys
        for edgeID in accelerator.edgeKeyCache
            # ζ += β * w
            axpy!(accelerator.beta, accelerator.wTrag[1][edgeID], accelerator.previousZeta[edgeID])
        end
        
        # Apply Anderson mixing with pre-computed coefficients
        for (i, gamma_i) in enumerate(gamma)
            for edgeID in accelerator.edgeKeyCache
                # ζ -= γ_i * E_i
                axpy!(-gamma_i, accelerator.E[i][edgeID], accelerator.previousZeta[edgeID])
                # ζ -= β * γ_i * F_i  
                axpy!(-accelerator.beta * gamma_i, accelerator.F[i][edgeID], accelerator.previousZeta[edgeID])
            end
        end
    end 

    # Step 5: Update history for next iteration
    safePushfirst!(accelerator.zetaTrag, deepcopy(accelerator.previousZeta))
    
    # Compute E matrix differences: ζ_{k+1} - ζ_k using optimized version
    if length(accelerator.zetaTrag) >= 2
        E_diff_buffer = computeDifferenceOptimized!(accelerator, accelerator.zetaTrag[1], accelerator.zetaTrag[2])
        safePushfirst!(accelerator.E, deepcopy(E_diff_buffer))  # Need to copy since we reuse buffer
    end
    
    # Step 6: Store result for ADMM
    copytoAnderson!(accelerator.converter.outputBuffer, accelerator.previousZeta)
end

"""
    accelerateAfterDualUpdates!(accelerator::AndersonAccelerator, info::ADMMIterationInfo)

Apply safeguarded Anderson acceleration after dual variable updates.
This is a placeholder function that currently does nothing.
"""
function accelerateAfterDualUpdates!(accelerator::AndersonAccelerator, info::ADMMIterationInfo)
    return 
end

"""
    getSafeguardStatus(accelerator::AndersonAccelerator)

Get the current safeguard status information.

**Returns**
- `NamedTuple`: Contains safeguard status information:
  * `triggered::Bool`: Whether safeguard was triggered in last iteration
  * `delta::Float64`: Current safeguard threshold
  * `historyLength::Int`: Current acceleration history length
  * `residualNorm::Float64`: Current residual norm (if available)

**Performance Optimizations**
- Uses optimized residual norm computation for faster execution
"""
function getSafeguardStatus(accelerator::AndersonAccelerator)
    # Use optimized residual norm computation for better performance
    residualNorm = length(accelerator.wTrag) > 0 ? computeResidualNormOptimized(accelerator, accelerator.wTrag[1]) : 0.0
    
    return (
        triggered = accelerator.safeguardTriggered,
        delta = accelerator.delta,
        historyLength = length(accelerator.E),
        residualNorm = residualNorm
    )
end

"""
    updateSafeguardThreshold!(accelerator::AndersonAccelerator, newDelta::Float64)

Update the safeguard threshold parameter during optimization.

This allows for adaptive safeguarding where the threshold can be adjusted
based on the problem characteristics or convergence behavior.

**Arguments**
- `accelerator::AndersonAccelerator`: Accelerator to update
- `newDelta::Float64`: New safeguard threshold value

**Parameter Guidelines**
- Smaller values (0.1-0.5): More conservative, triggers safeguard easily
- Larger values (0.5-1.0): More aggressive, allows more acceleration
- Values > 1.0: Very liberal, safeguard rarely triggers

**Example Usage**
```julia
# Start with conservative safeguarding
accelerator = AndersonAccelerator(delta=0.1)

# Later, if convergence is stable, allow more acceleration
updateSafeguardThreshold!(accelerator, 0.5)
```
"""
function updateSafeguardThreshold!(accelerator::AndersonAccelerator, newDelta::Float64)
    if newDelta <= 0.0
        throw(ArgumentError("Safeguard threshold must be positive"))
    end
    
    oldDelta = accelerator.delta
    accelerator.delta = newDelta
    
    # @info "Safeguard threshold updated from $oldDelta to $newDelta"
end

# ===============================================================================
# Helper Types and Functions from AndersonAccelerator.jl
# ===============================================================================

"""
    initialize!(converter::AndersonConverter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the Anderson converter with the ADMM graph structure and iteration information.
"""
function initialize!(converter::AndersonConverter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    @assert isempty(admmGraph.edges) == false "AndersonConverter: No edges to convert."
    for edgeID in keys(admmGraph.edges) 
        if haskey(info.dualSol, edgeID) == false 
            error("AndersonConverter: Edge $edgeID does not exist in the dual solution.")
        end 
        push!(converter.edges, edgeID)
        converter.inputBuffer[edgeID] = similar(info.dualSol[edgeID])
        converter.outputBuffer[edgeID] = similar(info.dualSol[edgeID])
    end 

    for nodeID in admmGraph.left 
        converter.isLeft[nodeID] = true
    end 
    for nodeID in admmGraph.right 
        converter.isLeft[nodeID] = false 
    end 
end 

""" 
    retrieveIterateInBuffer!(converter::AndersonConverter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Transform ADMM dual variables to fixed-point form and store in converter's input buffer.
"""
function retrieveIterateInBuffer!(converter::AndersonConverter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    rho = info.rhoHistory[end][1]
    for edgeID in converter.edges 
        edge = admmGraph.edges[edgeID]
        leftNodeID = converter.isLeft[edge.nodeID1] ? edge.nodeID1 : edge.nodeID2
        edge.mappings[leftNodeID](info.primalSol[leftNodeID], converter.inputBuffer[edgeID], false) # inputBuffer <- Ax
        axpy!(-1.0, edge.rhs, converter.inputBuffer[edgeID])                  # inputBuffer <- Ax-b 
        axpby!(1.0, info.dualSol[edgeID], rho, converter.inputBuffer[edgeID]) # inputBuffer <- y + rho (Ax-b)
    end 
end 

"""
    safePushfirst!(deq::CircularDeque{T}, item::T) where T

Safely add an item to the front of a circular deque, removing the oldest item if the deque is at capacity.
"""
function safePushfirst!(deq::CircularDeque{T}, item::T) where T
    if length(deq) == capacity(deq)
        pop!(deq)
    end 
    pushfirst!(deq, item)
end 

"""
    copytoAnderson!(toBuffer::Dict{String, NumericVariable}, fromBuffer::Dict{String, NumericVariable})

Copy all values from fromBuffer to toBuffer for each edge ID.
"""
function copytoAnderson!(toBuffer::Dict{String, NumericVariable}, fromBuffer::Dict{String, NumericVariable})
    for edgeID in keys(toBuffer)
        copyto!(toBuffer[edgeID], fromBuffer[edgeID])
    end 
end 

"""
    axpbyAnderson!(alpha::Float64, v1::Dict{String, NumericVariable}, beta::Float64, v2::Dict{String, NumericVariable})

Compute alpha*v1 + beta*v2 and store the result in v2 for each edge ID.
"""
function axpbyAnderson!(alpha::Float64, v1::Dict{String, NumericVariable}, beta::Float64, v2::Dict{String, NumericVariable})
    for edgeID in keys(v1)
        axpby!(alpha, v1[edgeID], beta, v2[edgeID])
    end 
end 

"""
    axpyAnderson!(alpha::Float64, v1::Dict{String, NumericVariable}, v2::Dict{String, NumericVariable})

Compute v2 += alpha*v1 for each edge ID.
"""
function axpyAnderson!(alpha::Float64, v1::Dict{String, NumericVariable}, v2::Dict{String, NumericVariable})
    for edgeID in keys(v1)
        axpy!(alpha, v1[edgeID], v2[edgeID])
    end 
end 

"""
    computeDifferenceInBuffer!(buffer::Dict{String, NumericVariable},
                              v1::Dict{String, NumericVariable}, 
                              v2::Dict{String, NumericVariable})

Compute v1 - v2 and store the result in buffer for each edge ID.
Optimized version that uses pre-allocated buffer to avoid memory allocations.
"""
function computeDifferenceInBuffer!(buffer::Dict{String, NumericVariable},
    v1::Dict{String, NumericVariable}, 
    v2::Dict{String, NumericVariable})
    # Fast path: use cached edge keys if available
    if haskey(v1, first(keys(v1))) # Quick check that v1 is not empty
        for edgeID in keys(v1)
            copyto!(buffer[edgeID], v1[edgeID])
            axpy!(-1.0, v2[edgeID], buffer[edgeID])
        end
    end
end 

"""
    computeDifference(v1::Dict{String, NumericVariable}, v2::Dict{String, NumericVariable})

Create a new buffer and compute v1 - v2 for each edge ID.
Returns the difference as a new dictionary.

**Performance Note**: This function creates new allocations. For better performance,
use `computeDifferenceOptimized!` with pre-allocated buffers when possible.
"""
function computeDifference(v1::Dict{String, NumericVariable}, v2::Dict{String, NumericVariable})
    buffer = Dict{String, NumericVariable}()
    for edgeID in keys(v1)
        buffer[edgeID] = similar(v1[edgeID])
    end 
    computeDifferenceInBuffer!(buffer, v1, v2)
    return buffer 
end

"""
    computeDifferenceOptimized!(accelerator::AndersonAccelerator, 
                               v1::Dict{String, NumericVariable}, 
                               v2::Dict{String, NumericVariable})

Optimized version that uses pre-allocated working buffer to compute v1 - v2.
Returns reference to the working buffer for immediate use.

**Performance Benefits**
- Zero memory allocations for difference computation
- Uses cached edge keys for faster iteration
- In-place operations for maximum efficiency
"""
function computeDifferenceOptimized!(accelerator::AndersonAccelerator,
    v1::Dict{String, NumericVariable}, 
    v2::Dict{String, NumericVariable})
    
    # Use pre-allocated working buffer
    buffer = accelerator.workingBuffer1
    
    # Fast iteration using cached edge keys
    for edgeID in accelerator.edgeKeyCache
        copyto!(buffer[edgeID], v1[edgeID])
        axpy!(-1.0, v2[edgeID], buffer[edgeID])
    end
    
    return buffer
end

"""
    initializeQR!(A::Matrix{Float64}, maxCols::Int)

Initialize a QR factorization cache for matrix A with a maximum number of columns.
"""
function initializeQR!(A::Matrix{Float64}, maxCols::Int)
    m, n = size(A)
    @assert n <= maxCols "Initial matrix has more columns than maximum allowed"
    cache = QRCacheType(m, maxCols)  # Initialize with maximum possible columns
    cache.R[1:m, 1:n] = copy(A)
    
    for j in 1:n
        for i in m:-1:j+1
            G, _ = givens(cache.R[j,j], cache.R[i,j], j, i)
            lmul!(G, cache.R)
            rmul!(cache.Q, G')
        end
    end
    
    return cache
end

"""
    updateQRFactorization!(cache::QRCacheType, A::Matrix{Float64})

Update QR factorization to match the column shifting in matrixCache.
"""
function updateQRFactorization!(cache::QRCacheType, A::Matrix{Float64})
    m, n = size(A)
    @assert m == cache.m "Matrix row dimension mismatch"
    @assert n <= cache.n "Matrix column dimension exceeds maximum"
    
    # 1. Shift columns right in R (matches matrixCache shift)
    if n > 1
        cache.R[:, 2:n] .= cache.R[:, 1:n-1]
    end
    
    # 2. Insert new column from A[:, 1] (newest data)
    cache.R[:, 1] = cache.Q' * A[:, 1]
    
    # 3. Restore upper triangular form using Givens rotations
    for j in 1:n
        for i in m:-1:j+1
            if !iszero(cache.R[i,j])
                G, _ = givens(cache.R[j,j], cache.R[i,j], j, i)
                lmul!(G, cache.R)
                rmul!(cache.Q, G')
            end
        end
    end
end

"""
    uppercaseSolve!(R::Matrix{Float64}, b::Vector{Float64})

Solve the upper triangular system Rx = b using back substitution.
"""
function uppercaseSolve!(R::Matrix{Float64}, b::Vector{Float64})
    n = length(b)
    x = similar(b)
    
    for i in n:-1:1
        sum = b[i]
        for j in (i+1):n
            sum -= R[i,j] * x[j]
        end
        x[i] = sum / R[i,i]
    end
    
    return x
end

"""
    initializeLeastSquaresSolver!(accelerator::AndersonAccelerator, dict::Dict{String, NumericVariable})

Initialize the least squares solver components of the Anderson accelerator.
"""
function initializeLeastSquaresSolver!(accelerator::AndersonAccelerator, dict::Dict{String, NumericVariable})
    accelerator.totalSize = sum(length(LinearAlgebra.vec(dict[id])) for id in keys(dict))
    accelerator.vectorKeys = sort(collect(keys(dict)))
    accelerator.matrixCache = zeros(accelerator.totalSize, accelerator.historyDepth)
    accelerator.vectorCache = zeros(accelerator.totalSize)
    accelerator.QRCache = nothing
end

"""
    updateMatrixColumn!(accelerator::AndersonAccelerator, dict::Dict{String, NumericVariable}, col::Int)

Update a specific column of the matrix cache with values from the dictionary.
"""
function updateMatrixColumn!(accelerator::AndersonAccelerator, dict::Dict{String, NumericVariable}, col::Int)
    rowIdx = 1
    for id in accelerator.vectorKeys
        vecLength = length(LinearAlgebra.vec(dict[id]))
        view(accelerator.matrixCache, rowIdx:rowIdx+vecLength-1, col) .= LinearAlgebra.vec(dict[id])
        rowIdx += vecLength
    end
end

"""
    updateVectorCache!(accelerator::AndersonAccelerator, dict::Dict{String, NumericVariable})

Update the vector cache with values from the dictionary.
"""
function updateVectorCache!(accelerator::AndersonAccelerator, dict::Dict{String, NumericVariable})
    rowIdx = 1
    for id in accelerator.vectorKeys
        vecLength = length(LinearAlgebra.vec(dict[id]))
        accelerator.vectorCache[rowIdx:rowIdx+vecLength-1] .= LinearAlgebra.vec(dict[id])
        rowIdx += vecLength
    end
end

"""
    solveLeastSquareForGamma(accelerator::AndersonAccelerator)

Solve the least squares problem for Anderson acceleration coefficients.
"""
function solveLeastSquareForGamma(accelerator::AndersonAccelerator)
    if accelerator.matrixCache === nothing
        initializeLeastSquaresSolver!(accelerator, accelerator.wTrag[1])
    end
    
    ncols = min(length(accelerator.F), accelerator.historyDepth)
    
    # 1. Shift existing columns right in matrixCache
    if ncols > 1
        accelerator.matrixCache[:, 2:ncols] .= accelerator.matrixCache[:, 1:ncols-1]
    end
    
    # 2. Update first column with newest data
    updateMatrixColumn!(accelerator, accelerator.F[1], 1)
    
    # 3. Update or initialize QR factorization
    if accelerator.QRCache === nothing
        accelerator.QRCache = initializeQR!(accelerator.matrixCache[:, 1:ncols], accelerator.historyDepth)
    else
        updateQRFactorization!(accelerator.QRCache, accelerator.matrixCache[:, 1:ncols])
    end
    
    updateVectorCache!(accelerator, accelerator.wTrag[1])
    
    # Solve the least squares problem
    b_rotated = accelerator.QRCache.Q' * accelerator.vectorCache
    gamma = uppercaseSolve!(accelerator.QRCache.R[1:ncols, 1:ncols], b_rotated[1:ncols])
    
    return gamma
end 