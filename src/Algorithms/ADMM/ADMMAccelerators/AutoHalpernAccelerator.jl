"""
    HalpernConverter

Helper structure for converting between ADMM iterates and fixed-point iterates for Halpern acceleration.

The Halpern acceleration method requires treating ADMM updates as fixed-point iterations.
This converter manages the transformation between ADMM variables (both primal and dual)
and the fixed-point representation needed for Halpern-type acceleration.

**Mathematical Foundation**

Unlike Anderson acceleration (which focuses on dual variables), Halpern acceleration
can work with both primal and dual variables directly. The fixed-point form is:
```math
x_{k+1} = T(x_k)
```
where `T` is the ADMM operator and `x` represents the combined primal-dual variables.

**Fields**
- `nodes::Vector{String}`: Node IDs whose primal variables are included in fixed-point iteration
- `edges::Vector{String}`: Edge IDs whose dual variables are included in fixed-point iteration
- `inputBuffer::Dict{String, NumericVariable}`: Buffer for input variables to fixed-point iteration
- `outputBuffer::Dict{String, NumericVariable}`: Buffer for output variables after acceleration

**Design Principles**

1. **Variable Selection**: Allows selective inclusion of primal and dual variables
2. **Buffer Management**: Maintains separate input/output buffers for efficient computation
3. **Disjoint Sets**: Ensures nodes and edges don't overlap to avoid conflicts

**Usage Pattern**
```julia
converter = HalpernConverter()
initialize!(converter, selected_nodes, selected_edges, info)
retrieveIterateInBuffer!(converter, info)
# Apply acceleration...
applyAccelerationToADMM!(converter, info)
```

**Performance Considerations**
- Memory usage scales with the number of selected variables
- Conversion overhead is O(number of selected variables)
- Buffer reuse minimizes memory allocations
"""
struct HalpernConverter 
    nodes::Vector{String} 
    edges::Vector{String} 
    inputBuffer::Dict{String, NumericVariable}   
    outputBuffer::Dict{String, NumericVariable}  

    HalpernConverter() = new(
        Vector{String}(),
        Vector{String}(),
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}()
    )
end 

"""
    initialize!(converter::HalpernConverter, nodes::Vector{String}, edges::Vector{String}, info::ADMMIterationInfo)

Initialize the HalpernConverter with specified nodes and edges for acceleration.

**Arguments**
- `converter::HalpernConverter`: Converter instance to initialize
- `nodes::Vector{String}`: Node IDs for primal variables to include
- `edges::Vector{String}`: Edge IDs for dual variables to include
- `info::ADMMIterationInfo`: Current ADMM iteration information

**Validation**
- Ensures nodes and edges are disjoint sets (no overlapping IDs)
- Verifies all specified IDs exist in the current solution
- Requires at least one node or edge to be specified

**Initialization Process**
1. **Validation**: Check input constraints and existence
2. **Buffer Creation**: Allocate input/output buffers for each variable
3. **Registration**: Store node and edge IDs for future reference

**Throws**
- `AssertionError`: If nodes and edges overlap
- `AssertionError`: If no nodes or edges specified
- `ErrorException`: If specified IDs don't exist in solution

**Example**
```julia
converter = HalpernConverter()
initialize!(converter, ["node1", "node2"], ["edge1", "edge2"], info)
```
"""
function initialize!(converter::HalpernConverter, 
    nodes::Vector{String}, 
    edges::Vector{String}, 
    info::ADMMIterationInfo)

    @assert isempty(intersect(Set(nodes), Set(edges))) "HalpernConverter: Nodes and edges must be disjoint."
    @assert length(nodes) + length(edges) > 0 "HalpernConverter: No nodes or edges to convert."

    for nodeID in nodes 
        if haskey(info.primalSol, nodeID) == false 
            error("HalpernConverter: Node $nodeID does not exist in the primal solution.")
        end
        push!(converter.nodes, nodeID)
        converter.inputBuffer[nodeID] = similar(info.primalSol[nodeID])
        converter.outputBuffer[nodeID] = similar(info.primalSol[nodeID])
    end 

    for edgeID in edges 
        if haskey(info.dualSol, edgeID) == false 
            error("HalpernConverter: Edge $edgeID does not exist in the dual solution.")
        end
        push!(converter.edges, edgeID)
        converter.inputBuffer[edgeID] = similar(info.dualSol[edgeID])
        converter.outputBuffer[edgeID] = similar(info.dualSol[edgeID])
    end 
end

"""
    retrieveIterateInBuffer!(converter::HalpernConverter, info::ADMMIterationInfo)

Copy current ADMM variables into the converter's input buffer.

This function performs a direct copy of the current ADMM iterates into the
input buffer, preparing them for Halpern acceleration. Unlike Anderson acceleration,
no mathematical transformation is required.

**Arguments**
- `converter::HalpernConverter`: Converter with initialized buffers
- `info::ADMMIterationInfo`: Current ADMM iteration information

**Operation**
- Copies primal variables from specified nodes
- Copies dual variables from specified edges
- Uses efficient `copyto!` operations for memory management

**Performance**
- O(total_variable_size) time complexity
- No additional memory allocation
- In-place buffer updates
"""
function retrieveIterateInBuffer!(converter::HalpernConverter, info::ADMMIterationInfo)
    for nodeID in converter.nodes 
        copyto!(converter.inputBuffer[nodeID], info.primalSol[nodeID])
    end 
    for edgeID in converter.edges 
        copyto!(converter.inputBuffer[edgeID], info.dualSol[edgeID])
    end  
end 

"""
    applyAccelerationToADMM!(converter::HalpernConverter, info::ADMMIterationInfo)

Apply accelerated iterates from output buffer back to ADMM variables.

After Halpern acceleration computes improved estimates in the output buffer,
this function copies them back to the ADMM iteration information structure.

**Arguments**
- `converter::HalpernConverter`: Converter with accelerated iterates in output buffer
- `info::ADMMIterationInfo`: ADMM iteration information to update

**Operation**
- Updates primal variables for specified nodes
- Updates dual variables for specified edges
- Maintains consistency with ADMM algorithm structure

**Side Effects**
- Modifies `info.primalSol` and `info.dualSol` with accelerated values
- Changes affect subsequent ADMM iterations
- Updates are applied immediately without validation
"""
function applyAccelerationToADMM!(converter::HalpernConverter, info::ADMMIterationInfo)
    for nodeID in converter.nodes 
        copyto!(info.primalSol[nodeID], converter.outputBuffer[nodeID])
    end 
    for edgeID in converter.edges 
        copyto!(info.dualSol[edgeID], converter.outputBuffer[edgeID])
    end 
end 

# Define a type alias for boolean variables 
const NumericVariableBool = Union{Bool, AbstractArray{Bool, N} where N}

"""
    AutoHalpernAccelerator <: AbstractADMMAccelerator

Automatic Halpern acceleration for ADMM with adaptive activation and period detection.

This accelerator implements a sophisticated variant of Halpern iteration that automatically
detects when to activate acceleration and adaptively manages the acceleration period.

**Mathematical Background**

The Halpern iteration scheme takes the form:
```math
x_{k+1} = \\frac{k}{k+1} T(x_k) + \\frac{1}{k+1} x_0
```

where:
- `T(x_k)` is the ADMM operator applied to the current iterate
- `x_0` is the anchor point (center point) for acceleration
- `k` is the iteration counter within the acceleration period

**Adaptive Features**

1. **Automatic Activation**: Detects when acceleration should be enabled
2. **Period Management**: Automatically restarts acceleration cycles
3. **Retrace Detection**: Monitors iterate behavior to detect oscillations
4. **Dual-Phase Operation**: Switches between exploration and acceleration phases

**Algorithm States**

- **Inactive**: Standard ADMM updates, monitoring for activation conditions
- **Semi-Surpassed**: Preparing for acceleration, validating conditions
- **Active**: Full Halpern acceleration with automatic period management

**Key Parameters**
- `maxPeriod::Int`: Maximum iterations in an acceleration cycle
- `iterDepth::Int`: Current position within acceleration cycle
- `isActive::Bool`: Whether acceleration is currently applied
- `isSemiSurpassed::Bool`: Whether preparing for activation

**Retrace Mechanism**

The accelerator tracks whether iterates are consistently increasing or decreasing.
When full retrace is detected (all components have changed direction), it triggers
acceleration restart with a new center point.

**Performance Characteristics**
- **Memory**: O(problem_size) for tracking iterate history
- **Computation**: O(problem_size) per iteration
- **Convergence**: Adaptive behavior can improve robustness over fixed schemes

**Example Usage**
```julia
accelerator = AutoHalpernAccelerator(maxPeriod=100)
initialize!(accelerator, info, admmGraph)

# Acceleration is applied automatically during ADMM iterations
accelerateAfterDualUpdates!(accelerator, info)
```
"""
mutable struct AutoHalpernAccelerator <: AbstractADMMAccelerator
    isActive::Bool 
    isSemiSurpassed::Bool 
    
    centerPoint::Dict{String, NumericVariable}
    lastPoint::Dict{String, NumericVariable}

    newIncrement::Dict{String, NumericVariableBool}
    lastIncrement::Dict{String, NumericVariableBool}
    accumulateRetrace::Dict{String, NumericVariableBool}

    hasLastIncrement::Bool 
    hasLastPoint::Bool 

    iterDepth::Int
    maxPeriod::Int
    
    converter::HalpernConverter 

    """
        AutoHalpernAccelerator(maxPeriod::Int=100)

    Construct an automatic Halpern accelerator with specified maximum period.

    **Arguments**
    - `maxPeriod::Int=100`: Maximum number of iterations in an acceleration cycle

    **Parameter Guidelines**
    - `maxPeriod`: Typical values 50-200. Higher values allow longer acceleration cycles
      but may delay adaptation to changing problem dynamics

    **Initialization**
    - All state variables are initialized to inactive/empty states
    - Converter is initialized but not configured until `initialize!` is called
    - Memory allocation is deferred until actual problem size is known

    **Example**
    ```julia
    accelerator = AutoHalpernAccelerator(maxPeriod=150)
    ```
    """
    AutoHalpernAccelerator(maxPeriod::Int=100) = new(
        false,
        false,
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariableBool}(),
        Dict{String, NumericVariableBool}(),
        Dict{String, NumericVariableBool}(),
        false,
        false,
        1,
        maxPeriod, 
        HalpernConverter())
end 

"""
    initialize!(accelerator::AutoHalpernAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the AutoHalpern accelerator with ADMM problem structure.

This function sets up the accelerator to work with the specific ADMM problem,
configuring variable selection and allocating necessary buffers.

**Variable Selection Strategy**
- **Primal Variables**: Uses right-side nodes from the bipartite graph
- **Dual Variables**: Uses all edges (constraints) in the graph
- **Rationale**: This selection typically captures the most important dynamics

**Initialization Process**
1. **Converter Setup**: Configure converter with selected variables
2. **Buffer Allocation**: Create buffers for center point, last point, and increment tracking
3. **State Reset**: Initialize all state variables to starting values

**Memory Allocation**
- Allocates buffers matching the size of selected variables
- Creates boolean arrays for increment and retrace tracking
- Memory usage scales with problem size

**Arguments**
- `accelerator::AutoHalpernAccelerator`: Accelerator instance to initialize
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of ADMM problem

**Post-Initialization State**
- Accelerator is ready for use but initially inactive
- All tracking arrays are initialized to false
- Iteration depth is set to 1
"""
function initialize!(accelerator::AutoHalpernAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    initialize!(accelerator.converter, admmGraph.right, collect(keys(admmGraph.edges)), info)

    for (id, buffer) in accelerator.converter.inputBuffer
        # initialize center and last point 
        accelerator.centerPoint[id] = similar(buffer)
        accelerator.lastPoint[id] = similar(buffer)

        # initialize increment and accumulate retrace 
        accelerator.newIncrement[id] = falses(size(buffer)...)
        accelerator.lastIncrement[id] = falses(size(buffer)...)
        accelerator.accumulateRetrace[id] = falses(size(buffer)...)
    end 
end 

"""
    accelerateBetweenPrimalUpdates!(accelerator::AutoHalpernAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Apply acceleration between primal updates (no-operation for AutoHalpern).

The AutoHalpern accelerator applies its acceleration after dual updates rather than
between primal updates. This function is provided for interface compliance.

**Arguments**
- `accelerator::AutoHalpernAccelerator`: Accelerator instance
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation

**Implementation**
This function performs no operations and returns immediately.
"""
function accelerateBetweenPrimalUpdates!(accelerator::AutoHalpernAccelerator, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    return 
end 

"""
    accelerateAfterDualUpdates!(accelerator::AutoHalpernAccelerator, info::ADMMIterationInfo)

Apply automatic Halpern acceleration after dual updates.

This is the main acceleration function that implements the complete AutoHalpern algorithm
including adaptive activation, period management, and retrace detection.

**Algorithm Phases**

1. **Iterate Retrieval**: Get current ADMM iterates into buffers
2. **Acceleration Application**: Apply Halpern formula (if active)
3. **History Tracking**: Update increment and retrace information
4. **State Management**: Check for activation/deactivation conditions

**Mathematical Implementation**

When active, applies the Halpern formula:
```math
x_{k+1} = \\frac{k}{k+1} T(x_k) + \\frac{1}{k+1} x_0
```

**Adaptive Logic**

The accelerator uses a sophisticated state machine:
- **Inactive → Semi-Surpassed**: When retrace conditions are met
- **Semi-Surpassed → Active**: When retrace conditions are met again
- **Active → Inactive**: When maximum period is reached or retrace detected

**Retrace Detection**

Tracks element-wise increment direction:
- `newIncrement[i] = (current[i] > last[i])`
- `accumulateRetrace[i] |= (newIncrement[i] ⊕ lastIncrement[i])`
- Full retrace when all elements have changed direction

**Performance Impact**
- **Memory**: O(problem_size) for tracking arrays
- **Computation**: O(problem_size) per iteration
- **Convergence**: Adaptive behavior can significantly improve convergence

**Arguments**
- `accelerator::AutoHalpernAccelerator`: Accelerator with adaptive state
- `info::ADMMIterationInfo`: Current ADMM iteration information to update
"""
function accelerateAfterDualUpdates!(accelerator::AutoHalpernAccelerator, info::ADMMIterationInfo)
    retrieveIterateInBuffer!(accelerator.converter, info)
    
    # 1. Acceleration
    if accelerator.isActive 
        coef1 = accelerator.iterDepth / (1.0 + accelerator.iterDepth)
        coef2 = 1.0 - coef1 
        for (id, buffer) in accelerator.converter.outputBuffer 
            copyto!(buffer, accelerator.converter.inputBuffer[id])
            axpby!(coef2, 
                accelerator.centerPoint[id], 
                coef1, 
                buffer)
        end 

    else 
        for (id, buffer) in accelerator.converter.outputBuffer
            copyto!(buffer, accelerator.converter.inputBuffer[id])
        end 
    end 
    accelerator.iterDepth += 1 
    
    # 2. Cache history information 
    if accelerator.hasLastPoint == false 
        for (id, buffer) in accelerator.converter.outputBuffer
            copyto!(accelerator.lastPoint[id], buffer)
            accelerator.accumulateRetrace[id] .= false 
        end 
        accelerator.hasLastPoint = true 
    elseif accelerator.hasLastIncrement == false 
        for (id, buffer) in accelerator.converter.outputBuffer
            accelerator.lastIncrement[id] .= buffer .> accelerator.lastPoint[id]
            copyto!(accelerator.lastPoint[id], buffer)
        end 
        accelerator.hasLastIncrement = true 
    else 
        for (id, buffer) in accelerator.converter.outputBuffer
            accelerator.newIncrement[id] .= buffer .> accelerator.lastPoint[id]
            accelerator.accumulateRetrace[id] .= accelerator.accumulateRetrace[id] .|| 
                (accelerator.newIncrement[id] .⊻ accelerator.lastIncrement[id])
            copyto!(accelerator.lastIncrement[id], accelerator.newIncrement[id])
            copyto!(accelerator.lastPoint[id], buffer)
        end 
    end 

    # 3. Switch active state and restart 
    isFullRetrace = all(all(value) for value in values(accelerator.accumulateRetrace))
    if isFullRetrace || (accelerator.iterDepth > accelerator.maxPeriod)
        if accelerator.isSemiSurpassed 
            accelerator.isActive = true 
            for (id, buffer) in accelerator.converter.outputBuffer
                copyto!(accelerator.centerPoint[id], buffer)
            end 
            accelerator.hasLastIncrement = false 
            for (id, accumulateRetrace) in accelerator.accumulateRetrace
                accumulateRetrace .= false 
            end 
            accelerator.isSemiSurpassed = false 
            accelerator.iterDepth = 1 
        else 
            for (id, accumulateRetrace) in accelerator.accumulateRetrace
                accumulateRetrace .= false 
            end 
            accelerator.isSemiSurpassed = true 
        end 
    end 

    applyAccelerationToADMM!(accelerator.converter, info)
end 

