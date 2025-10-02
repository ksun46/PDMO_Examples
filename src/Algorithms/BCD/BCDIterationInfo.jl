"""
    BCDTerminationStatus

Enumeration of possible termination statuses for the BCD algorithm.

- `BCD_TERMINATION_UNSPECIFIED`: Default status before termination
- `BCD_TERMINATION_OPTIMAL`: Converged to an optimal solution
- `BCD_TERMINATION_ITERATION_LIMIT`: Reached maximum iterations
- `BCD_TERMINATION_TIME_LIMIT`: Reached time limit
- `BCD_TERMINATION_NUMERICAL_ERROR`: Numerical issues encountered
- `BCD_TERMINATION_UNKNOWN`: Terminated with unknown status
"""
@enum BCDTerminationStatus begin 
    BCD_TERMINATION_UNSPECIFIED
    BCD_TERMINATION_OPTIMAL
    BCD_TERMINATION_ITERATION_LIMIT
    BCD_TERMINATION_TIME_LIMIT
    BCD_TERMINATION_UNKNOWN
end

"""
    BCDIterationInfo

Data structure to track the progress and results of BCD iterations.

# Fields
- `obj::Vector{Float64}`: Objective values
- `gradNorms::Vector{Float64}`: Gradient norms (if computed)
- `varChanges::Vector{Float64}`: Variable change norms
- `subproblemStatus::Vector{Bool}`: Success status of each subproblem solve
- `solution::Vector{NumericVariable}`: Current solution
- `solutionPrev::Vector{NumericVariable}`: Previous solution
- `stopIter::Int64`: Iteration at which the algorithm stopped
- `totalTime::Float64`: Total computation time
- `terminationStatus::BCDTerminationStatus`: Termination status
"""
mutable struct BCDIterationInfo
    # history info  
    obj::Vector{Float64}
    dresL2::Vector{Float64}
    dresLInf::Vector{Float64}
    
    # buffer for solution computations 
    solution::Vector{NumericVariable}
    solutionPrev::Vector{NumericVariable}
    blockDres::Vector{NumericVariable}

    # dres info 
    blockDresL2::Vector{Float64}
    blockDresLInf::Vector{Float64}
    
    # termination info
    stopIter::Int64
    totalTime::Float64
    terminationStatus::BCDTerminationStatus
    
    """
        BCDIterationInfo()

    Construct an empty BCDIterationInfo structure with default values.
    """
    BCDIterationInfo() = new(
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{NumericVariable}(),
        Vector{NumericVariable}(),
        Vector{NumericVariable}(),
        Vector{Float64}(),
        Vector{Float64}(),
        -1, 0.0,
        BCD_TERMINATION_UNSPECIFIED)
end

"""
    BCDIterationInfo(initialSolution::Vector{NumericVariable}) -> BCDIterationInfo

Construct and initialize a BCDIterationInfo structure for BCD algorithm execution.

This constructor creates a fully initialized iteration information structure that tracks
the state and progress of the BCD algorithm. It initializes the solution and buffer spaces.

# Arguments
- `initialSolution::Vector{NumericVariable}`: Initial solution blocks for the BCD algorithm

# Returns
- `BCDIterationInfo`: Fully initialized iteration info object ready for BCD execution

# Initialization Process
1. **Variable Initialization**:
   - Creates solution variables by copying each block from initial solution
   - Creates previous solution as copies of initial solution
   - Sets up buffer spaces for computations

2. **History Initialization**:
   - Initializes all history vectors as empty
   - Sets termination status to unspecified
   - Sets iteration counter to -1 (will be incremented to 0 on first iteration)

# Memory Allocation
- Allocates memory for all solution and buffer variables
- Creates buffer spaces matching variable dimensions
- Initializes all history tracking vectors
- Memory layout optimized for efficient computation

# Usage Context
This constructor is typically called at the beginning of the BCD algorithm:
```julia
info = BCDIterationInfo(initialSolution)
```

# Notes
- All solution blocks are deep copied to avoid aliasing
- Buffer spaces are pre-allocated for efficient computation
- The structure is ready for immediate use in BCD iterations
"""
function BCDIterationInfo(mbp::MultiblockProblem)
    info = BCDIterationInfo()

    # initialize solution buffers
    numberBlocks = length(mbp.blocks)
    for i in 1:numberBlocks
        x = mbp.blocks[i].val
        push!(info.solution, similar(x))
        push!(info.solutionPrev, similar(x))
        push!(info.blockDres, similar(x))

        proximalOracle!(info.solution[i], mbp.blocks[i].g, x, 1.0)
        copyto!(info.solutionPrev[i], info.solution[i])
    end

    # initialize block dres 
    info.blockDresL2 = zeros(numberBlocks)
    info.blockDresLInf = zeros(numberBlocks)

    # record objective value 
    recordBCDObjectiveValue(info, mbp)
    
    # initialize dres 
    push!(info.dresL2, Inf)
    push!(info.dresLInf, Inf)

    return info
end


function recordBCDObjectiveValue(info::BCDIterationInfo, mbp::MultiblockProblem)
    obj = mbp.couplingFunction(info.solution)
    for i in 1:length(mbp.blocks)
        obj += mbp.blocks[i].f(info.solution[i])
        obj += mbp.blocks[i].g(info.solution[i])
    end 

    push!(info.obj, obj)
end 

