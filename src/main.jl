# basic algorithmic components
include("Components/Functions/AbstractFunction.jl")
include("Components/Mappings/AbstractMapping.jl")


# formulations 
include("Formulations/BlockVariable.jl")
include("Formulations/BlockConstraint.jl")
include("Formulations/MultiblockProblem.jl")
include("Formulations/MultiblockProblemJuMP.jl")
include("Formulations/MultiblockProblemScaling.jl")
include("Formulations/MultiblockGraph.jl")
include("Formulations/BipartizationAlgorithms.jl")
include("Formulations/ADMMBipartiteGraph.jl")

# algorithmic components 
include("Algorithms/ADMM/BipartiteADMM.jl")
include("Algorithms/AdaPDM/AdaPDM.jl") 


"""
    runBipartiteADMM(mbp::MultiblockProblem, param::ADMMParam; kwargs...)

Solve a multiblock optimization problem using the bipartite ADMM algorithm.

This function implements a bipartite ADMM approach for solving large-scale multiblock optimization problems.
It includes automatic bipartization of the problem graph, scaling options, and comprehensive solution validation.

# Arguments
- `mbp::MultiblockProblem`: The multiblock optimization problem to solve
- `param::ADMMParam`: ADMM algorithm parameters including penalty parameter, tolerances, max iterations, 
  scaling options (`applyScaling`), and algorithmic components (solver, adapter, accelerator)

# Keyword Arguments
- `saveSolutionInMultiblockProblem::Bool = true`: Whether to save the solution back to the problem structure
- `bipartizationAlgorithm::BipartizationAlgorithm = BFS_BIPARTIZATION`: Algorithm for graph bipartization
  - `BFS_BIPARTIZATION`: Breadth-first search bipartization (default)
  - `DFS_BIPARTIZATION`: Depth-first search bipartization
  - `MILP_BIPARTIZATION`: Mixed-integer linear programming bipartization
  - `SPANNING_TREE_BIPARTIZATION`: Spanning tree bipartization
- `trueObj::Float64 = Inf`: Known true objective value for validation (if available)
- `tryJuMP::Bool = true`: Whether to verify solution using JuMP solver

# Returns
- `NamedTuple`: Contains solution and iteration information
  - `solution::Dict{BlockID, NumericVariable}`: Optimal solution for each block
  - `iterationInfo::ADMMIterationInfo`: Detailed iteration information including convergence history

# Examples
```julia
# Basic usage
mbp = MultiblockProblem()
# ... add blocks and constraints ...
param = ADMMParam()
param.initialRho = 1.0
param.maxIter = 1000
param.presTolL2 = 1e-6
param.dresTolL2 = 1e-6
result = runBipartiteADMM(mbp, param)

# With scaling and custom bipartization
param = ADMMParam()
param.applyScaling = true
param.solver = OriginalADMMSubproblemSolver()
result = runBipartiteADMM(mbp, param; 
    bipartizationAlgorithm=MILP_BIPARTIZATION)

# Access solution
solution = result.solution
info = result.iterationInfo
```

# Algorithm Steps
1. **Validation**: Checks multiblock problem validity
2. **Scaling**: Optionally applies problem scaling for numerical stability (controlled by `param.applyScaling`)
3. **Graph Construction**: Creates multiblock graph representation
4. **Bipartization**: Applies bipartization algorithm to create bipartite graph
5. **ADMM Execution**: Runs bipartite ADMM algorithm
6. **Solution Retrieval**: Extracts and unscales primal solution
7. **Verification**: Validates solution feasibility and objective value

See also: `runAdaPDM`, `MultiblockProblem`, `ADMMParam`
"""
function runBipartiteADMM(mbp::MultiblockProblem, 
    param::ADMMParam;
    saveSolutionInMultiblockProblem::Bool = true,  
    bipartizationAlgorithm::BipartizationAlgorithm = BFS_BIPARTIZATION,
    trueObj::Float64 = Inf,
    tryJuMP::Bool = true)

    @info "Run Bipartite ADMM with threads = $(Threads.nthreads())."
    
    # 1. sanity check for the problem 
    if checkMultiblockProblemValidity(mbp) == false 
        error("MultiblockProblem: The instance is not valid. ")
    end 
    summary(mbp)

    # 2. scale the problem 
    scalingInfo = param.applyScaling ? scaleMultiblockProblem!(mbp, scalingOptions=moderateScaling()) : nothing

    # 3. create a multblock graph instance from the problem
    graph = MultiblockGraph(mbp)
    summary(graph)

    # 4. generate ADMM bipartite graph 
    admmGraph = ADMMBipartiteGraph(graph, mbp, bipartizationAlgorithm)
    summary(admmGraph)

    # 5. execute ADMM
    info = BipartiteADMM(admmGraph, param)
    
    # 6. retrieve primal solution
    primalSolution = Dict{BlockID, NumericVariable}() 
    for block in mbp.blocks 
        admmNodeID = admmGraph.mbpBlockID2admmNodeID[block.id]
        primalSolution[block.id] = similar(info.primalSol[admmNodeID])
        copyto!(primalSolution[block.id], info.primalSol[admmNodeID])
    end 

    # unscale the problem and solution
    if scalingInfo != nothing && scalingInfo.isScaled
        unscaleMultiblockProblem!(mbp, primalSolution, scalingInfo)
    end 

    if saveSolutionInMultiblockProblem 
        for block in mbp.blocks 
            copyto!(block.val, primalSolution[block.id])
        end 
    end 

    # 6. summarize the ADMM result    
    if trueObj < Inf
        ADMMLog(info, trueObj)
    else 
        objByJuMP = Inf
        if tryJuMP
            try 
                objByJuMP = solveMultiblockProblemByJuMP(mbp)
            catch e
                @warn "MultiblockProblem: Failed to verify objective value using JuMP ($e)"
            end 
        end 
        ADMMLog(info, objByJuMP)
    end

    presL2, presLinf = checkMultiblockProblemFeasibility(mbp, primalSolution)
    msg = Printf.@sprintf("Infeasibility measures of original blocks: Pres (L2) = %.4e, Pres (LInf) = %.4e", presL2, presLinf)
    @info msg 
    
    return (solution=primalSolution, iterationInfo=info)
end 


"""
    runAdaPDM(mbp::MultiblockProblem, param::AbstractAdaPDMParam; kwargs...)

Solve a composite multiblock optimization problem using the Adaptive Primal-Dual Method (AdaPDM).

This function implements various adaptive primal-dual methods for solving composite optimization problems
with the structure: minimize f(x) + g(Ax), where some blocks may be proximal-only (g functions).

# Arguments
- `mbp::MultiblockProblem`: The composite multiblock optimization problem to solve
- `param::AbstractAdaPDMParam`: AdaPDM algorithm parameters. Can be one of:
  - `AdaPDMParam`: Standard adaptive primal-dual method parameters
  - `AdaPDMPlusParam`: Enhanced AdaPDM with additional features
  - `MalitskyPockParam`: Malitsky-Pock algorithm parameters
  - `CondatVuParam`: Condat-Vũ algorithm parameters

# Keyword Arguments
- `saveSolutionInMultiblockProblem::Bool = true`: Whether to save the solution back to the problem structure
- `trueObj::Float64 = Inf`: Known true objective value for validation (if available)
- `tryJuMP::Bool = true`: Whether to verify solution using JuMP solver

# Returns
- `NamedTuple`: Contains solution and iteration information
  - `solution::Dict{BlockID, NumericVariable}`: Optimal solution for each block
  - `iterationInfo::AdaPDMIterationInfo`: Detailed iteration information including convergence history

# Examples
```julia
# Basic AdaPDM usage
mbp = MultiblockProblem()
# ... add blocks with composite structure ...
param = AdaPDMParam(mbp; maxIter=1000, primalTol=1e-6, dualTol=1e-6)
result = runAdaPDM(mbp, param)

# Using Condat-Vũ algorithm
param = CondatVuParam(mbp; maxIter=1000)
result = runAdaPDM(mbp, param)

# Access solution
solution = result.solution
info = result.iterationInfo
```

# Problem Structure
The method is designed for composite problems where:
- Some blocks have smooth functions (f) with gradient oracles
- Some blocks have proximal functions (g) with proximal oracles
- The last block typically represents the coupling constraint variable

# Algorithm Variants
- **AdaPDM**: Standard adaptive primal-dual method
- **AdaPDM+**: Enhanced version with additional acceleration
- **Malitsky-Pock**: Malitsky-Pock primal-dual algorithm
- **Condat-Vũ**: Condat-Vũ primal-dual algorithm

See also: `runBipartiteADMM`, `MultiblockProblem`, `AdaPDMParam`
"""
function runAdaPDM(mbp::MultiblockProblem, param::AbstractAdaPDMParam; 
    saveSolutionInMultiblockProblem::Bool = true, 
    trueObj::Float64 = Inf,
    tryJuMP::Bool = true)
    @info "Run AdaPDM with threads = $(Threads.nthreads())."

    # 1. sanity check for the problem 
    if checkMultiblockProblemValidity(mbp) == false 
        error("runAdaPDM: The instance is not ready for adpative primal-dual method.")
    end 
    summary(mbp)
    
    # 2. run the algorithm 
    info = AdaptivePrimalDualMethod(mbp, param)

    # 3. summarize the resi;t
    if trueObj < Inf
        AdaPDMLog(info, trueObj)
    else 
        objByJuMP = Inf
        if tryJuMP
            try 
                objByJuMP = solveMultiblockProblemByJuMP(mbp)
            catch 
                @info "MultiblockProblem: Failed to solve the whole problem using JuMP. "
            end 
        end 
        AdaPDMLog(info, objByJuMP)
    end
    
    # 4. save the solution 
    primalSolution = Dict{BlockID, NumericVariable}()
    for block in mbp.blocks[1:end-1]
        primalSolution[block.id] = info.primalSol[block.id]
        if saveSolutionInMultiblockProblem
            copyto!(block.val, info.primalSol[block.id])
        end
    end
    primalSolution[mbp.blocks[end].id] = info.bufferAx
    if saveSolutionInMultiblockProblem
        copyto!(mbp.blocks[end].val, info.bufferAx) 
    end
 
    return (solution=primalSolution, iterationInfo=info)
end 
