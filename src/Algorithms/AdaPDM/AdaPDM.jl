include("AbstractAdaPDMParam/AbstractAdaPDMParam.jl")
include("AdaPDMIterationInfo/AdaPDMIterationInfo.jl")
include("AdaPDMTerminationCriteria.jl")
include("AdaPDMUtil.jl")

"""
    AdaptivePrimalDualMethod(mbp::MultiblockProblem, param::AbstractAdaPDMParam)

Internal implementation of the Adaptive Primal-Dual Method for solving composite optimization problems.

This function implements the core algorithm for various adaptive primal-dual methods. It is the internal
implementation called by the public `runAdaPDM` function. The method solves composite problems of the form:

    minimize f₁(x₁) + g₁(x₁) + ... + fₙ(xₙ) + gₙ(xₙ) + h(A₁x₁ + ... + Aₙxₙ)

where fᵢ are smooth convex functions and gᵢ are proximable convex functions.

# Mathematical Formulation
The method assumes the problem is formulated as a multiblock problem:

    minimize f₁(x₁) + g₁(x₁) + ... + fₙ(xₙ) + gₙ(xₙ) + g_{n+1}(x_{n+1})
    subject to A₁x₁ + ... + Aₙxₙ - x_{n+1} = 0

where:
- f₁, ..., fₙ are smooth convex functions with gradient oracles
- g₁, ..., gₙ are proximable convex functions with proximal oracles  
- h = g_{n+1} is a proximable convex function representing the coupling constraint

# Arguments
- `mbp::MultiblockProblem`: The composite multiblock optimization problem
- `param::AbstractAdaPDMParam`: Algorithm parameters, can be one of:
  - `AdaPDMParam`: Standard adaptive primal-dual method
  - `AdaPDMPlusParam`: Enhanced AdaPDM with line search
  - `MalitskyPockParam`: Malitsky-Pock algorithm with backtracking
  - `CondatVuParam`: Condat-Vũ algorithm with fixed step sizes

# Returns
- `AdaPDMIterationInfo`: Complete iteration information including:
  - Primal and dual solutions
  - Convergence history (residuals and objective values)
  - Timing information
  - Termination status

# Algorithm Steps
1. **Validation**: Checks if the problem has composite structure
2. **Initialization**: Sets up iteration info and termination criteria
3. **Iteration Loop**: For each iteration:
   - Update dual solution using algorithm-specific rules
   - Update primal solutions using proximal operators
   - Compute residuals and objective values
   - Check termination criteria
   - Log progress information
4. **Termination**: Returns complete iteration information

# Algorithm Variants
The function dispatches to different update rules based on the parameter type:
- **AdaPDM**: Uses adaptive step sizes based on problem geometry
- **AdaPDM+**: Adds line search for enhanced operator norm estimation
- **Malitsky-Pock**: Uses backtracking line search for automatic step size selection
- **Condat-Vũ**: Uses fixed step sizes with convergence guarantees

# Threading
The algorithm uses Julia's threading capabilities for parallel processing
of independent block updates where possible.

# Error Conditions
- Throws an error if the problem is not a valid composite problem
- Individual algorithms may have specific requirements (e.g., smooth functions)

# Notes
This is an internal function. Users should call `runAdaPDM` instead, which provides
additional features like solution validation and result formatting.

See also: `runAdaPDM`, `AdaPDMIterationInfo`, `AdaPDMTerminationCriteria`
"""
function AdaptivePrimalDualMethod(mbp::MultiblockProblem, param::AbstractAdaPDMParam) 
    startTime = time() 
    nThreads = Threads.nthreads() 

    @PDMOInfo param.logLevel "#"^40 * " Adaptive Primal-dual Method " * "#"^40
    @PDMOInfo param.logLevel "Method = $(getAdaPDMName(param))"
    if checkCompositeProblemValidity!(mbp) == false 
        @PDMOError param.logLevel "AdaptiveProximalGradientMethod: the input problem is not a valid composite problem."
        return
    end 

    # Initialize iteration info and termination criteria 
    info = AdaPDMIterationInfo(mbp, param)
    terminationCriteria = AdaPDMTerminationCriteria(param)

    msg = Printf.@sprintf("AdaPDM: initialization took %.2f seconds \n", time() - startTime)
    @PDMOInfo param.logLevel msg 

    startTime = time()
    AdaPDMLog(0, info, param)

    # start the iteration 
    for iter in 1:param.maxIter 
        updateDualSolution!(mbp, info, param)
        updatePrimalSolution!(mbp, info, param)
        computePDMResidualsAndObjective!(info, mbp, param)

        # log iteration info 
        info.totalTime = time() - startTime 
        iterLogged = AdaPDMLog(iter, info, param)

        # check termination criteria 
        checkTerminationCriteria(info, terminationCriteria)
        if terminationCriteria.terminated 
            if iterLogged == false 
                AdaPDMLog(iter, info, param; final = true)
            end 
            break 
        end 
    end 

    return info 
end
