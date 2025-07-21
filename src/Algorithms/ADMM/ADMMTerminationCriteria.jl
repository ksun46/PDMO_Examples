"""
    ADMMTerminationCriteria.jl

Comprehensive termination criteria system for ADMM algorithms.

This module provides a sophisticated termination framework for ADMM optimization algorithms
that goes beyond simple residual checking. It implements multiple levels of termination
criteria including basic convergence tests, advanced problem classification, and
sophisticated infeasibility/unboundedness detection.

# Termination Levels

## Level 1: Basic Termination Criteria
- **Optimality**: Primal and dual residuals satisfy tolerances
- **Iteration limit**: Maximum iterations reached
- **Time limit**: Wall-clock time limit exceeded
- **Numerical errors**: NaN or Inf values detected

## Level 2: Advanced Classification (with Metrics)
- **Infeasibility detection**: Problem has no feasible solution
- **Unboundedness detection**: Objective function is unbounded below
- **Ill-posed problem detection**: Problem is unsuitable for ADMM

# Problem Classification System

The module implements a sophisticated classification system that categorizes
optimization problems based on convergence behavior patterns:

- **Case A**: Well-posed problem with optimal solution and ADMM convergence
- **Case B**: Has optimal solution but ADMM may not be applicable
- **Case C**: Lower bounded but no optimal solution exists
- **Case D**: Lower bounded but ADMM may not be applicable
- **Case E**: Lower unbounded (objective decreases without bound)
- **Case F**: Infeasible (constraints are inconsistent)

# Mathematical Foundation

The classification is based on sophisticated convergence analysis including:
- Logarithmic scaling tests for sequence convergence
- Ratio tests for boundedness detection
- Series convergence analysis using mathematical convergence criteria
- Statistical pattern recognition over extended iteration sequences

# Usage Patterns

## Basic Usage (All Solvers)
```julia
criteria = ADMMTerminationCriteria(param, info)
checkTerminationCriteria(info, criteria)
```

## Advanced Usage (OriginalADMMSubproblemSolver)
```julia
criteria = ADMMTerminationCriteria(param, info)  # Includes metrics
collectTerminationMetricsBetweenPrimalUpdates!(criteria, info, graph)
collectTerminationMetricsAfterDualUpdates!(criteria, info, graph)
checkTerminationCriteria(info, criteria)
```

# Performance Considerations

- Basic termination criteria have minimal computational overhead
- Advanced metrics collection is only activated for OriginalADMMSubproblemSolver
- Sophisticated analysis requires 1000+ iterations for robust classification
- Memory-efficient implementation with pre-allocated buffers

# Integration with ADMM Algorithm

The termination system is tightly integrated with the ADMM iteration loop:
1. Criteria are initialized based on solver type
2. Metrics are collected at appropriate points in the iteration
3. Comprehensive termination checking is performed each iteration
4. Early termination prevents unnecessary computation

# Notes

- The advanced classification system is based on mathematical convergence theory
- Problem classification helps users understand why ADMM may not converge
- The system is designed to be robust against numerical issues
- Different termination reasons provide valuable debugging information
"""

"""
    ADMM termination module containing termination criteria and functions to check various termination conditions
"""

"""
    Termination criteria for ADMM algorithm
    
Contains tolerances and limits for termination conditions, as well as flags
to control termination behavior.
"""
mutable struct ADMMTerminationCriteria
    """
    Absolute tolerance for primal residual L2 norm
    """
    presTolL2::Float64

    """
    Absolute tolerance for primal residual L-infinity norm
    """
    presTolLInf::Float64

    """
    Absolute tolerance for dual residual L2 norm
    """
    dresTolL2::Float64

    """
    Absolute tolerance for dual residual L-infinity norm
    """
    dresTolLInf::Float64

    """
    Maximum number of iterations
    """
    iterLimit::Int64

    """
    Time limit in seconds
    """
    timeLimit::Float64
    
    """
    Flag indicating whether any termination condition has been met
    """
    terminated::Bool 

    """
    Metrics for termination criteria
    """
    metrics::Union{ADMMTerminationMetrics, Nothing}
    
    function ADMMTerminationCriteria(;
        presTolL2::Float64 = 1e-4,
        presTolLInf::Float64 = 1e-4,
        dresTolL2::Float64 = 1e-4,
        dresTolLInf::Float64 = 1e-4,
        iterLimit::Int64 = 1000,
        timeLimit::Float64 = 3600.0, 
        terminated::Bool = false)
        return new(presTolL2, 
            presTolLInf, 
            dresTolL2, 
            dresTolLInf, 
            iterLimit, 
            timeLimit, 
            terminated, 
            nothing
        )
    end
end

"""
    ADMMTerminationCriteria(param::ADMMParam)
    
Construct termination criteria from ADMM parameters, validating the values.
"""
function ADMMTerminationCriteria(param::ADMMParam, info::ADMMIterationInfo)
    if param.presTolL2 < 0
        error("ADMMTerminationCriteria: presTolL2 must be nonnegative")
    end
    if param.presTolLInf < 0
        error("ADMMTerminationCriteria: presTolLInf must be nonnegative")
    end
    if param.dresTolL2 < 0
        error("ADMMTerminationCriteria: dresTolL2 must be nonnegative")
    end     
    if param.dresTolLInf < 0
        error("ADMMTerminationCriteria: dresTolLInf must be nonnegative")
    end
    if param.maxIter <= 0
        error("ADMMTerminationCriteria: iterLimit must be positive")
    end 
    if param.timeLimit <= 0
        error("ADMMTerminationCriteria: timeLimit must be positive")
    end

    criteria = ADMMTerminationCriteria(
        presTolL2 = param.presTolL2,
        presTolLInf = param.presTolLInf,        
        dresTolL2 = param.dresTolL2,
        dresTolLInf = param.dresTolLInf,
        iterLimit = param.maxIter,
        timeLimit = param.timeLimit,
        terminated = false)

    # initialize metrics only for original ADMM solver 
    if isa(param.solver, OriginalADMMSubproblemSolver) && param.enablePathologyCheck
        criteria.metrics = ADMMTerminationMetrics(info)
    end 

    return criteria 
end


"""
    checkOptimalTermination(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) -> Bool

Check if the algorithm should terminate because optimality criteria have been met.
Updates the termination status in info and sets criteria.terminated = true if successful.
Returns true if optimality criteria are met, false otherwise.
"""
function checkOptimalTermination(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)
    if info.presL2[end] <= criteria.presTolL2 &&
        info.presLInf[end] <= criteria.presTolLInf &&
        info.dresL2[end] <= criteria.dresTolL2 &&
        info.dresLInf[end] <= criteria.dresTolLInf
        info.terminationStatus = ADMM_TERMINATION_OPTIMAL
        criteria.terminated = true
        return true 
    end
    return false
end

"""
    checkIterationLimit(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) -> Bool

Check if the algorithm has reached the iteration limit.
Updates the termination status in info and sets criteria.terminated = true if limit reached.
Returns true if iteration limit reached, false otherwise.
"""
function checkIterationLimit(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)
    iter = length(info.presL2) - 1
    if iter >= criteria.iterLimit
        info.terminationStatus = ADMM_TERMINATION_ITERATION_LIMIT
        criteria.terminated = true
        return true
    end
    return false
end

"""
    checkTimeLimit(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) -> Bool

Check if the algorithm has reached the time limit.
Updates the termination status in info and sets criteria.terminated = true if limit reached.
Returns true if time limit reached, false otherwise.
"""
function checkTimeLimit(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)
    if info.totalTime >= criteria.timeLimit
        info.terminationStatus = ADMM_TERMINATION_TIME_LIMIT
        criteria.terminated = true
        return true
    end
    return false
end

"""
    checkNumericalError(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) -> Bool

Check for numerical errors (NaN or Inf values) in the residuals.
Updates the termination status in info and sets criteria.terminated = true if errors found.
Returns true if numerical errors are detected, false otherwise.
"""
function checkNumericalError(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)
    # Check for NaN or Inf in residuals
    if isnan(info.presL2[end]) || isinf(info.presL2[end]) ||
       isnan(info.dresL2[end]) || isinf(info.dresL2[end]) ||
       isnan(info.presLInf[end]) || isinf(info.presLInf[end]) ||
       isnan(info.dresLInf[end]) || isinf(info.dresLInf[end]) ||
       isnan(info.obj[end]) || isinf(info.obj[end])
        
        info.terminationStatus = ADMM_TERMINATION_UNKNOWN
        criteria.terminated = true
        return true
    end
    return false
end

"""
    checkInfeasibility(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) -> Bool

Stub function for infeasibility checking without metrics.

This version always returns false as it doesn't have access to the advanced 
metrics needed for infeasibility detection. The actual infeasibility detection
is performed by the 3-argument version that takes metrics as input.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object

# Returns
- `Bool`: Always returns false (no infeasibility detection without metrics)
"""
function checkInfeasibility(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)
    return false 
end 

"""
    checkUnboundedness(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) -> Bool

Stub function for unboundedness checking without metrics.

This version always returns false as it doesn't have access to the advanced 
metrics needed for unboundedness detection. The actual unboundedness detection
is performed by the 3-argument version that takes metrics as input.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object

# Returns
- `Bool`: Always returns false (no unboundedness detection without metrics)
"""
function checkUnboundedness(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)
    return false 
end 

"""
    collectTerminationMetricsBetweenPrimalUpdates!(criteria::ADMMTerminationCriteria, info::ADMMIterationInfo, param::ADMMParam)

Collect termination metrics between primal updates for the ADMM algorithm.

This function delegates to the corresponding metrics collection function if metrics are enabled. 
The metrics collected at this point include primal residuals and intermediate dual variable values 
that are needed for advanced termination criteria and algorithm analysis.

# Arguments
- `criteria::ADMMTerminationCriteria`: The termination criteria object containing metrics
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `param::ADMMParam`: ADMM algorithm parameters

# Notes
- This function is called after primal variable updates but before dual variable updates
- Only collects metrics if `criteria.metrics` is not `nothing`
- Used primarily for the original ADMM subproblem solver
"""
function collectTerminationMetricsBetweenPrimalUpdates!(criteria::ADMMTerminationCriteria, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    if criteria.metrics != nothing 
        collectTerminationMetricsBetweenPrimalUpdates!(criteria.metrics, info, admmGraph)
    end 
end 

"""
    collectTerminationMetricsAfterDualUpdates!(criteria::ADMMTerminationCriteria, info::ADMMIterationInfo, param::ADMMParam)

Collect termination metrics after dual updates for the ADMM algorithm.

This function delegates to the corresponding metrics collection function if metrics are enabled.
The metrics collected at this point include dual norms, Bz norms, dual objective values, and
dual variable differences that are used for advanced termination criteria and convergence analysis.

# Arguments
- `criteria::ADMMTerminationCriteria`: The termination criteria object containing metrics
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `param::ADMMParam`: ADMM algorithm parameters

# Notes
- This function is called after dual variable updates
- Only collects metrics if `criteria.metrics` is not `nothing`
- Used primarily for the original ADMM subproblem solver
- Metrics collected include dual norms, Bz norms, and dual objective values
"""
function collectTerminationMetricsAfterDualUpdates!(criteria::ADMMTerminationCriteria, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    if criteria.metrics != nothing 
        collectTerminationMetricsAfterDualUpdates!(criteria.metrics, info, admmGraph)
    end 
end 
 
"""
    checkInfeasibility(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics) -> Bool

Check for infeasibility in the optimization problem using advanced metrics analysis.

This function detects if the optimization problem is infeasible (Case F) based on the 
convergence behavior analysis performed by the termination metrics. It checks if the
problem has been classified as Case F, which indicates that the problem constraints
are inconsistent and no feasible solution exists.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object
- `metrics::ADMMTerminationMetrics`: Metrics object containing problem classification

# Returns
- `Bool`: `true` if infeasibility is detected, `false` otherwise

# Side Effects
- Sets `info.terminationStatus = ADMM_TERMINATION_INFEASIBLE` if infeasible
- Sets `criteria.terminated = true` if infeasible

# Notes
- Only triggers if `info.stopIter < 0` (algorithm hasn't terminated yet)
- Based on problem classification from convergence behavior analysis
"""
function checkInfeasibility(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics)
    if info.stopIter < 0 && metrics.problem_classification == CLASSIFICATION_CASE_F        
        info.terminationStatus = ADMM_TERMINATION_INFEASIBLE
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkUnboundedness(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics) -> Bool

Check for unboundedness in the optimization problem using advanced metrics analysis.

This function detects if the optimization problem is unbounded below (Case E) based on the 
convergence behavior analysis performed by the termination metrics. It checks if the
problem has been classified as Case E, which indicates that the objective function
can decrease without bound.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object
- `metrics::ADMMTerminationMetrics`: Metrics object containing problem classification

# Returns
- `Bool`: `true` if unboundedness is detected, `false` otherwise

# Side Effects
- Sets `info.terminationStatus = ADMM_TERMINATION_UNBOUNDED` if unbounded
- Sets `criteria.terminated = true` if unbounded

# Notes
- Only triggers if `info.stopIter < 0` (algorithm hasn't terminated yet)
- Based on problem classification from convergence behavior analysis
- Specifically detects lower unboundedness (Case E)
"""
function checkUnboundedness(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics)
    if info.stopIter < 0 && metrics.problem_classification == CLASSIFICATION_CASE_E
        # println("The optimization problem belongs to Case (e)") 
        # println("The optimization problem is lower unbounded.")

        info.terminationStatus = ADMM_TERMINATION_UNBOUNDED
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkIllposedCaseD(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics) -> Bool

Check for ill-posed optimization problem Case D using advanced metrics analysis.

This function detects if the optimization problem belongs to Case D, which indicates
that the problem is lower bounded but ADMM may not be applicable due to the specific
convergence behavior patterns observed in the iteration sequences.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object
- `metrics::ADMMTerminationMetrics`: Metrics object containing problem classification

# Returns
- `Bool`: `true` if Case D is detected, `false` otherwise

# Side Effects
- Sets `info.terminationStatus = ADMM_TERMINATION_ILLPOSED_CASE_D` if detected
- Sets `criteria.terminated = true` if detected

# Notes
- Only triggers if `info.stopIter < 0` (algorithm hasn't terminated yet)
- Case D indicates the problem is lower bounded but may not be suitable for ADMM
- Based on advanced convergence behavior analysis from termination metrics
"""
function checkIllposedCaseD(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics)
    if info.stopIter < 0 && metrics.problem_classification == CLASSIFICATION_CASE_D
        # println("The optimization problem belongs to Case (d)") 
        # println("The optimization problem is lower bounded, but ADMM may not applicable for this optimization problem.")

        info.terminationStatus = ADMM_TERMINATION_ILLPOSED_CASE_D
        criteria.terminated = true
        return true
    end
    return false
end 


"""
    checkIllposedCaseC(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics) -> Bool

Check for ill-posed optimization problem Case C using advanced metrics analysis.

This function detects if the optimization problem belongs to Case C, which indicates
that the problem is lower bounded but does not have an optimal solution. This suggests
that ADMM may not be applicable for this type of optimization problem due to the
lack of a well-defined optimum.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object
- `metrics::ADMMTerminationMetrics`: Metrics object containing problem classification

# Returns
- `Bool`: `true` if Case C is detected, `false` otherwise

# Side Effects
- Sets `info.terminationStatus = ADMM_TERMINATION_ILLPOSED_CASE_C` if detected
- Sets `criteria.terminated = true` if detected

# Notes
- Only triggers if `info.stopIter < 0` (algorithm hasn't terminated yet)
- Case C indicates the problem is lower bounded but lacks an optimal solution
- ADMM may not be applicable for this problem class
- Based on advanced convergence behavior analysis from termination metrics
"""
function checkIllposedCaseC(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics)
    if info.stopIter < 0 && metrics.problem_classification == CLASSIFICATION_CASE_C
        # println("The optimization problem belongs to Case (c)")
        # println("The optimization problem is lower bounded, but does not have an optimal solution. ADMM may not applicable for this optimization problem.")
        
        info.terminationStatus = ADMM_TERMINATION_ILLPOSED_CASE_C
        criteria.terminated = true
        return true
    end
    return false
end


"""
    checkIllposedCaseB(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics) -> Bool

Check for ill-posed optimization problem Case B using advanced metrics analysis.

This function detects if the optimization problem belongs to Case B, which indicates
that the problem has an optimal solution but ADMM may not be applicable due to
specific convergence behavior patterns. Despite having a well-defined optimum,
the algorithm's iteration sequences suggest that ADMM may not converge properly.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information
- `criteria::ADMMTerminationCriteria`: Termination criteria object
- `metrics::ADMMTerminationMetrics`: Metrics object containing problem classification

# Returns
- `Bool`: `true` if Case B is detected, `false` otherwise

# Side Effects
- Sets `info.terminationStatus = ADMM_TERMINATION_ILLPOSED_CASE_B` if detected
- Sets `criteria.terminated = true` if detected

# Notes
- Only triggers if `info.stopIter < 0` (algorithm hasn't terminated yet)
- Case B indicates the problem has an optimal solution but ADMM may not be suitable
- Different from infeasibility - the problem is well-posed but not ADMM-friendly
- Based on advanced convergence behavior analysis from termination metrics
"""
function checkIllposedCaseB(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria, metrics::ADMMTerminationMetrics)
    if info.stopIter < 0 && metrics.problem_classification == CLASSIFICATION_CASE_B
        # println("The optimization problem belongs to Case (b)")
        # println("The optimization has an optimal solution, but ADMM may not applicable for this optimization problem.")
        
        info.terminationStatus = ADMM_TERMINATION_ILLPOSED_CASE_B
        criteria.terminated = true
        return true
    end
    return false
end


"""
    checkTerminationCriteria(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria)

Check all termination criteria and update the termination status in the criteria object.

This function performs a comprehensive check of all possible termination conditions including:
- Standard optimality conditions (primal/dual residual tolerances)
- Iteration and time limits  
- Numerical errors (NaN/Inf detection)
- Advanced problem classification (infeasibility, unboundedness, ill-posed cases)

The function first updates termination metrics if available, then checks termination
conditions in a specific order. If metrics are available, it also checks for advanced
problem classifications (Cases B, C, D, E, F) that indicate various types of 
problematic optimization problems.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information including residuals and timing
- `criteria::ADMMTerminationCriteria`: Termination criteria object containing tolerances and metrics

# Side Effects
- Updates `criteria.terminated` to `true` if any termination condition is met
- Updates `info.terminationStatus` with the specific termination reason
- Updates `info.stopIter` with the iteration number when termination occurs
- Calls `updateTerminationMetrics!` if metrics are available

# Termination Conditions Checked
1. **Optimal termination**: All residuals satisfy tolerances
2. **Iteration limit**: Maximum iterations reached
3. **Time limit**: Maximum time exceeded
4. **Numerical errors**: NaN or Inf values detected
5. **Basic infeasibility/unboundedness**: Stub checks (always false)
6. **Advanced classification** (if metrics available):
   - Case F: Infeasible problem
   - Case E: Lower unbounded problem  
   - Case D: Lower bounded but ADMM may not be applicable
   - Case C: Lower bounded but no optimal solution
   - Case B: Has optimal solution but ADMM may not be suitable

# Notes
- This function does not return anything - check `criteria.terminated` to determine if stopping
- Advanced classification requires `criteria.metrics` to be non-null
- Classification is based on sophisticated convergence behavior analysis
"""
function checkTerminationCriteria(info::ADMMIterationInfo, criteria::ADMMTerminationCriteria) 

    if criteria.metrics != nothing 
        updateTerminationMetrics!(criteria.metrics, info, criteria.presTolL2, criteria.dresTolL2)
    end 

    # Check stopping criteria 
    while true
        if checkNumericalError(info, criteria)
            break 
        end 
         
        if checkOptimalTermination(info, criteria)
            break 
        end 

        if checkIterationLimit(info, criteria)
            break 
        end 

        if checkTimeLimit(info, criteria)
            break 
        end 

        if checkInfeasibility(info, criteria)
            break 
        end 

        if checkUnboundedness(info, criteria)
            break 
        end 

        if criteria.metrics != nothing 

            if checkInfeasibility(info, criteria, criteria.metrics)
                break 
            end 

            if checkUnboundedness(info, criteria, criteria.metrics)
                break 
            end 

            if checkIllposedCaseB(info, criteria, criteria.metrics)
                break 
            end 

            if checkIllposedCaseC(info, criteria, criteria.metrics)
                break 
            end     

            if checkIllposedCaseD(info, criteria, criteria.metrics)
                break 
            end 
        end 

        break
    end 

    if criteria.terminated 
        info.stopIter = length(info.presL2) - 1
    end 
end

"""
    getTerminationStatus(status::ADMMTerminationStatus) -> String
    
Convert an ADMMTerminationStatus enum value to a human-readable string description.
"""
function getTerminationStatus(status::ADMMTerminationStatus)
    if status == ADMM_TERMINATION_OPTIMAL
        return "ADMM_TERMINATION_OPTIMAL"
    elseif status == ADMM_TERMINATION_ITERATION_LIMIT
        return "ADMM_TERMINATION_ITERATION_LIMIT"
    elseif status == ADMM_TERMINATION_TIME_LIMIT
        return "ADMM_TERMINATION_TIME_LIMIT"
    elseif status == ADMM_TERMINATION_INFEASIBLE
        return "ADMM_TERMINATION_INFEASIBLE"
    elseif status == ADMM_TERMINATION_UNBOUNDED
        return "ADMM_TERMINATION_UNBOUNDED"
    elseif status == ADMM_TERMINATION_UNKNOWN
        return "ADMM_TERMINATION_UNKNOWN"
    elseif status == ADMM_TERMINATION_ILLPOSED_CASE_D
        return "ADMM_TERMINATION_ILLPOSED_CASE_D"
    elseif status == ADMM_TERMINATION_ILLPOSED_CASE_C
        return "ADMM_TERMINATION_ILLPOSED_CASE_C"
    elseif status == ADMM_TERMINATION_ILLPOSED_CASE_B
        return "ADMM_TERMINATION_ILLPOSED_CASE_B"
    elseif status == ADMM_TERMINATION_UNKNOWN
        return "ADMM_TERMINATION_UNKNOWN"
    else
        return "ADMM_TERMINATION_UNSPECIFIED"
    end
end