"""
    AdaPDMTerminationCriteria

Structure containing termination criteria and status for the Adaptive Primal-Dual Method.

This structure defines the convergence tolerances and limits used to determine when
the AdaPDM algorithm should terminate. It tracks both the criteria values and the
current termination status.

# Fields
- `presTolL2::Float64`: Primal residual tolerance in L2 norm
- `presTolLInf::Float64`: Primal residual tolerance in L∞ norm  
- `dresTolL2::Float64`: Dual residual tolerance in L2 norm
- `dresTolLInf::Float64`: Dual residual tolerance in L∞ norm
- `maxIter::Int64`: Maximum number of iterations allowed
- `timeLimit::Float64`: Maximum execution time in seconds
- `terminated::Bool`: Whether the algorithm has terminated

# Termination Conditions
The algorithm terminates when ANY of the following conditions is met:
1. **Optimal**: All residuals fall below their respective tolerances
2. **Iteration Limit**: Maximum number of iterations reached
3. **Time Limit**: Maximum execution time exceeded
4. **Numerical Error**: NaN or Inf detected in residuals
5. **Unbounded**: Problem is detected as unbounded (placeholder)

# Constructor
```julia
AdaPDMTerminationCriteria(;
    presTolL2::Float64 = 1e-4,
    presTolLInf::Float64 = 1e-4,
    dresTolL2::Float64 = 1e-4,
    dresTolLInf::Float64 = 1e-4,
    maxIter::Int64 = 1000,
    timeLimit::Float64 = 3600.0,
    terminated::Bool = false)
```

See also: `AbstractAdaPDMParam`, `AdaPDMIterationInfo`, `checkTerminationCriteria`
"""
mutable struct AdaPDMTerminationCriteria
    presTolL2::Float64
    presTolLInf::Float64
    dresTolL2::Float64
    dresTolLInf::Float64
    maxIter::Int64
    timeLimit::Float64
    terminated::Bool 

    function AdaPDMTerminationCriteria(;
        presTolL2::Float64 = 1e-4,
        presTolLInf::Float64 = 1e-4,
        dresTolL2::Float64 = 1e-4,
        dresTolLInf::Float64 = 1e-4,
        maxIter::Int64 = 1000,
        timeLimit::Float64 = 3600.0,
        terminated::Bool = false)
        
        return new(presTolL2, presTolLInf, dresTolL2, dresTolLInf, maxIter, timeLimit, terminated)
    end
end

"""
    AdaPDMTerminationCriteria(param::AbstractAdaPDMParam)

Create termination criteria from AdaPDM algorithm parameters.

This constructor extracts the termination criteria from an AdaPDM parameter object
and validates that all values are within acceptable ranges.

# Arguments
- `param::AbstractAdaPDMParam`: Algorithm parameters containing termination criteria

# Returns
- `AdaPDMTerminationCriteria`: Initialized termination criteria object

# Validation
The constructor validates that:
- All tolerance values are non-negative
- Maximum iterations is positive
- Time limit is positive

# Throws
- `ErrorException`: If any validation fails

# Examples
```julia
# Create from AdaPDM parameters
param = AdaPDMParam(mbp; presTolL2=1e-6, maxIter=5000)
criteria = AdaPDMTerminationCriteria(param)
```

See also: `AdaPDMParam`, `AdaPDMPlusParam`, `MalitskyPockParam`, `CondatVuParam`
"""
function AdaPDMTerminationCriteria(param::AbstractAdaPDMParam)
    if param.presTolL2 < 0
        error("AdaPDMTermination: presTolL2 must be nonnegative")
    end
    if param.presTolLInf < 0
        error("AdaPDMTermination: presTolLInf must be nonnegative")
    end     
    if param.dresTolL2 < 0
        error("AdaPDMTermination: dresTolL2 must be nonnegative")
    end
    if param.dresTolLInf < 0
        error("AdaPDMTermination: dresTolLInf must be nonnegative")
    end
    if param.maxIter <= 0
        error("AdaPDMTermination: maxIter must be positive")
    end
    if param.timeLimit <= 0
        error("AdaPDMTermination: timeLimit must be positive")
    end
    
    return AdaPDMTerminationCriteria(
        presTolL2 = param.presTolL2,
        presTolLInf = param.presTolLInf,
        dresTolL2 = param.dresTolL2,
        dresTolLInf = param.dresTolLInf,
        maxIter = param.maxIter,
        timeLimit = param.timeLimit,
        terminated = false)
end 

"""
    checkOptimalTermination(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)

Check if the algorithm has reached optimal termination based on residual tolerances.

This function determines if all primal and dual residuals (in both L2 and L∞ norms)
have fallen below their respective tolerance thresholds, indicating convergence.

# Arguments
- `info::AdaPDMIterationInfo`: Current iteration information containing residuals
- `criteria::AdaPDMTerminationCriteria`: Termination criteria containing tolerances

# Returns
- `Bool`: `true` if optimal termination achieved, `false` otherwise

# Side Effects
If termination is achieved:
- Sets `info.terminationStatus` to `ADA_PDM_TERMINATION_OPTIMAL`
- Sets `criteria.terminated` to `true`

# Termination Condition
Returns `true` when ALL of the following hold:
- `presL2 ≤ presTolL2`
- `presLInf ≤ presTolLInf`  
- `dresL2 ≤ dresTolL2`
- `dresLInf ≤ dresTolLInf`

See also: `checkTerminationCriteria`, `AdaPDMTerminationStatus`
"""
function checkOptimalTermination(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)
    if info.presL2[end] <= criteria.presTolL2 && 
        info.presLInf[end] <= criteria.presTolLInf &&
        info.dresL2[end] <= criteria.dresTolL2 && 
        info.dresLInf[end] <= criteria.dresTolLInf
        info.terminationStatus = ADA_PDM_TERMINATION_OPTIMAL
        criteria.terminated = true
        return true
    end
    return false
end

"""
    checkIterationLimit(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)

Check if the algorithm has reached the maximum iteration limit.

# Arguments
- `info::AdaPDMIterationInfo`: Current iteration information
- `criteria::AdaPDMTerminationCriteria`: Termination criteria containing maximum iterations

# Returns
- `Bool`: `true` if iteration limit reached, `false` otherwise

# Side Effects
If iteration limit is reached:
- Sets `info.terminationStatus` to `ADA_PDM_TERMINATION_ITERATION_LIMIT`
- Sets `criteria.terminated` to `true`

See also: `checkTerminationCriteria`, `AdaPDMTerminationStatus`
"""
function checkIterationLimit(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)
    iter = length(info.presL2) - 1
    if iter >= criteria.maxIter
        info.terminationStatus = ADA_PDM_TERMINATION_ITERATION_LIMIT
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkTimeLimit(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)

Check if the algorithm has exceeded the maximum execution time limit.

# Arguments
- `info::AdaPDMIterationInfo`: Current iteration information containing total time
- `criteria::AdaPDMTerminationCriteria`: Termination criteria containing time limit

# Returns
- `Bool`: `true` if time limit exceeded, `false` otherwise

# Side Effects
If time limit is exceeded:
- Sets `info.terminationStatus` to `ADA_PDM_TERMINATION_TIME_LIMIT`
- Sets `criteria.terminated` to `true`

See also: `checkTerminationCriteria`, `AdaPDMTerminationStatus`
"""
function checkTimeLimit(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)
    if info.totalTime >= criteria.timeLimit
        info.terminationStatus = ADA_PDM_TERMINATION_TIME_LIMIT
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkNumericalError(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)

Check if the algorithm has encountered numerical errors (NaN or Inf values).

This function detects numerical instability by checking if any of the residual
values contain NaN or Inf, which typically indicates numerical breakdown.

# Arguments
- `info::AdaPDMIterationInfo`: Current iteration information containing residuals
- `criteria::AdaPDMTerminationCriteria`: Termination criteria object

# Returns
- `Bool`: `true` if numerical errors detected, `false` otherwise

# Side Effects
If numerical errors are detected:
- Sets `info.terminationStatus` to `ADA_PDM_TERMINATION_UNKNOWN`
- Sets `criteria.terminated` to `true`

# Checked Values
The function checks for NaN or Inf in:
- Primal residuals (L2 and L∞)
- Dual residuals (L2 and L∞)

See also: `checkTerminationCriteria`, `AdaPDMTerminationStatus`
"""
function checkNumericalError(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)
    if isnan(info.presL2[end]) || isinf(info.presL2[end]) ||
       isnan(info.dresL2[end]) || isinf(info.dresL2[end]) ||
       isnan(info.presLInf[end]) || isinf(info.presLInf[end]) ||
       isnan(info.dresLInf[end]) || isinf(info.dresLInf[end])
        info.terminationStatus = ADA_PDM_TERMINATION_UNKNOWN
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkUnboundedness(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)

Check if the problem is unbounded (placeholder implementation).

This function is a placeholder for detecting unbounded problems. Currently
always returns `false` as unboundedness detection is not implemented.

# Arguments
- `info::AdaPDMIterationInfo`: Current iteration information
- `criteria::AdaPDMTerminationCriteria`: Termination criteria object

# Returns
- `Bool`: Always `false` (not implemented)

# Note
This is a placeholder for future implementation of unboundedness detection.
Potential approaches include monitoring objective value divergence or
variable magnitude growth.

See also: `checkTerminationCriteria`, `AdaPDMTerminationStatus`
"""
function checkUnboundedness(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)
    return false
end 

"""
    checkTerminationCriteria(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)

Check all termination criteria and update termination status.

This function systematically checks all possible termination conditions in order
of priority and updates the algorithm status accordingly. It sets the stop
iteration if any termination condition is met.

# Arguments
- `info::AdaPDMIterationInfo`: Current iteration information to check
- `criteria::AdaPDMTerminationCriteria`: Termination criteria and status

# Termination Check Order
1. **Optimal termination**: All residuals below tolerances
2. **Iteration limit**: Maximum iterations reached
3. **Time limit**: Maximum execution time exceeded
4. **Numerical errors**: NaN or Inf detected in residuals
5. **Unboundedness**: Problem detected as unbounded (placeholder)

# Side Effects
- Updates `info.terminationStatus` based on termination reason
- Sets `criteria.terminated = true` if any criterion is met
- Sets `info.stopIter` to current iteration number upon termination

# Usage
```julia
# Check termination after each iteration
checkTerminationCriteria(info, criteria)
if criteria.terminated
    break
end
```

See also: `checkOptimalTermination`, `checkIterationLimit`, `checkTimeLimit`
"""
function checkTerminationCriteria(info::AdaPDMIterationInfo, criteria::AdaPDMTerminationCriteria)
    while true 
        if checkOptimalTermination(info, criteria)
            break 
        end 

        if checkIterationLimit(info, criteria)
            break 
        end     

        if checkTimeLimit(info, criteria)
            break 
        end 

        if checkNumericalError(info, criteria)
            break 
        end 

        if checkUnboundedness(info, criteria)
            break 
        end 

        break 
    end 
    
    if criteria.terminated 
        info.stopIter = length(info.presL2) - 1
    end 
end 

"""
    getTerminationStatus(status::AdaPDMTerminationStatus)

Convert termination status enum to human-readable string.

# Arguments
- `status::AdaPDMTerminationStatus`: Termination status enum value

# Returns
- `String`: Human-readable description of termination status

See also: `AdaPDMTerminationStatus`, `AdaPDMIterationInfo`
"""
function getTerminationStatus(status::AdaPDMTerminationStatus)
    if status == ADA_PDM_TERMINATION_OPTIMAL
        return "ADA_PDM_TERMINATION_OPTIMAL"
    elseif status == ADA_PDM_TERMINATION_ITERATION_LIMIT
        return "ADA_PDM_TERMINATION_ITERATION_LIMIT"
    elseif status == ADA_PDM_TERMINATION_TIME_LIMIT
        return "ADA_PDM_TERMINATION_TIME_LIMIT"
    elseif status == ADA_PDM_TERMINATION_UNBOUNDED
        return "ADA_PDM_TERMINATION_UNBOUNDED"
    elseif status == ADA_PDM_TERMINATION_UNKNOWN
        return "ADA_PDM_TERMINATION_UNKNOWN"
    else
        return "ADA_PDM_TERMINATION_UNSPECIFIED"
    end
end
