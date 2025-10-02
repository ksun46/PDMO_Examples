"""
    BCDTerminationCriteria

Structure containing termination criteria and status for the Block Coordinate Descent algorithm.

This structure defines the convergence tolerances and limits used to determine when
the BCD algorithm should terminate. It tracks both the criteria values and the
current termination status.

# Fields
- `dresTolL2::Float64`: Dual residual tolerance in L2 norm
- `dresTolLInf::Float64`: Dual residual tolerance in L∞ norm
- `maxIter::Int64`: Maximum number of iterations allowed
- `timeLimit::Float64`: Maximum execution time in seconds
- `terminated::Bool`: Whether the algorithm has terminated

# Termination Conditions
The algorithm terminates when ANY of the following conditions is met:
1. **Optimal**: All dual residuals fall below their respective tolerances
2. **Iteration Limit**: Maximum number of iterations reached
3. **Time Limit**: Maximum execution time exceeded
4. **Numerical Error**: NaN or Inf detected in residuals
5. **Unbounded**: Problem is detected as unbounded (placeholder)

# Constructor
```julia
BCDTerminationCriteria(;
    dresTolL2::Float64 = 1e-4,
    dresTolLInf::Float64 = 1e-4,
    maxIter::Int64 = 1000,
    timeLimit::Float64 = 3600.0,
    terminated::Bool = false)
```

See also: `BCDParam`, `BCDIterationInfo`, `checkTerminationCriteria`
"""
mutable struct BCDTerminationCriteria
    dresTolL2::Float64
    dresTolLInf::Float64
    maxIter::Int64
    timeLimit::Float64
    terminated::Bool 

    function BCDTerminationCriteria(;
        dresTolL2::Float64 = 1e-4,
        dresTolLInf::Float64 = 1e-4,
        maxIter::Int64 = 1000,
        timeLimit::Float64 = 3600.0,
        terminated::Bool = false)
        
        return new(dresTolL2, dresTolLInf, maxIter, timeLimit, terminated)
    end
end

"""
    BCDTerminationCriteria(param::BCDParam)

Create termination criteria from BCD algorithm parameters.

This constructor extracts the termination criteria from a BCD parameter object
and validates that all values are within acceptable ranges.

# Arguments
- `param::BCDParam`: Algorithm parameters containing termination criteria

# Returns
- `BCDTerminationCriteria`: Initialized termination criteria object

# Validation
The constructor validates that:
- All tolerance values are non-negative
- Maximum iterations is positive
- Time limit is positive

# Throws
- `ErrorException`: If any validation fails

# Examples
```julia
# Create from BCD parameters
param = BCDParam(; dresTolL2=1e-6, maxIter=5000)
criteria = BCDTerminationCriteria(param)
```

See also: `BCDParam`, `BCDIterationInfo`
"""
function BCDTerminationCriteria(param::BCDParam)
    if param.dresTolL2 < 0
        error("BCDTermination: dresTolL2 must be nonnegative")
    end
    if param.dresTolLInf < 0
        error("BCDTermination: dresTolLInf must be nonnegative")
    end
    if param.maxIter <= 0
        error("BCDTermination: maxIter must be positive")
    end
    if param.timeLimit <= 0
        error("BCDTermination: timeLimit must be positive")
    end
    
    return BCDTerminationCriteria(
        dresTolL2 = param.dresTolL2,
        dresTolLInf = param.dresTolLInf,
        maxIter = param.maxIter,
        timeLimit = param.timeLimit,
        terminated = false)
end 

"""
    checkOptimalTermination(info::BCDIterationInfo, criteria::BCDTerminationCriteria)

Check if the algorithm has reached optimal termination based on dual residual tolerances.

This function determines if all dual residuals (in both L2 and L∞ norms)
have fallen below their respective tolerance thresholds, indicating convergence.

# Arguments
- `info::BCDIterationInfo`: Current iteration information containing residuals
- `criteria::BCDTerminationCriteria`: Termination criteria containing tolerances

# Returns
- `Bool`: `true` if optimal termination achieved, `false` otherwise

# Side Effects
If termination is achieved:
- Sets `info.terminationStatus` to `BCD_TERMINATION_OPTIMAL`
- Sets `criteria.terminated` to `true`

# Termination Condition
Returns `true` when ALL of the following hold:
- `dresL2 ≤ dresTolL2`
- `dresLInf ≤ dresTolLInf`

See also: `checkTerminationCriteria`, `BCDTerminationStatus`
"""
function checkOptimalTermination(info::BCDIterationInfo, criteria::BCDTerminationCriteria)
    if !isempty(info.dresL2) && !isempty(info.dresLInf) &&
       info.dresL2[end] <= criteria.dresTolL2 && 
       info.dresLInf[end] <= criteria.dresTolLInf
        info.terminationStatus = BCD_TERMINATION_OPTIMAL
        criteria.terminated = true
        return true
    end
    return false
end

"""
    checkIterationLimit(info::BCDIterationInfo, criteria::BCDTerminationCriteria)

Check if the algorithm has reached the maximum iteration limit.

# Arguments
- `info::BCDIterationInfo`: Current iteration information
- `criteria::BCDTerminationCriteria`: Termination criteria containing maximum iterations

# Returns
- `Bool`: `true` if iteration limit reached, `false` otherwise

# Side Effects
If iteration limit is reached:
- Sets `info.terminationStatus` to `BCD_TERMINATION_ITERATION_LIMIT`
- Sets `criteria.terminated` to `true`

See also: `checkTerminationCriteria`, `BCDTerminationStatus`
"""
function checkIterationLimit(info::BCDIterationInfo, criteria::BCDTerminationCriteria)
    iter = length(info.obj) - 1
    if iter >= criteria.maxIter
        info.terminationStatus = BCD_TERMINATION_ITERATION_LIMIT
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkTimeLimit(info::BCDIterationInfo, criteria::BCDTerminationCriteria)

Check if the algorithm has exceeded the maximum execution time limit.

# Arguments
- `info::BCDIterationInfo`: Current iteration information containing total time
- `criteria::BCDTerminationCriteria`: Termination criteria containing time limit

# Returns
- `Bool`: `true` if time limit exceeded, `false` otherwise

# Side Effects
If time limit is exceeded:
- Sets `info.terminationStatus` to `BCD_TERMINATION_TIME_LIMIT`
- Sets `criteria.terminated` to `true`

See also: `checkTerminationCriteria`, `BCDTerminationStatus`
"""
function checkTimeLimit(info::BCDIterationInfo, criteria::BCDTerminationCriteria)
    if info.totalTime >= criteria.timeLimit
        info.terminationStatus = BCD_TERMINATION_TIME_LIMIT
        criteria.terminated = true
        return true
    end
    return false
end 

"""
    checkNumericalError(info::BCDIterationInfo, criteria::BCDTerminationCriteria)

Check if the algorithm has encountered numerical errors (NaN or Inf values).

This function detects numerical instability by checking if any of the residual
or objective values contain NaN or Inf, which typically indicates numerical breakdown.

# Arguments
- `info::BCDIterationInfo`: Current iteration information containing residuals
- `criteria::BCDTerminationCriteria`: Termination criteria object

# Returns
- `Bool`: `true` if numerical errors detected, `false` otherwise

# Side Effects
If numerical errors are detected:
- Sets `info.terminationStatus` to `BCD_TERMINATION_UNKNOWN`
- Sets `criteria.terminated` to `true`

# Checked Values
The function checks for NaN or Inf in:
- Dual residuals (L2 and L∞)
- Objective values

See also: `checkTerminationCriteria`, `BCDTerminationStatus`
"""
function checkNumericalError(info::BCDIterationInfo, criteria::BCDTerminationCriteria)
    # Check dual residuals if they exist
    if !isempty(info.dresL2) && (isnan(info.dresL2[end]) || isinf(info.dresL2[end]))
        info.terminationStatus = BCD_TERMINATION_UNKNOWN
        criteria.terminated = true
        return true
    end
    
    if !isempty(info.dresLInf) && (isnan(info.dresLInf[end]) || isinf(info.dresLInf[end]))
        info.terminationStatus = BCD_TERMINATION_UNKNOWN
        criteria.terminated = true
        return true
    end
    
    # Check objective values if they exist
    if !isempty(info.obj) && (isnan(info.obj[end]) || isinf(info.obj[end]))
        info.terminationStatus = BCD_TERMINATION_UNKNOWN
        criteria.terminated = true
        return true
    end
    
    return false
end 

"""
    checkUnboundedness(info::BCDIterationInfo, criteria::BCDTerminationCriteria)

Check if the problem is unbounded (placeholder implementation).

This function is a placeholder for detecting unbounded problems. Currently
always returns `false` as unboundedness detection is not implemented.

# Arguments
- `info::BCDIterationInfo`: Current iteration information
- `criteria::BCDTerminationCriteria`: Termination criteria object

# Returns
- `Bool`: Always `false` (not implemented)

# Note
This is a placeholder for future implementation of unboundedness detection.
Potential approaches include monitoring objective value divergence or
variable magnitude growth.

See also: `checkTerminationCriteria`, `BCDTerminationStatus`
"""
function checkUnboundedness(info::BCDIterationInfo, criteria::BCDTerminationCriteria)
    return false
end 

"""
    checkTerminationCriteria(info::BCDIterationInfo, criteria::BCDTerminationCriteria)

Check all termination criteria and update termination status.

This function systematically checks all possible termination conditions in order
of priority and updates the algorithm status accordingly. It sets the stop
iteration if any termination condition is met.

# Arguments
- `info::BCDIterationInfo`: Current iteration information to check
- `criteria::BCDTerminationCriteria`: Termination criteria and status

# Termination Check Order
1. **Optimal termination**: All dual residuals below tolerances
2. **Iteration limit**: Maximum iterations reached
3. **Time limit**: Maximum execution time exceeded
4. **Numerical errors**: NaN or Inf detected in residuals or objectives
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
function checkTerminationCriteria(info::BCDIterationInfo, criteria::BCDTerminationCriteria)
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
        info.stopIter = length(info.obj) - 1 
    end 
end 

"""
    getTerminationStatus(status::BCDTerminationStatus)

Convert termination status enum to human-readable string.

# Arguments
- `status::BCDTerminationStatus`: Termination status enum value

# Returns
- `String`: Human-readable description of termination status

See also: `BCDTerminationStatus`, `BCDIterationInfo`
"""
function getTerminationStatus(status::BCDTerminationStatus)
    if status == BCD_TERMINATION_OPTIMAL
        return "BCD_TERMINATION_OPTIMAL"
    elseif status == BCD_TERMINATION_ITERATION_LIMIT
        return "BCD_TERMINATION_ITERATION_LIMIT"
    elseif status == BCD_TERMINATION_TIME_LIMIT
        return "BCD_TERMINATION_TIME_LIMIT"
    elseif status == BCD_TERMINATION_UNKNOWN
        return "BCD_TERMINATION_UNKNOWN"
    else
        return "BCD_TERMINATION_UNSPECIFIED"
    end
end