"""
    BCDLog(iter, info::BCDIterationInfo, param::BCDParam; final::Bool = false)

Log iteration information for the Block Coordinate Descent algorithm.

This function provides formatted output of algorithm progress including objective values,
dual residuals, and timing information. It handles both regular iteration logging and final
summary logging.

# Arguments
- `iter::Int`: Current iteration number (0 for initialization)
- `info::BCDIterationInfo`: Current iteration information containing residuals and objective
- `param::BCDParam`: Algorithm parameters containing logging interval
- `final::Bool=false`: Whether this is the final log entry

# Returns
- `Bool`: `true` if logging was performed, `false` if skipped due to logging interval

# Behavior
- **Header Display**: Shows column headers every 20 log intervals and at iteration 0
- **Selective Logging**: Only logs at specified intervals unless `final=true`
- **Formatted Output**: Displays iteration number, objective, dual residuals, and timing

# Output Format
```
ITERATION    OBJECTIVE     DRES(l2)    DRES(lInf)     TIME
        0    1.2345e+02    1.23e-04     1.23e-04     1.23
```

# Examples
```julia
# Regular iteration logging
logged = BCDLog(100, info, param)

# Force final logging regardless of interval
BCDLog(150, info, param; final=true)
```

See also: `BCDIterationInfo`, `BCDParam`
"""
function BCDLog(iter, info::BCDIterationInfo, param::BCDParam; final::Bool = false) 
    if param.logLevel < 1
        return false  
    end 

    if (final == false && iter > 1 && iter % param.logInterval != 0)
        return false 
    end 

    header = false 
    if (iter == 1)
        header = true 
    elseif (iter > param.logInterval && (iter/param.logInterval) % 20 == 1)
        header = true 
    end 

    if (header)
        Printf.@printf("%10s ", "ITERATION") 
        Printf.@printf("%12s ", "OBJECTIVE")  
        Printf.@printf("%12s ", "DRES(l2)")
        Printf.@printf("%12s ", "DRES(lInf)")
        Printf.@printf("%9s\n", "TIME")
    end 
    
    obj = !isempty(info.obj) ? info.obj[end] : 0.0
    dresL2 = !isempty(info.dresL2) ? info.dresL2[end] : 0.0
    dresLInf = !isempty(info.dresLInf) ? info.dresLInf[end] : 0.0
    time = info.totalTime

    Printf.@printf("%10d %12.4e %12.4e %12.4e %9.2f\n", 
            iter, 
            obj, 
            dresL2, dresLInf,
            time)

    return true 
end 

"""
    BCDLog(info::BCDIterationInfo, trueObj::Float64=Inf)

Display a comprehensive summary of the Block Coordinate Descent algorithm results.

This function prints a formatted summary of the algorithm's final state, including
convergence status, final objective and dual residual values, iteration count, and timing.
Optionally compares against a known true objective value.

# Arguments
- `info::BCDIterationInfo`: Final iteration information from the algorithm
- `trueObj::Float64=Inf`: True objective value for comparison (if known)

# Output Information
- **Solver Status**: Termination reason (optimal, iteration limit, time limit, etc.)
- **Objective**: Final objective value
- **Dual Residuals**: Final L2 and L∞ dual residuals  
- **Iteration Count**: Total iterations performed
- **Total Time**: Algorithm execution time in seconds
- **Objective Difference**: Difference from true objective (if provided)

# Examples
```julia
# Basic summary without true objective
BCDLog(info)

# Summary with true objective comparison
BCDLog(info, 42.0)
```

# Sample Output
```
BCD Summary: 
    Solver Status   =   BCD_TERMINATION_OPTIMAL
    Objective       =   4.2345e+01
    Dres (L2)       =   8.7654e-07
    Dres (LInf)     =   1.1111e-06
    Stop. Iter      =          150
    Total Time      =        12.34
    True Obj. Diff  =         0.23
```

See also: `BCDIterationInfo`, `getTerminationStatus`
"""
function BCDLog(info::BCDIterationInfo, logLevel::Int64, trueObj::Float64=Inf)
    if logLevel < 1
        return false  
    end 

    @PDMOInfo logLevel "BCD Summary: "
    Printf.@printf("    Solver Status   =   %s\n", getTerminationStatus(info.terminationStatus))
    
    if !isempty(info.obj)
        Printf.@printf("    Objective       = %12.4e\n", info.obj[end])
    else
        Printf.@printf("    Objective       = %12s\n", "N/A")
    end
    
    if !isempty(info.dresL2)
        Printf.@printf("    Dres (L2)       = %12.4e\n", info.dresL2[end])
    else
        Printf.@printf("    Dres (L2)       = %12s\n", "N/A")
    end
    
    if !isempty(info.dresLInf)
        Printf.@printf("    Dres (LInf)     = %12.4e\n", info.dresLInf[end])
    else
        Printf.@printf("    Dres (LInf)     = %12s\n", "N/A")
    end
    
    Printf.@printf("    Stop. Iter      = %12d\n",  info.stopIter) 
    Printf.@printf("    Total Time      = %12.2f\n", info.totalTime)
    
    if (trueObj < Inf && !isempty(info.obj))
        diff = abs(trueObj - info.obj[end])
        Printf.@printf("    True Obj. Diff  = %12.2f\n", diff)
    else 
        Printf.@printf("    True Obj. Diff  = Unknown\n")
    end 
end 

