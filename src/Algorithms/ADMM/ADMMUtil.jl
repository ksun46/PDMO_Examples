"""
    ADMMUtil.jl

Utility functions for logging and displaying ADMM algorithm progress and results.

This module provides formatted logging functions for tracking ADMM iterations,
displaying algorithm progress, and summarizing final results.
"""

"""
    ADMMLog(iter::Int, info::ADMMIterationInfo, param::ADMMParam, rhoUpdated::Bool = false; final::Bool = false) -> Bool

Log ADMM iteration progress with formatted output.

This function provides detailed logging of ADMM algorithm progress, including
iteration numbers, objective values, residuals, and timing information. It
automatically formats the output with appropriate headers and handles special
cases like penalty parameter updates.

# Arguments
- `iter::Int`: Current iteration number (0 for initialization)
- `info::ADMMIterationInfo`: Current iteration information containing metrics
- `param::ADMMParam`: ADMM parameters including log interval
- `rhoUpdated::Bool = false`: Whether penalty parameter ρ was updated this iteration
- `final::Bool = false`: Whether this is the final iteration log

# Returns
- `Bool`: `true` if logging was performed, `false` if skipped due to log interval

# Logging Format
The log displays the following columns:
- **ITERATION**: Iteration number
- **AL_OBJ**: Augmented Lagrangian objective value
- **OBJ**: Primal objective value
- **PRES(l2)**: Primal residual L2 norm
- **PRES(lInf)**: Primal residual L∞ norm
- **DRES(l2)**: Dual residual L2 norm
- **DRES(lInf)**: Dual residual L∞ norm
- **TIME**: Elapsed time in seconds

# Header Display
- Headers are displayed at iteration 0 (initialization)
- Headers are repeated every 20 log intervals for readability
- Headers clearly identify each column for easy interpretation

# Special Annotations
- When `rhoUpdated = true`, displays the new ρ value: `(rho <- X.XX)`
- Final iteration logs are always displayed regardless of log interval

# Logging Control
- Logs are displayed based on `param.logInterval`
- Iteration 0 is always logged (initialization)
- Final iteration is always logged if `final = true`
- Intermediate iterations are logged every `param.logInterval` iterations

# Performance Notes
- Uses `Printf.@sprintf` for efficient formatted output
- Minimal computational overhead when logging is skipped
- Thread-safe logging operations

# Usage Examples
```julia
# Log initialization
ADMMLog(0, info, param, false)

# Log regular iteration
logged = ADMMLog(100, info, param, false)

# Log iteration with penalty parameter update
ADMMLog(150, info, param, true)

# Log final iteration
ADMMLog(iter, info, param, false, final=true)
```
"""
function ADMMLog(iter, info::ADMMIterationInfo, param::ADMMParam, rhoUpdated::Bool = false; final::Bool = false) 
    if param.logLevel < 1
        return false  
    end 

    if (final == false && iter > 0 && iter % param.logInterval != 0)
        return false 
    end 

    header = false 
    if (iter == 0)
        header = true 
    elseif (iter > param.logInterval && (iter/param.logInterval) % 20 == 1)
        header = true 
    end 

    if (header)
        Printf.@printf("%10s ", "ITERATION") 
        Printf.@printf("%12s ", "AL_OBJ")   
        Printf.@printf("%12s ", "OBJ")  
        Printf.@printf("%12s ", "PRES(l2)")
        Printf.@printf("%12s ", "PRES(lInf)")  
        Printf.@printf("%12s ", "DRES(l2)")
        Printf.@printf("%12s ", "DRES(lInf)")
        Printf.@printf("%9s\n", "TIME")
    end 
    
    alObj = info.alObj[end]
    obj = info.obj[end]
    presL2 = info.presL2[end]
    presLInf = info.presLInf[end]
    dresL2 = info.dresL2[end]
    dresLInf = info.dresLInf[end]
    time = info.totalTime

    if (rhoUpdated)
        newRho = info.rhoHistory[end][1]
        Printf.@printf("%10d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %9.2f (rho <- %6.2f)\n",
            iter, 
            alObj, obj, 
            presL2, presLInf, 
            dresL2, dresLInf, 
            time, newRho)
    else
        Printf.@printf("%10d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %9.2f\n", 
            iter, 
            alObj, obj, 
            presL2, presLInf, 
            dresL2, dresLInf,
            time)
    end 

    return true 
end 

"""
    ADMMLog(info::ADMMIterationInfo, trueObj::Float64 = Inf)

Display a comprehensive summary of ADMM algorithm results.

This function provides a detailed summary of the ADMM algorithm's final state,
including termination status, objective values, residuals, iteration count,
timing information, and optional true objective comparison.

# Arguments
- `info::ADMMIterationInfo`: Final iteration information from ADMM algorithm
- `trueObj::Float64 = Inf`: True objective value for comparison (optional)

# Summary Information
The summary displays:
- **Solver Status**: Human-readable termination status
- **Objective**: Final primal objective value
- **Pres (L2)**: Final primal residual L2 norm
- **Pres (LInf)**: Final primal residual L∞ norm
- **Dres (L2)**: Final dual residual L2 norm
- **Dres (LInf)**: Final dual residual L∞ norm
- **Stop. Iter**: Iteration at which algorithm terminated
- **Total Time**: Total computation time in seconds
- **True Obj. Diff**: Difference from true objective (if provided)

# Termination Status
The termination status is converted to a human-readable string indicating:
- Optimal solution found
- Iteration limit reached
- Time limit exceeded
- Infeasible problem detected
- Unbounded problem detected
- Numerical issues encountered
- Unknown termination cause

# Objective Comparison
When `trueObj` is provided and finite:
- Computes absolute difference: `|trueObj - finalObj|`
- Useful for benchmarking and validation
- Displays "Unknown" if no true objective is available

# Formatting
- Uses consistent scientific notation for numerical values
- Aligns columns for easy readability
- Provides clear labels for each metric
- Uses appropriate precision for different value types

# Usage Examples
```julia
# Basic summary
ADMMLog(info)

# Summary with true objective comparison
ADMMLog(info, -142.857)

# After algorithm completion
result = BipartiteADMM(graph, param)
ADMMLog(result, known_optimal_value)
```

# Notes
- This function is typically called after algorithm completion
- Provides a complete picture of algorithm performance
- Useful for debugging convergence issues
- Essential for performance benchmarking
"""
function ADMMLog(info::ADMMIterationInfo, logLevel::Int64, trueObj::Float64=Inf)
    if logLevel < 1
        return
    end 

    @PDMOInfo logLevel "ADMM Summary: "
    Printf.@printf("    Solver Status   =   %s\n", getTerminationStatus(info.terminationStatus))
    Printf.@printf("    Objective       = %12.4e\n", info.obj[end])
    Printf.@printf("    Pres (L2)       = %12.4e\n", info.presL2[end])
    Printf.@printf("    Pres (LInf)     = %12.4e\n", info.presLInf[end])
    Printf.@printf("    Dres (L2)       = %12.4e\n", info.dresL2[end])
    Printf.@printf("    Dres (LInf)     = %12.4e\n", info.dresLInf[end])
    Printf.@printf("    Stop. Iter      = %12d\n",  info.stopIter) 
    Printf.@printf("    Total Time      = %12.2f\n", info.totalTime)
    if (trueObj < Inf)
        diff = abs(trueObj - info.obj[end])
        Printf.@printf("    True Obj. Diff  = %12.2f\n", diff)
    else 
        Printf.@printf("    True Obj. Diff  = Unknown\n")
    end 
end 
