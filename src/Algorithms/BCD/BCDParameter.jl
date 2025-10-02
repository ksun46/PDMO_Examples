"""
    BCDParam

Parameters for the Block Coordinate Descent algorithm.

# Fields
- `blockOrderRule::AbstractBlockUpdateOrder`: Strategy for selecting which blocks to update and in what order
- `solver::AbstractBCDSubproblemSolver`: Solver for individual block subproblems
- `dresTolL2::Float64`: L2 norm tolerance for dual residual convergence
- `dresTolLInf::Float64`: L∞ norm tolerance for dual residual convergence
- `maxIter::Int64`: Maximum number of iterations allowed
- `timeLimit::Float64`: Maximum time limit for the algorithm (in seconds)
- `logInterval::Int64`: Interval for logging iteration information

# Constructor
```julia
param = BCDParam(
    blockOrderRule = CyclicRule(),
    solver = BCDOriginalSubproblemSolver(),
    dresTolL2 = 1e-6,
    dresTolLInf = 1e-6,
    maxIter = 1000,
    timeLimit = 3600.0,
    logInterval = 10
)
```

# Notes
- The algorithm terminates when either dual residual tolerance is satisfied, maximum iterations reached, or time limit exceeded
- Block order is determined by the `blockOrderRule` at the beginning of each iteration
- Subproblems are solved using the specified `solver` strategy
"""
struct BCDParam   
    blockOrderRule::AbstractBlockUpdateOrder
    solver::AbstractBCDSubproblemSolver 
        
    dresTolL2::Float64
    dresTolLInf::Float64

    maxIter::Int64 
    timeLimit::Float64
    logInterval::Int64
    logLevel::Int64
    
    function BCDParam(;
        blockOrderRule::AbstractBlockUpdateOrder = CyclicRule(),
        solver::AbstractBCDSubproblemSolver = BCDProximalSubproblemSolver(),
        dresTolL2::Float64 = 1e-6,
        dresTolLInf::Float64 = 1e-6,
        maxIter::Int64 = 1000,
        timeLimit::Float64 = 3600.0,
        logInterval::Int64 = 10, 
        logLevel::Int64 = 1
    )
        if maxIter <= 0
            error("BCDParam: maxIter must be positive")
        end
        if dresTolL2 <= 0
            error("BCDParam: dresTolL2 must be positive")
        end
        if dresTolLInf <= 0
            error("BCDParam: dresTolLInf must be positive")
        end
        if timeLimit <= 0
            error("BCDParam: timeLimit must be positive")
        end
        if logInterval <= 0
            error("BCDParam: logInterval must be positive")
        end
        
        new(blockOrderRule, solver, dresTolL2, dresTolLInf, maxIter, timeLimit, logInterval, logLevel)
    end
end