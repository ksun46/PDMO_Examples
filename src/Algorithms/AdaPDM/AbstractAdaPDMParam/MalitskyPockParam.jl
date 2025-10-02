"""
    MalitskyPockParam <: AbstractAdaPDMParam

Parameters for the Malitsky-Pock primal-dual algorithm with adaptive step sizes.

The Malitsky-Pock algorithm is an adaptive primal-dual method that automatically adjusts
step sizes during iteration using backtracking line search. It is designed for solving
composite optimization problems of the form: minimize f(x) + g(Ax), where f and g are
convex functions and A is a linear operator.

# Fields

## Algorithm Parameters
- `initialTheta::Float64`: Initial primal step size ratio (θ₀)
- `initialSigma::Float64`: Initial dual step size (σ₀)
- `backtrackDescentRatio::Float64`: Backtracking descent ratio ∈ (0,1)
- `t::Float64`: Ratio parameter for primal-dual step size relationship
- `mu::Float64`: Backtracking parameter ∈ (0,1) for step size reduction

## Convergence Tolerances
- `presTolL2::Float64`: Primal residual tolerance in L2 norm
- `dresTolL2::Float64`: Dual residual tolerance in L2 norm
- `presTolLInf::Float64`: Primal residual tolerance in L∞ norm
- `dresTolLInf::Float64`: Dual residual tolerance in L∞ norm

## Iteration Control
- `lineSearchMaxIter::Int64`: Maximum iterations for backtracking line search
- `maxIter::Int64`: Maximum number of algorithm iterations
- `logInterval::Int64`: Interval for logging algorithm progress
- `timeLimit::Float64`: Time limit in seconds
- `logLevel::Int64`: Level for logging information

# Algorithm Theory
The Malitsky-Pock algorithm uses adaptive step sizes with the following key features:
1. **Adaptive Step Sizes**: Automatically adjusts θₖ and σₖ based on progress
2. **Backtracking Line Search**: Ensures sufficient decrease in objective
3. **Convergence Guarantee**: Proven convergence without knowing operator norms

The algorithm updates follow:
1. x̄ᵏ = xᵏ + θₖ(xᵏ - xᵏ⁻¹)  
2. yᵏ⁺¹ = prox_{σₖg*}(yᵏ + σₖAx̄ᵏ)
3. xᵏ⁺¹ = prox_{τₖf}(xᵏ - τₖA^Tyᵏ⁺¹)

Where step sizes are adapted based on backtracking conditions.

# Advantages
- **Parameter-Free**: No need to estimate operator norms or Lipschitz constants
- **Robust**: Adaptive step sizes handle problem conditioning automatically
- **Efficient**: Often converges faster than fixed step size methods

# References
- Malitsky, Y., & Pock, T. (2018). A first-order primal-dual algorithm with linesearch
- Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems

See also: `CondatVuParam`, `AdaPDMParam`, `runAdaPDM`
"""
mutable struct MalitskyPockParam <: AbstractAdaPDMParam
    # parameters for the algorithm
    initialTheta::Float64           # initial primal step size ratio
    initialSigma::Float64           # initial dual step size 
    backtrackDescentRatio::Float64  # backtrack descent ratio in (0,1)
    t::Float64                      # ratio for primal dual step sizes
    mu::Float64                     # backtrack parameter in (0,1)
    
    # tolerances for the stopping criteria 
    presTolL2::Float64        # primal residual tolerance in L2 norm 
    dresTolL2::Float64        # dual residual tolerance in L2 norm 
    presTolLInf::Float64      # primal residual tolerance in LInf norm 
    dresTolLInf::Float64      # dual residual tolerance in LInf norm 

    # limit and logging parameters 
    lineSearchMaxIter::Int64 # Maximum number of iterations for line search
    maxIter::Int64           # Maximum number of iterations
    logInterval::Int64       # Interval for logging information
    timeLimit::Float64       # Time limit in seconds
    logLevel::Int64          # Level for logging information
end


"""
    MalitskyPockParam(mbp::MultiblockProblem; kwargs...)

Create parameters for the Malitsky-Pock adaptive primal-dual algorithm.

This constructor creates an adaptive parameter set that automatically adjusts step sizes
during the algorithm execution. The Malitsky-Pock method is particularly useful when
operator norms or Lipschitz constants are unknown or difficult to estimate.

# Arguments
- `mbp::MultiblockProblem`: The composite multiblock problem to solve

# Keyword Arguments

## Algorithm Parameters
- `initialTheta::Float64=1.0`: Initial primal step size ratio (θ₀). Recommended: [0.5, 2.0]
- `initialSigma::Float64=1.0`: Initial dual step size (σ₀). Recommended: [0.1, 10.0]
- `backtrackDescentRatio::Float64=0.95`: Descent ratio for backtracking ∈ (0,1). Higher values = more aggressive
- `t::Float64=1.0`: Ratio parameter for primal-dual step size relationship
- `mu::Float64=0.8`: Backtracking reduction factor ∈ (0,1). Smaller values = more conservative backtracking

## Convergence Tolerances
- `presTolL2::Float64=1e-4`: Primal residual tolerance in L2 norm
- `dresTolL2::Float64=1e-4`: Dual residual tolerance in L2 norm
- `presTolLInf::Float64=1e-6`: Primal residual tolerance in L∞ norm
- `dresTolLInf::Float64=1e-6`: Dual residual tolerance in L∞ norm

## Algorithm Control
- `lineSearchMaxIter::Int64=1000`: Maximum iterations for backtracking line search per step
- `maxIter::Int64=10000`: Maximum number of algorithm iterations
- `logInterval::Int64=1000`: Logging interval for progress reporting
- `timeLimit::Float64=3600.0`: Time limit in seconds
- `logLevel::Int64=1`: Level for logging information
# Returns
- `MalitskyPockParam`: Configured parameters for the Malitsky-Pock algorithm

# Examples
```julia
# Basic usage with default adaptive parameters
mbp = MultiblockProblem()
# ... set up composite problem ...
param = MalitskyPockParam(mbp)

# Custom initial step sizes for difficult problems
param = MalitskyPockParam(mbp; 
    initialTheta=0.5, 
    initialSigma=0.1,
    mu=0.5)  # More conservative backtracking

# High precision settings
param = MalitskyPockParam(mbp; 
    presTolL2=1e-8, 
    dresTolL2=1e-8, 
    maxIter=50000)

# Solve with Malitsky-Pock algorithm
result = runAdaPDM(mbp, param)
```

# Parameter Tuning Guidelines

## Initial Step Sizes
- **Large Problems**: Start with smaller `initialSigma` (0.1-0.5)
- **Well-Conditioned**: Use larger initial steps (1.0-2.0)
- **Ill-Conditioned**: Use conservative initial steps (0.1-0.5)

## Backtracking Parameters
- **Conservative**: `mu=0.5`, `backtrackDescentRatio=0.9`
- **Aggressive**: `mu=0.9`, `backtrackDescentRatio=0.99`
- **Balanced**: `mu=0.8`, `backtrackDescentRatio=0.95` (default)

# Algorithm Advantages
1. **No Parameter Estimation**: No need to compute operator norms or Lipschitz constants
2. **Automatic Adaptation**: Step sizes adjust based on problem geometry
3. **Robust Convergence**: Proven convergence guarantees with line search
4. **Efficient**: Often outperforms fixed step size methods

# Problem Requirements
The problem must satisfy the composite structure required by Malitsky-Pock:
- Convex functions with available proximal and gradient oracles
- Linear coupling constraints between blocks
- Validated using `checkCompositeProblemValidity!`

# Error Conditions
Throws an error if the input problem is not a valid composite problem structure.

See also: `runAdaPDM`, `CondatVuParam`, `AdaPDMParam`
"""
function MalitskyPockParam(mbp::MultiblockProblem; 
    # parameters for the algorithm
    initialTheta::Float64=1.0,      
    initialSigma::Float64=1.0,  
    backtrackDescentRatio::Float64=0.95,   
    t::Float64=1.0,
    mu::Float64=0.8,
    presTolL2::Float64=1e-4,
    dresTolL2::Float64=1e-4,
    presTolLInf::Float64=1e-6,
    dresTolLInf::Float64=1e-6,
    lineSearchMaxIter::Int64=1000,
    maxIter::Int64=10000,
    logInterval::Int64=1000,
    timeLimit::Float64=3600.0,
    logLevel::Int64=1)

    if checkCompositeProblemValidity!(mbp) == false 
        error("AdaPDMParam: the input problem is not a valid composite problem.")
    end 

    # for block in mbp.blocks[1:end-1]
    #     if isa(block.f, Zero) == false 
    #         error("MaliskyPockParam: the input problem is not a valid composite problem.")
    #     end 
    # end 
    return MalitskyPockParam(
        initialTheta, 
        initialSigma, 
        backtrackDescentRatio,
        t, 
        mu,
        presTolL2, 
        dresTolL2, 
        presTolLInf, 
        dresTolLInf, 
        lineSearchMaxIter,
        maxIter, 
        logInterval, 
        timeLimit, 
        logLevel)

end 