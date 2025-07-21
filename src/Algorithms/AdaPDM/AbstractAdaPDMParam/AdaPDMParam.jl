"""
    AdaPDMParam <: AbstractAdaPDMParam

Parameters for the Adaptive Primal-Dual Method.

# Fields
- `t::Float64`: Primal/dual step size ratio
- `stepSizeEpsilon::Float64`: Step size parameter epsilon, i.e., 1e-6 in the paper
- `stepSizeNu::Float64`: Step size parameter nu, i.e., 1.2 in the paper
- `initialGamma::Float64`: Initial gamma, i.e., gamma_0
- `initialGammaPrev::Float64`: Initial gamma prev, i.e., gamma_{-1}
- `initialNormEstimate::Float64`: ||A|| or its estimate eta_0
- `presTolL2::Float64`: Primal residual tolerance in L2 norm
- `dresTolL2::Float64`: Dual residual tolerance in L2 norm
- `presTolLInf::Float64`: Primal residual tolerance in LInf norm
- `dresTolLInf::Float64`: Dual residual tolerance in LInf norm
- `maxIter::Int64`: Maximum number of iterations
- `logInterval::Int64`: Interval for logging information
- `timeLimit::Float64`: Time limit in seconds
"""
mutable struct AdaPDMParam <: AbstractAdaPDMParam
    # parameters for the algorithm 
    t::Float64                  # primal/dual step size ratio 
    stepSizeEpsilon::Float64    # step size parameter epsilon, i.e., 1e-6 in the paper 
    stepSizeNu::Float64         # step size parameter nu, i.e., 1.2 in the paper 
    initialGamma::Float64       # initial gamma, i.e., gamma_0
    initialGammaPrev::Float64   # initial gamma prev, i.e., gamma_{-1}
    initialNormEstimate::Float64           # ||A|| or its estimate eta_0 

    # tolerances for the stopping criteria 
    presTolL2::Float64          # primal residual tolerance in L2 norm 
    dresTolL2::Float64          # dual residual tolerance in L2 norm 
    presTolLInf::Float64        # primal residual tolerance in LInf norm 
    dresTolLInf::Float64        # dual residual tolerance in LInf norm 

    # limit and logging parameters 
    maxIter::Int64           # Maximum number of iterations
    logInterval::Int64       # Interval for logging information
    timeLimit::Float64       # Time limit in seconds
end

"""
    AdaPDMParam(mbp::MultiblockProblem; t::Float64=1.0, stepSizeEpsilon::Float64=1e-6, stepSizeNu::Float64=1.2, opNormEstimateProvided::Float64=Inf, presTolL2::Float64=1e-4, dresTolL2::Float64=1e-4, presTolLInf::Float64=1e-6, dresTolLInf::Float64=1e-6, maxIter::Int64=10000, logInterval::Int64=1000, timeLimit::Float64=3600.0)

Construct a parameter object for the Adaptive Primal-Dual Method.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to analyze.      
- `t::Float64=1.0`: Primal/dual step size ratio.
- `stepSizeEpsilon::Float64=1e-6`: Step size parameter epsilon.
- `stepSizeNu::Float64=1.2`: Step size parameter nu.
- `opNormEstimateProvided::Float64=Inf`: Provided estimate of the operator norm. If Inf, it will be computed.
- `presTolL2::Float64=1e-4`: Primal residual tolerance in L2 norm.
- `dresTolL2::Float64=1e-4`: Dual residual tolerance in L2 norm.
- `presTolLInf::Float64=1e-6`: Primal residual tolerance in LInf norm.
- `dresTolLInf::Float64=1e-6`: Dual residual tolerance in LInf norm.
- `maxIter::Int64=10000`: Maximum number of iterations.
- `logInterval::Int64=1000`: Interval for logging information.
- `timeLimit::Float64=3600.0`: Time limit in seconds.

# Returns
- `AdaPDMParam`: A parameter object for the Adaptive Primal-Dual Method.

# Throws
- `ErrorException`: If the input problem is not a valid composite problem.
"""
function AdaPDMParam(mbp::MultiblockProblem; 
    t::Float64=1.0,
    stepSizeEpsilon::Float64=1e-6,
    stepSizeNu::Float64=1.2,
    opNormEstimateProvided::Float64=Inf,   
    presTolL2::Float64=1e-4,
    dresTolL2::Float64=1e-4,
    presTolLInf::Float64=1e-6,
    dresTolLInf::Float64=1e-6,
    maxIter::Int64=10000,
    logInterval::Int64=1000,
    timeLimit::Float64=3600.0)
    
    if checkCompositeProblemValidity!(mbp) == false 
        error("AdaPDMParam: the input problem is not a valid composite problem.")
    end 
    
    # estimate ||A|| 
    opNormEstimate = opNormEstimateProvided < Inf ? opNormEstimateProvided : computeNormEstimate(mbp)

    # estimate gamma_0 and gamma_{-1}
    initialGamma = 1/(2.0 * stepSizeNu * t * opNormEstimate)
    initialGammaPrev = initialGamma 

    # initialize the parameter object 
    return AdaPDMParam(
        t, 
        stepSizeEpsilon, 
        stepSizeNu, 
        initialGamma, 
        initialGammaPrev, 
        opNormEstimate,  
        presTolL2, 
        dresTolL2, 
        presTolLInf, 
        dresTolLInf, 
        maxIter, 
        logInterval, 
        timeLimit)
end