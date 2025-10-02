"""
    IndicatorBallL2(r::Float64)

Indicator function of the L2 ball with radius r.

# Mathematical Definition
The indicator function of the set B₂(r) = {x : ||x||₂ ≤ r}:

f(x) = 0    if ||x||₂ ≤ r
f(x) = +∞   otherwise

# Arguments
- `r::Float64`: Radius of the L2 ball (must be positive)

# Properties
- **Smooth**: No, not differentiable on the boundary
- **Convex**: Yes, indicator functions of convex sets are convex
- **Proximal**: Yes, has explicit proximal operator (projection onto ball)
- **Set Indicator**: Yes, this is an indicator function

# Mathematical Properties
- **Proximal Operator**: Projection onto the L2 ball
  - If ||x||₂ ≤ r: prox_f(x) = x
  - If ||x||₂ > r: prox_f(x) = (r/||x||₂) * x

"""
struct IndicatorBallL2 <: AbstractFunction
    r::Float64 
    function IndicatorBallL2(r::Float64)
        if r <= 0.0
            error("IndicatorBallL2: the radius must be positive (r = $r) ")
        end
        new(r)
    end 
end 

# Override traits for IndicatorBallL2
isProximal(::Type{IndicatorBallL2}) = true 
isConvex(::Type{IndicatorBallL2}) = true 
isSet(::Type{IndicatorBallL2}) = true 
isSupportedByJuMP(f::Type{<:IndicatorBallL2}) = true

# function value
function (f::IndicatorBallL2)(x::NumericVariable, enableParallel::Bool=false)
    if norm(x) <= f.r + FeasTolerance 
        return 0.0
    else
        return Inf
    end 
end 


function proximalOracle!(y::NumericVariable, f::IndicatorBallL2, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("IndicatorBallL2: proximal oracle does not support in-place operations for scalar inputs.")
    end
    if gamma < 0.0
        error("IndicatorBallL2: proximal oracle encountered gamma = $gamma < 0.")
    end

    norm_x = norm(x)
    if norm_x <= f.r + FeasTolerance
        y .= x
    else
        scal = f.r / norm_x
        y .= scal .* x
    end
end

function proximalOracle(f::IndicatorBallL2, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        if abs(x) <= f.r + FeasTolerance
            return x
        else
            return f.r * sign(x)
        end
    else
        y = similar(x)
        proximalOracle!(y, f, x, gamma, enableParallel)
        return y
    end
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorBallL2, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    # Set bounds on variables to tighten the domain: -r <= x[k] <= r
    for k in 1:length(var)
        JuMP.set_lower_bound(var[k], -g.r)
        JuMP.set_upper_bound(var[k], g.r)
    end
    
    # Add L2 ball constraint: ||x||_2 <= r
    JuMP.@constraint(model, sum(var[k]^2 for k in 1:length(var)) <= g.r^2)
    
    return nothing  # L2 ball constraint doesn't contribute to objective
end

