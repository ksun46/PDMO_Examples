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

# Examples
```julia
# Unit L2 ball (radius 1)
f = IndicatorBallL2(1.0)
x = [0.5, 0.5]
val = f(x)  # Returns 0.0 since ||x||₂ = √0.5 < 1

# Point outside the ball
x = [2.0, 2.0]
val = f(x)  # Returns +∞ since ||x||₂ = 2√2 > 1

# Proximal operator (projection onto ball)
f = IndicatorBallL2(1.0)
x = [3.0, 4.0]  # ||x||₂ = 5
prox_x = proximalOracle(f, x)  # Returns [0.6, 0.8] (normalized to unit length)
```

# Applications
- Constraint sets in optimization
- Regularization in machine learning
- Trust region methods
- Robust optimization
- Signal processing (bounded energy constraints)
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