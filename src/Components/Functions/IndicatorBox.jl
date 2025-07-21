"""
    IndicatorBox(lb::NumericVariable, ub::NumericVariable)

Indicator function of a box constraint set [lb, ub].

# Mathematical Definition
The indicator function of the box constraint set:
{x : lb ≤ x ≤ ub} (element-wise inequalities)

f(x) = 0    if lb ≤ x ≤ ub (element-wise)
f(x) = +∞   otherwise

# Arguments
- `lb::NumericVariable`: Lower bound vector/scalar
- `ub::NumericVariable`: Upper bound vector/scalar

# Properties
- **Smooth**: No, not differentiable on the boundary
- **Convex**: Yes, indicator functions of convex sets are convex
- **Proximal**: Yes, has explicit proximal operator (projection onto box)
- **Set Indicator**: Yes, this is an indicator function

# Mathematical Properties
- **Proximal Operator**: Element-wise projection onto the box
  - prox_f(x) = clamp(x, lb, ub) = max(lb, min(x, ub))

# Examples
```julia
# Unit box constraint [-1, 1]ⁿ
lb = [-1.0, -1.0]
ub = [1.0, 1.0]
f = IndicatorBox(lb, ub)
x = [0.5, -0.5]
val = f(x)  # Returns 0.0 since x is within bounds

# Point outside the box
x = [2.0, -2.0]
val = f(x)  # Returns +∞ since x violates bounds

# Proximal operator (projection onto box)
f = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
x = [2.0, -2.0]
prox_x = proximalOracle(f, x)  # Returns [1.0, -1.0] (clamped to bounds)
```

# Applications
- Variable bounds in optimization
- Constraint sets in quadratic programming
- Image processing (pixel value bounds)
- Control systems (actuator limits)
- Portfolio optimization (position limits)
"""
struct IndicatorBox <: AbstractFunction 
    lb::NumericVariable
    ub::NumericVariable

    function IndicatorBox(lb::NumericVariable, ub::NumericVariable)
        if size(lb) != size(ub)
            error("IndicatorBox: lb and ub have different shapes.")
        end
        if any(lb .> ub)
            error("IndicatorBox: infeasible domain; some lb > ub. ")
        end 
        new(lb, ub)
    end 
end 

# Override traits for IndicatorBox
isProximal(f::Type{<:IndicatorBox}) = true 
isConvex(f::Type{<:IndicatorBox}) = true
isSet(f::Type{<:IndicatorBox}) = true

# function value
function (f::IndicatorBox)(x::NumericVariable, enableParallel::Bool=false)
    # if size(x) != size(f.lb)
    #     error("IndicatorBox: function evaluation encountered dimension mismatch.")
    # end
    
    if isa(x, Number)
        if x < f.lb - FeasTolerance || x > f.ub + FeasTolerance
            return Inf
        else
            return 0.0
        end
    else
        for k in eachindex(x)
            if x[k] < f.lb[k] - FeasTolerance || x[k] > f.ub[k] + FeasTolerance
                return Inf
            end
        end
        return 0.0
    end
end

# proximal oracle
function proximalOracle!(y::NumericVariable, f::IndicatorBox, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # if size(x) != size(f.lb) || size(y) != size(f.lb)
    #     error("IndicatorBox: proximal oracle encountered dimension mismatch.")
    # end
    
    if isa(x, Number)
        error("IndicatorBox: proximal oracle does not support in-place operations for scalar inputs.")
    end
    y .= clamp.(x, f.lb, f.ub)
end

function proximalOracle(f::IndicatorBox, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # if size(x) != size(f.lb)
    #     error("IndicatorBox: proximal oracle encountered dimension mismatch.")
    # end
    
    return isa(x, Number) ? clamp(x, f.lb, f.ub) : clamp.(x, f.lb, f.ub)
end 