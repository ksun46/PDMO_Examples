""" 
    IndicatorNonnegativeOrthant()

Indicator function of the nonnegative orthant ℝⁿ₊.

# Mathematical Definition
The indicator function of the nonnegative orthant:
```
I_{ℝⁿ₊}(x) = 0     if xᵢ ≥ 0 for all i ∈ {1,...,n}
I_{ℝⁿ₊}(x) = +∞    otherwise
```

This represents the constraint that all components of x must be non-negative.

# Constructor
```julia
IndicatorNonnegativeOrthant()
```
No parameters needed - the constraint applies to any dimensional input.

# Properties
- **Smooth**: No, not differentiable at the boundary (xᵢ = 0)
- **Convex**: Yes, the nonnegative orthant is a convex cone
- **Proximal**: Yes, proximal operator is element-wise projection
- **Set Indicator**: Yes, represents the constraint set ℝⁿ₊
"""
struct IndicatorNonnegativeOrthant <: AbstractFunction 
    IndicatorNonnegativeOrthant() = new()
end 

isProximal(::Type{IndicatorNonnegativeOrthant}) = true 
isConvex(::Type{IndicatorNonnegativeOrthant}) = true 
isSet(::Type{IndicatorNonnegativeOrthant}) = true 
isSupportedByJuMP(f::Type{<:IndicatorNonnegativeOrthant}) = true

function (f::IndicatorNonnegativeOrthant)(x::NumericVariable, enableParallel::Bool=false)
    if any(x .< -FeasTolerance)
        return +Inf
    end
    return 0.0 
end 


function proximalOracle!(y::NumericVariable, f::IndicatorNonnegativeOrthant, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("IndicatorNonnegativeOrthant: proximal oracle does not support in-place operations for scalar inputs.")
    end
    if gamma < 0.0
        error("IndicatorNonnegativeOrthant: proximal oracle encountered gamma = $gamma < 0.")
    end

    @assert(size(x) == size(y), "IndicatorNonnegativeOrthant: input dimension mismatch.")

    if enableParallel && length(x) > 1000  # some threshold
        Threads.@threads for i in eachindex(x)
            @inbounds y[i] = max(x[i], 0.0)
        end
    else
        y .= max.(x, 0.0)
    end

end 

function proximalOracle(f::IndicatorNonnegativeOrthant, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        return max(x, 0.0)
    end 
    
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 

function JuMPAddProximableFunction(g::IndicatorNonnegativeOrthant, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    for k in 1:length(var)  
        JuMP.set_lower_bound(var[k], 0.0)
    end
    return nothing  # Nonnegative orthant constraints don't contribute to objective
end



 
