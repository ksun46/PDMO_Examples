"""
    Zero()

Represents the zero function f(x) = 0 for all x.

# Mathematical Definition
f(x) = 0 for all x ∈ ℝⁿ

# Properties
- **Smooth**: Yes, constant functions are infinitely differentiable
- **Convex**: Yes, constant functions are convex
- **Proximal**: Yes, has explicit proximal operator

# Mathematical Properties
- **Gradient**: ∇f(x) = 0 (zero vector/scalar)
- **Proximal Operator**: prox_γf(x) = x (identity function)

See also: [`AbstractFunction`](@ref), [`proximalOracle!`](@ref), [`gradientOracle!`](@ref)
"""
struct Zero <: AbstractFunction end

# Override traits for Zero function
isProximal(::Type{Zero}) = true 
isSmooth(::Type{Zero}) = true 
isConvex(::Type{Zero}) = true
isSupportedByJuMP(::Type{Zero}) = true 

# function value
(::Zero)(x::NumericVariable, enableParallel::Bool=false) = 0.0

# gradient oracle 
function gradientOracle!(y::NumericVariable, f::Zero, x::NumericVariable, enableParallel::Bool=false)
    # @assert(size(y) == size(x), "Zero: gradient oracle encountered dimension mismatch.")
    if isa(x, Number)
        error("Zero: gradient oracle does not support in-place operations for scalar inputs.")
    end 
    fill!(y, 0.0) 
end 

function gradientOracle(f::Zero, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        return 0.0
    else
        y = similar(x)
        gradientOracle!(y, f, x, enableParallel)
        return y
    end
end 

# proximal oracle
function proximalOracle!(y::NumericVariable, f::Zero, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # @assert(size(y) == size(x), "Zero: proximal oracle encountered dimension mismatch.")
    if isa(x, Number)
        error("Zero: proximal oracle does not support in-place operations for scalar inputs.")
    end 
    copy!(y, x)
end 

function proximalOracle(f::Zero, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        return x
    else
        y = similar(x)
        proximalOracle!(y, f, x, gamma, enableParallel)
        return y
    end
end

# JuMP APIs  
function JuMPAddProximableFunction(g::Zero, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    return nothing 
end

function JuMPAddSmoothFunction(f::Zero, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    return nothing
end 