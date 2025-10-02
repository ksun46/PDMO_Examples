"""
    UserDefinedProximalFunction(func::Function, proximalFunc::Function, convex::Bool=true)

Wrapper for user-defined functions that have a proximal operator.

This allows users to define custom functions by providing both the function evaluation
and its proximal operator, enabling integration with proximal algorithms.

# Arguments
- `func::Function`: Function evaluation f(x) → Float64
- `proximalFunc::Function`: Proximal operator (x, γ) → result
- `convex::Bool=true`: Whether the function is convex

# Properties
- **Smooth**: No, proximal functions are typically non-smooth
- **Convex**: User-specified (default true)
- **Proximal**: Yes, by definition

# Mathematical Properties
- **Function evaluation**: f(x) provided by user
- **Proximal operator**: prox_γf(x) provided by user

# L1 Norm Function
# f(x) = λ||x||₁ (L1 norm with coefficient λ)
λ = 0.5
func = x -> λ * sum(abs.(x))
proximalFunc = (x, gamma) -> sign.(x) .* max.(abs.(x) .- gamma * λ, 0.0)  # Soft thresholding
# Indicator Function of Box Constraints
# f(x) = I_{[a,b]}(x) (indicator function of box [a,b])
a, b = -1.0, 1.0
func = x -> all(a .<= x .<= b) ? 0.0 : Inf
proximalFunc = (x, gamma) -> clamp.(x, a, b)  # Projection onto box
# Custom Regularization Function
# f(x) = α * g(x) where g has known proximal operator
α = 0.1
func = x -> α * myCustomFunction(x)
proximalFunc = (x, gamma) -> myCustomProximal(x, gamma * α)
# Integration with Bipartization
```julia
# In your optimization problems
block_x = BlockVariable(xID)
# Requirements
- `func(x)` must return a Float64 value
- `proximalFunc(x, gamma)` must return a result of the same type as x
- Both functions must be consistent with the mathematical definition
- The proximal operator must satisfy: prox_γf(x) = argmin_z { f(z) + (1/(2γ))||z - x||² }
"""
struct UserDefinedProximalFunction <: AbstractFunction
    func::Function
    proximalFunc::Function
    convex::Bool

    function UserDefinedProximalFunction(func::Function, proximalFunc::Function, convex::Bool=true)
        new(func, proximalFunc, convex)
    end
end

# Override traits for UserDefinedProximalFunction
isProximal(::Type{UserDefinedProximalFunction}) = true
isSmooth(::Type{UserDefinedProximalFunction}) = false  # Proximal functions are typically non-smooth
isConvex(f::UserDefinedProximalFunction) = f.convex
isConvex(::Type{UserDefinedProximalFunction}) = false  # Default to false, check instance

# Function evaluation
function (f::UserDefinedProximalFunction)(x::NumericVariable, enableParallel::Bool=false)
    return f.func(x)
end

# Proximal oracle - in-place version
function proximalOracle!(y::NumericVariable, f::UserDefinedProximalFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("UserDefinedProximalFunction: proximal oracle does not support in-place operations for scalar inputs.")
    end
    if gamma <= 0.0
        error("UserDefinedProximalFunction: proximal oracle encountered gamma = $gamma <= 0.")
    end
    y .= f.proximalFunc(x, gamma)
end

# Proximal oracle - allocating version
function proximalOracle(f::UserDefinedProximalFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if gamma <= 0.0
        error("UserDefinedProximalFunction: proximal oracle encountered gamma = $gamma <= 0.")
    end
    if isa(x, Number)
        return f.proximalFunc(x, gamma)
    else
        y = similar(x)
        proximalOracle!(y, f, x, gamma, enableParallel)
        return y
    end
end

# Example usage:
#
# # L1 Norm Function
# # f(x) = λ||x||₁ (L1 norm with coefficient λ)
# λ = 0.5
# func = x -> λ * sum(abs.(x))
# proximalFunc = (x, gamma) -> sign.(x) .* max.(abs.(x) .- gamma * λ, 0.0)  # Soft thresholding
# # Indicator Function of Box Constraints
# # f(x) = I_{[a,b]}(x) (indicator function of box [a,b])
# a, b = -1.0, 1.0
# func = x -> all(a .<= x .<= b) ? 0.0 : Inf
# proximalFunc = (x, gamma) -> clamp.(x, a, b)  # Projection onto box
# # Custom Regularization Function
# # f(x) = α * g(x) where g has known proximal operator
# α = 0.1
# func = x -> α * myCustomFunction(x)
# proximalFunc = (x, gamma) -> myCustomProximal(x, gamma * α)
# # Integration with Bipartization
# # In your optimization problems
# block_x = BlockVariable(xID)
# addBlockVariable!(nlp, block_x) 