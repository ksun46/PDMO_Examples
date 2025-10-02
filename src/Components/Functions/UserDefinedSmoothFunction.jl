"""
    UserDefinedSmoothFunction(func::Function, gradientFunc::Function, convex::Bool=true)

Wrapper for user-defined smooth functions that have a gradient.

This allows users to define custom smooth functions by providing both the function 
evaluation and its gradient, enabling integration with gradient-based algorithms.

# Arguments
- `func::Function`: Function evaluation f(x) → Float64
- `gradientFunc::Function`: Gradient function x → ∇f(x)
- `convex::Bool=true`: Whether the function is convex

# Properties
- **Smooth**: Yes, by definition
- **Convex**: User-specified (default true)
- **Proximal**: No, user-defined functions typically don't have proximal oracles

# Mathematical Properties
- **Function evaluation**: f(x) provided by user
- **Gradient**: ∇f(x) provided by user

# Simple Quadratic Function
# f(x) = x₁² + 2x₂² + x₁x₂
func = x -> x[1]^2 + 2*x[2]^2 + x[1]*x[2]
gradientFunc = x -> [2*x[1] + x[2], 4*x[2] + x[1]]
# Non-convex Function
# f(x) = sin(x₁) + cos(x₂)
func = x -> sin(x[1]) + cos(x[2])
gradientFunc = x -> [cos(x[1]), -sin(x[2])]
# Rosenbrock Function (classic optimization test function)
# f(x) = (1-x₁)² + 100(x₂-x₁²)²
func = x -> (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
gradientFunc = x -> [-2*(1-x[1]) - 400*x[1]*(x[2]-x[1]^2), 200*(x[2]-x[1]^2)]
# Integration with Bipartization
```julia
# In your optimization problems
block_x = BlockVariable(xID)
# Requirements
- `func(x)` must return a Float64 value
- `gradientFunc(x)` must return a gradient of the same type as x
- Both functions must be consistent with the mathematical definition
- The gradient must satisfy: ∇f(x) = lim_{h→0} [f(x+h) - f(x)]/h
- For correctness, consider using automatic differentiation tools to compute gradients
"""
struct UserDefinedSmoothFunction <: AbstractFunction
    func::Function
    gradientFunc::Function
    convex::Bool

    function UserDefinedSmoothFunction(func::Function, gradientFunc::Function, convex::Bool=true)
        new(func, gradientFunc, convex)
    end
end

# Override traits for UserDefinedSmoothFunction
isSmooth(::Type{UserDefinedSmoothFunction}) = true
isConvex(f::UserDefinedSmoothFunction) = f.convex
isConvex(::Type{UserDefinedSmoothFunction}) = false  # Default to false, check instance
isProximal(::Type{UserDefinedSmoothFunction}) = false  # User-defined functions typically don't have proximal oracles

# Function evaluation
function (f::UserDefinedSmoothFunction)(x::NumericVariable, enableParallel::Bool=false)
    return f.func(x)
end

# Gradient oracle - in-place version
function gradientOracle!(grad::NumericVariable, f::UserDefinedSmoothFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        error("UserDefinedSmoothFunction: gradient oracle does not support in-place operations for scalar inputs.")
    end
    grad .= f.gradientFunc(x)
end

# Gradient oracle - allocating version
function gradientOracle(f::UserDefinedSmoothFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        return f.gradientFunc(x)
    else
        grad = similar(x)
        gradientOracle!(grad, f, x, enableParallel)
        return grad
    end
end

# Example usage:
#
# # Simple Quadratic Function
# # f(x) = x₁² + 2x₂² + x₁x₂
# func = x -> x[1]^2 + 2*x[2]^2 + x[1]*x[2]
# gradientFunc = x -> [2*x[1] + x[2], 4*x[2] + x[1]]
# # Non-convex Function
# # f(x) = sin(x₁) + cos(x₂)
# func = x -> sin(x[1]) + cos(x[2])
# gradientFunc = x -> [cos(x[1]), -sin(x[2])]
# # Integration with Bipartization
# # In your optimization problems
# block_x = BlockVariable(xID)
# addBlockVariable!(nlp, block_x) 