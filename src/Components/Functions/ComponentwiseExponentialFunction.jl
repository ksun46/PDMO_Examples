"""
    ComponentwiseExponentialFunction(coefficients::Vector{Float64})

Represents a component-wise exponential function f(x) = ∑ᵢ aᵢ exp(xᵢ).

# Mathematical Definition
f(x) = ∑ᵢ aᵢ exp(xᵢ)

where a = coefficients is a vector of non-negative weights.

# Arguments
- `coefficients::Vector{Float64}`: Vector of non-negative coefficients

# Constructors
- `ComponentwiseExponentialFunction(coefficients)`: With specified coefficients
- `ComponentwiseExponentialFunction(n::Int64)`: With unit coefficients (ones vector)

# Properties
- **Smooth**: Yes, exponential functions are infinitely differentiable
- **Convex**: Yes, exponential functions are convex
- **Proximal**: No, proximal operator requires Lambert W function (not implemented)

# Mathematical Properties
- **Gradient**: ∇f(x) = [a₁ exp(x₁), a₂ exp(x₂), ..., aₙ exp(xₙ)]
- **Hessian**: ∇²f(x) = diag(a₁ exp(x₁), a₂ exp(x₂), ..., aₙ exp(xₙ))

# Examples
```julia
# Standard exponential function f(x) = exp(x₁) + exp(x₂)
f = ComponentwiseExponentialFunction([1.0, 1.0])
x = [0.0, 1.0]
val = f(x)  # Returns exp(0) + exp(1) = 1 + e ≈ 3.718

# Weighted exponential function f(x) = 2exp(x₁) + 3exp(x₂)
f = ComponentwiseExponentialFunction([2.0, 3.0])
x = [0.0, 0.0]
val = f(x)  # Returns 2*1 + 3*1 = 5

# Gradient computation
grad = gradientOracle(f, x)  # Returns [2.0, 3.0]
```

# Applications
- Exponential utility functions
- Log-sum-exp approximations
- Entropic regularization
- Barrier methods in optimization
- Statistical modeling (exponential families)

# Note
The proximal operator requires the Lambert W function and is not currently implemented.
"""
struct ComponentwiseExponentialFunction <: AbstractFunction 
    coefficients::Vector{Float64}
    function ComponentwiseExponentialFunction(coefficients::Vector{Float64})
        if length(coefficients) == 0
            error("coefficients must be non-empty")
        end 
        if any(coefficients .< -ZeroTolerance)
            error("coefficients must be non-negative")
        end 
        return new(coefficients)
    end 

    function ComponentwiseExponentialFunction(n::Int64)
        return new(ones(n))
    end 
end 

isConvex(::Type{ComponentwiseExponentialFunction}) = true 
isSmooth(::Type{ComponentwiseExponentialFunction}) = true 
isProximal(::Type{ComponentwiseExponentialFunction}) = false  # TODO: LambertW function 

function (f::ComponentwiseExponentialFunction)(x::Vector{Float64})
    @assert length(f.coefficients) == length(x)
    return sum(f.coefficients .* exp.(x))
end 

function gradientOracle!(y::Vector{Float64}, f::ComponentwiseExponentialFunction, x::Vector{Float64}, enableParallel::Bool=false)
    @assert length(f.coefficients) == length(x)
    @assert length(y) == length(x)
    y .= f.coefficients .* exp.(x)
end 

function gradientOracle(f::ComponentwiseExponentialFunction, x::Vector{Float64}, enableParallel::Bool=false)
    y = similar(x)
    gradientOracle!(y, f, x, enableParallel)
    return y
end 


# function proximalOracle!(y::Vector{Float64}, f::ComponentwiseExponentialFunction, x::Vector{Float64}, gamma::Float64 = 1.0, enableParallel::Bool=false)
#     @assert length(f.coefficients) == length(x)
#     @assert length(y) == length(x)
#     # TODO: The proximal operator requires calculation of the LambertW function. Do we need this now? 
# end 

# function proximalOracle(f::ComponentwiseExponentialFunction, x::Vector{Float64}, gamma::Float64 = 1.0, enableParallel::Bool=false)
#     y = similar(x)
#     proximalOracle!(y, f, x, gamma, enableParallel)
#     return y
# end 