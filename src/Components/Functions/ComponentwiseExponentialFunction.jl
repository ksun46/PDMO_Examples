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

# Standard exponential function f(x) = exp(x₁) + exp(x₂)
# Weighted exponential function f(x) = 2exp(x₁) + 3exp(x₂)
# Gradient computation
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
isSupportedByJuMP(f::Type{<:ComponentwiseExponentialFunction}) = true 

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

# JuMP support
function JuMPAddSmoothFunction(f::ComponentwiseExponentialFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    @assert length(f.coefficients) == length(var) "ComponentwiseExponentialFunction: coefficients length must match variable dimension"
    
    # Create nonlinear expression: ∑ᵢ aᵢ exp(xᵢ)
    # This returns a nonlinear expression that JuMP can handle
    # The actual expression will be handled by JuMP's nonlinear interface
    return f.coefficients' * exp.(var)
end 