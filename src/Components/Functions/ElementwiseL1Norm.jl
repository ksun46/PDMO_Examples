"""
    ElementwiseL1Norm(coefficient::Float64=1.0)

Represents the element-wise L1 norm function f(x) = coefficient * ||x||₁.

# Mathematical Definition
f(x) = coefficient * ∑ᵢ |xᵢ|

where ||x||₁ is the L1 norm (sum of absolute values).

# Arguments
- `coefficient::Float64=1.0`: Positive scaling coefficient

# Properties
- **Smooth**: No, not differentiable at zero
- **Convex**: Yes, L1 norm is convex
- **Proximal**: Yes, has explicit proximal operator (soft thresholding)

# Mathematical Properties
- **Subdifferential**: ∂f(x) = coefficient * sign(x) (element-wise)
- **Proximal Operator**: Soft thresholding operator
  - prox_γf(x)ᵢ = sign(xᵢ) * max(0, |xᵢ| - γ*coefficient)

# Examples
```julia
# Standard L1 norm
f = ElementwiseL1Norm()
x = [1.0, -2.0, 3.0]
val = f(x)  # Returns |1| + |-2| + |3| = 6

# Scaled L1 norm with coefficient 0.5
f = ElementwiseL1Norm(0.5)
x = [4.0, -6.0]
val = f(x)  # Returns 0.5 * (4 + 6) = 5

# Proximal operator (soft thresholding)
f = ElementwiseL1Norm(1.0)
x = [2.0, -3.0, 0.5]
prox_x = proximalOracle(f, x, 1.0)  # γ = 1.0
# Returns [1.0, -2.0, 0.0] (soft thresholding with threshold 1.0)
```

# Applications
- Sparse regression (LASSO)
- Compressed sensing
- Feature selection
- Regularization in machine learning
- Signal denoising
"""
struct ElementwiseL1Norm <: AbstractFunction 
    coefficient::Float64

    function ElementwiseL1Norm(coe::Float64=1.0)
        if (coe < 0.0)
            error("ElementwiseL1Norm: negative coefficient. ")
        end 
        new(coe)
    end 
end 

# override traits
isProximal(f::Type{<:ElementwiseL1Norm}) = true 
isConvex(f::Type{<:ElementwiseL1Norm}) = true

# function value
function (f::ElementwiseL1Norm)(x::NumericVariable, enableParallel::Bool=false)
    return norm(x, 1) * f.coefficient
end

# proximal oracle
function proximalOracle!(y::NumericVariable, f::ElementwiseL1Norm, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # @assert(size(x) == size(y), "ElementwiseL1Norm: proximal oracle encountered dimension mismatch.")
    if isa(x, Number)
        error("ElementwiseL1Norm: proximal oracle does not support in-place operations for scalar inputs.")
    end

    gl = gamma * f.coefficient
    @inbounds @simd for i in eachindex(x)
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end 
end

function proximalOracle(f::ElementwiseL1Norm, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        gl = gamma * f.coefficient
        return x + (x <= -gl ? gl : (x >= gl ? -gl : -x))
    end

    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)   
    return y 
end
