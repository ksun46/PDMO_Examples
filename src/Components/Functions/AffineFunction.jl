"""
    AffineFunction(A::NumericVariable, r::Float64=0.0)

Represents an affine function of the form f(x) = ⟨A, x⟩ + r, where A is a coefficient
vector/matrix and r is a scalar offset.

# Mathematical Definition
- For vector input x: f(x) = A'x + r  
- For scalar input x: f(x) = A*x + r

# Arguments
- `A::NumericVariable`: Coefficient vector/matrix/scalar
- `r::Float64=0.0`: Scalar offset term

# Properties
- **Smooth**: Yes, gradient is constant
- **Convex**: Yes, affine functions are convex
- **Proximal**: Yes, has explicit proximal operator

# Mathematical Properties
- **Gradient**: ∇f(x) = A (constant)
- **Proximal Operator**: prox_γf(x) = x - γA

# Examples
```julia
# Linear function f(x) = 2x₁ + 3x₂ + 1
A = [2.0, 3.0]
r = 1.0
f = AffineFunction(A, r)
x = [1.0, 2.0]
val = f(x)  # Returns 2*1 + 3*2 + 1 = 9

# Scalar function f(x) = 5x + 2
f = AffineFunction(5.0, 2.0)
val = f(3.0)  # Returns 5*3 + 2 = 17
```

# Applications
- Linear constraints in optimization
- Objective functions in linear programming
- Penalty terms in regularization
- Building blocks for more complex functions
"""
struct AffineFunction <: AbstractFunction 
    A::NumericVariable 
    r::Float64

    function AffineFunction(A::NumericVariable, r::Float64=0.0)
        new(A, r)
    end 
end 

# Override traits for AffineFunction
isProximal(f::Type{<:AffineFunction}) = true
isSmooth(f::Type{<:AffineFunction}) = true
isConvex(f::Type{<:AffineFunction}) = true

# function value
function (f::AffineFunction)(x::NumericVariable, enableParallel::Bool=false)
    # @assert(size(x) == size(f.A), "AffineFunction: function evaluation encountered dimension mismatch.")
    if isa(x, Number)
        return f.A * x + f.r
    else
        return dot(f.A, x) + f.r
    end
end

# gradient oracle
function gradientOracle!(y::NumericVariable, f::AffineFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        error("AffineFunction: gradient oracle does not support in-place operations for scalar inputs.")
    end
    y .= f.A
end

function gradientOracle(f::AffineFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        return f.A
    else
        y = similar(x)
        gradientOracle!(y, f, x, enableParallel)    
        return y
    end
end

# proximal oracle
function proximalOracle!(y::NumericVariable, f::AffineFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)
    if isa(x, Number)
        error("AffineFunction: proximal oracle does not support in-place operations for scalar inputs.")
    end
    if gamma < 0.0
        error("AffineFunction: proximal oracle encountered gamma = $gamma < 0. ")
    end
    y .= x .- gamma .* f.A
end

function proximalOracle(f::AffineFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)
    if isa(x, Number)
        if gamma < 0.0
            error("AffineFunction: proximal oracle encountered gamma = $gamma < 0. ")
        end
        return x - gamma * f.A
    else
        y = similar(x)
        proximalOracle!(y, f, x, gamma, enableParallel)
        return y
    end
end
