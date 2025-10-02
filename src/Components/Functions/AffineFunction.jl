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
isSupportedByJuMP(f::Type{<:AffineFunction}) = true

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

# JuMP support
function JuMPAddProximableFunction(g::AffineFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    # AffineFunction as a proximable function doesn't add constraints
    # (it's unconstrained), but it contributes to the objective
    # Create linear expression: A'x + r
    obj_expr = JuMP.AffExpr(g.r)  # Start with constant term
    
    # Add linear terms
    if isa(g.A, Number)
        # Scalar case: A*x[1] 
        JuMP.add_to_expression!(obj_expr, g.A * var[1])
    else
        # Vector case: A'*x = sum(A[i] * x[i])
        for i in 1:length(g.A)
            JuMP.add_to_expression!(obj_expr, g.A[i] * var[i])
        end
    end
    
    return obj_expr
end

function JuMPAddSmoothFunction(f::AffineFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    return JuMPAddProximableFunction(f, model, var)
end


