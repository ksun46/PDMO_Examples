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
isSupportedByJuMP(f::Type{<:ElementwiseL1Norm}) = true

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

# JuMP support
function JuMPAddProximableFunction(g::ElementwiseL1Norm, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    dim = length(var)
    
    # Create auxiliary variables to represent |x_i| (non-negative)
    aux = JuMP.@variable(model, [k = 1:dim], lower_bound = 0.0)
    
    # Add constraints: aux[k] >= |var[k]|
    # This is modeled as: aux[k] >= var[k] and aux[k] >= -var[k]
    JuMP.@constraint(model, [k in 1:dim], var[k] <= aux[k])
    JuMP.@constraint(model, [k in 1:dim], -var[k] <= aux[k])
    
    # Create objective term: coefficient * sum(aux) = coefficient * ||x||_1
    obj_expr = JuMP.AffExpr(0.0)
    for k in 1:dim 
        JuMP.add_to_expression!(obj_expr, g.coefficient * aux[k])
    end
    
    return obj_expr
end
