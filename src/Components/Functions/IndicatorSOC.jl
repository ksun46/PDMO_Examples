""" 
    IndicatorSOC(dim::Int64, radiusIndex::Int64)

Represents the indicator function of the second-order cone (also known as the Lorentz cone).

The indicator function of the second-order cone is defined as:
```math
I_{\\mathcal{L}^n}(x) = \\begin{cases}
0 & \\text{if } x \\in \\mathcal{L}^n \\\\
\\infty & \\text{otherwise}
\\end{cases}
```

where the second-order cone is defined as:
- If `radiusIndex = n`: ``\\mathcal{L}^n = \\{x \\in \\mathbb{R}^n : \\|x_{1:n-1}\\|_2 \\leq x_n\\}``
- If `radiusIndex = 1`: ``\\mathcal{L}^n = \\{x \\in \\mathbb{R}^n : \\|x_{2:n}\\|_2 \\leq x_1\\}``

# Mathematical Properties
- **Convex**: The SOC is a convex cone, making this a convex indicator function
- **Self-dual**: The SOC is self-dual under the standard inner product
- **Closed**: The SOC is closed, ensuring well-defined projections
- **Pointed**: The SOC has a non-empty interior and is pointed (contains no lines)

# Arguments
- `dim::Int64`: The dimension n of the ambient space (must be ≥ 2)
- `radiusIndex::Int64`: Position of the radius variable (must be 1 or `dim`)

# Fields
- `dim::Int64`: Dimension of the ambient space
- `radiusIndex::Int64`: Index of the radius variable (1 or `dim`)

# Function Properties
- `isProximal(IndicatorSOC)`: `true` - admits efficient proximal operator
- `isConvex(IndicatorSOC)`: `true` - indicator of convex cone
- `isSet(IndicatorSOC)`: `true` - indicator function of a set

# Proximal Operator
The proximal operator (projection onto SOC) is computed analytically:

For a point ``x`` with radius component ``r`` and vector component ``v``:
1. If ``\\|v\\|_2 \\leq r``: ``x`` is already in the cone, so ``\\text{proj}(x) = x``
2. If ``\\|v\\|_2 \\leq -r``: ``x`` is in the polar cone, so ``\\text{proj}(x) = 0``
3. Otherwise: ``\\text{proj}(x) = \\frac{\\|v\\|_2 + r}{2\\|v\\|_2}\\begin{pmatrix} v \\\\ \\|v\\|_2 \\end{pmatrix}``
"""
struct IndicatorSOC <: AbstractFunction 
    dim::Int64 
    radiusIndex::Int64 # index of the radius variable, which is either 1 or dim 
    function IndicatorSOC(dim::Int64, radiusIndex::Int64)
        @assert dim >= 2 "Dimension must be at least 2" 
        @assert radiusIndex == 1 || radiusIndex == dim "Radius index must be 1 or dim"
        new(dim, radiusIndex)
    end
end

isProximal(::Type{IndicatorSOC}) = true 
isConvex(::Type{IndicatorSOC}) = true 
isSet(::Type{IndicatorSOC}) = true
isSupportedByJuMP(f::Type{<:IndicatorSOC}) = true
 

function (f::IndicatorSOC)(x::Vector{Float64}, enableParallel::Bool=false)
    @assert length(x) == f.dim "Dimension of x must be equal to the dimension of the function"
    if f.radiusIndex == 1
        return norm(x[2:end]) <= x[1] + FeasTolerance ? 0.0 : +Inf
    else
        return norm(x[1:end-1]) <= x[end] + FeasTolerance ? 0.0 : +Inf
    end
end

function proximalOracle!(y::Vector{Float64}, f::IndicatorSOC, x::Vector{Float64}, gamma::Float64 = 1.0, enableParallel::Bool=false)
    @assert length(x) == f.dim "Dimension of x must be equal to the dimension of the function"
    @assert length(y) == f.dim "Dimension of y must be equal to the dimension of the function"
    
    vecNorm = f.radiusIndex == 1 ? norm(x[2:end]) : norm(x[1:end-1])
    radius = f.radiusIndex == 1 ? x[1] : x[end] 
    if vecNorm <= radius
        y .= x
    elseif vecNorm <= -radius
        y .= 0.0 
    else # vecNorm > |radius|
        scaler = (vecNorm + radius) / (2 * vecNorm + FeasTolerance)
        if f.radiusIndex == 1
            y[1] = scaler * vecNorm 
            y[2:end] .= scaler * x[2:end]
        else
            y[1:end-1] .= scaler * x[1:end-1]
            y[end] = scaler * vecNorm 
        end
    end 
end

function proximalOracle(f::IndicatorSOC, x::Vector{Float64}, gamma::Float64 = 1.0, enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorSOC, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    @assert length(var) == g.dim "Variable dimension must match SOC dimension"
    
    # Add second-order cone constraint based on radius position
    if g.radiusIndex == 1
        # Constraint: ||x[2:end]||_2 <= x[1]
        # JuMP format: [x[1]; x[2:end]] in SecondOrderCone()
        # JuMP.@constraint(model, [var[1]; var[2:end]] in JuMP.SecondOrderCone())
        JuMP.@constraint(model, dot(var[2:end], var[2:end]) <= var[1]^2)
        JuMP.@constraint(model, var[1] >= 0.0)
    else
        # Constraint: ||x[1:end-1]||_2 <= x[end]
        # JuMP format: [x[end]; x[1:end-1]] in SecondOrderCone()
        # JuMP.@constraint(model, [var[end]; var[1:end-1]] in JuMP.SecondOrderCone())
        JuMP.@constraint(model, dot(var[1:end-1], var[1:end-1]) <= var[end]^2)
        JuMP.@constraint(model, var[end] >= 0.0)
    end
    
    return nothing  # SOC constraints don't contribute to objective
end




