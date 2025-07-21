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

# Implementation Details
- Projection formula handles both `radiusIndex = 1` and `radiusIndex = dim` cases
- Uses tolerance `FeasTolerance` for membership testing
- Efficient O(n) implementation avoiding matrix operations
- Handles edge cases (zero norm, negative radius) robustly

# Applications
- **Second-Order Cone Programming (SOCP)**: Constraint ``\\|Ax + b\\|_2 \\leq c^T x + d``
- **Robust Optimization**: Uncertainty sets and robust constraints
- **Signal Processing**: Beamforming and antenna array design
- **Machine Learning**: Support vector machines with nonlinear kernels
- **Control Theory**: Linear matrix inequalities and stability analysis

# Examples
```julia
# Create 3D SOC with radius at last position
f = IndicatorSOC(3, 3)

# Test point inside the cone
x_in = [0.5, 0.3, 1.0]  # ||[0.5, 0.3]||₂ = 0.583 < 1.0
@assert f(x_in) == 0.0  # Inside cone

# Test point outside the cone  
x_out = [2.0, 1.0, 1.0]  # ||[2.0, 1.0]||₂ = 2.236 > 1.0
@assert f(x_out) == Inf  # Outside cone

# Project onto cone
x_proj = proximalOracle(f, x_out)
@assert f(x_proj) == 0.0  # Projection is in cone

# Alternative: radius at first position
f_alt = IndicatorSOC(3, 1)
x_alt = [1.0, 0.5, 0.3]  # ||[0.5, 0.3]||₂ = 0.583 < 1.0
@assert f_alt(x_alt) == 0.0  # Inside cone
```

# Geometric Interpretation
The second-order cone is the set of points where the Euclidean norm of the vector part 
is bounded by the scalar radius part. It has several equivalent representations:
- **Ice cream cone**: The familiar "ice cream cone" shape in 3D
- **Epigraph**: Epigraph of the Euclidean norm function
- **Intersection**: Can be represented as intersection of half-spaces

# Performance Notes
- Projection requires O(n) operations (single norm computation)
- Memory allocation minimal - uses in-place operations when possible
- Numerically stable for well-conditioned problems
- Handles degenerate cases (zero radius, zero vector) gracefully


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
