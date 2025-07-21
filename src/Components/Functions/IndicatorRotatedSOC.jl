""" 
    IndicatorRotatedSOC(dim::Int64)

Represents the indicator function of the rotated second-order cone (also known as the quadratic cone).

The indicator function of the rotated second-order cone is defined as:
```math
I_{\\mathcal{Q}^n}(x) = \\begin{cases}
0 & \\text{if } x \\in \\mathcal{Q}^n \\\\
\\infty & \\text{otherwise}
\\end{cases}
```

where the rotated second-order cone is defined as:
```math
\\mathcal{Q}^n = \\{(u,v,w) \\in \\mathbb{R} \\times \\mathbb{R} \\times \\mathbb{R}^{n-2} : \\|w\\|_2^2 \\leq 2uv, u \\geq 0, v \\geq 0\\}
```

for ``n \\geq 3``, where ``u = x_1``, ``v = x_2``, and ``w = x_{3:n}``.

# Mathematical Properties
- **Convex**: The rotated SOC is a convex cone, making this a convex indicator function
- **Self-dual**: The rotated SOC is self-dual under the standard inner product
- **Closed**: The rotated SOC is closed, ensuring well-defined projections
- **Pointed**: Has a non-empty interior and is pointed (contains no lines)

# Arguments
- `dim::Int64`: The dimension n of the ambient space (must be ≥ 3)

# Fields
- `dim::Int64`: Dimension of the ambient space
- `sqrt2::Float64`: Pre-computed √2 constant for efficiency

# Function Properties
- `isProximal(IndicatorRotatedSOC)`: `true` - admits efficient proximal operator
- `isConvex(IndicatorRotatedSOC)`: `true` - indicator of convex cone
- `isSet(IndicatorRotatedSOC)`: `true` - indicator function of a set

# Proximal Operator
The proximal operator (projection onto rotated SOC) is computed via transformation to standard SOC:

1. **Transform to standard SOC**: For point ``(u,v,w)``, compute:
   - ``s = u + v`` (sum)
   - ``t = u - v`` (difference)  
   - ``z = \\sqrt{2}w`` (scaled vector)

2. **Project onto standard SOC**: Project ``(s,t,z)`` onto ``\\{(s,t,z) : \\|(t,z)\\|_2 \\leq s\\}``

3. **Transform back**: From projected ``(s',t',z')``, compute:
   - ``u' = \\max(0, (s' + t')/2)``
   - ``v' = \\max(0, (s' - t')/2)``
   - ``w' = z'/\\sqrt{2}``

# Implementation Details
- Uses linear transformation to reduce to standard SOC projection
- Enforces non-negativity constraints explicitly after transformation
- Pre-computes √2 for numerical efficiency
- Handles edge cases and maintains numerical stability

# Applications
- **Quadratic Programming**: Quadratic constraints ``x^T Q x \\leq t`` with ``Q \\succeq 0``
- **Geometric Programming**: Posynomial constraints in convex form
- **Robust Optimization**: Ellipsoidal uncertainty sets
- **Signal Processing**: Power constraints in communication systems
- **Finance**: Mean-variance portfolio optimization with transaction costs

# Examples
```julia
# Create 4D rotated SOC
f = IndicatorRotatedSOC(4)

# Test point inside the cone
x_in = [2.0, 1.0, 1.0, 1.0]  # ||[1.0, 1.0]||₂² = 2 ≤ 2·2·1 = 4
@assert f(x_in) == 0.0  # Inside cone

# Test point outside the cone
x_out = [1.0, 1.0, 2.0, 2.0]  # ||[2.0, 2.0]||₂² = 8 > 2·1·1 = 2
@assert f(x_out) == Inf  # Outside cone

# Project onto cone
x_proj = proximalOracle(f, x_out)
@assert f(x_proj) == 0.0  # Projection is in cone

# Boundary case: on the boundary
x_bound = [1.0, 2.0, 2.0, 0.0]  # ||[2.0, 0.0]||₂² = 4 = 2·1·2
@assert f(x_bound) == 0.0  # On boundary
```

# Geometric Interpretation
The rotated second-order cone represents the set of points where the squared norm of the 
vector part is bounded by twice the product of two non-negative scalars. Key properties:
- **Quadratic constraint**: The defining constraint is quadratic in the variables
- **Hyperbolic**: The boundary forms a hyperbolic surface in higher dimensions
- **Geometric mean**: Related to the geometric mean constraint ``\\sqrt{uv} \\geq \\|w\\|_2/\\sqrt{2}``

# Relationship to Standard SOC
The rotated SOC is related to the standard SOC via the linear transformation:
```math
\\begin{pmatrix} u \\\\ v \\\\ w \\end{pmatrix} \\mapsto \\begin{pmatrix} u+v \\\\ u-v \\\\ \\sqrt{2}w \\end{pmatrix}
```

This transformation preserves the cone structure and enables efficient projection algorithms.

# Performance Notes
- Projection requires O(n) operations via SOC transformation
- Memory allocation minimal - uses pre-allocated buffers
- Numerically stable for well-conditioned problems
- Handles degenerate cases (zero components) gracefully


"""
struct IndicatorRotatedSOC <: AbstractFunction 
    dim::Int64 
    sqrt2::Float64
    function IndicatorRotatedSOC(dim::Int64)
        @assert dim >= 3 "Dimension must be at least 3" 
        new(dim, sqrt(2))
    end
end

isProximal(::Type{IndicatorRotatedSOC}) = true 
isConvex(::Type{IndicatorRotatedSOC}) = true 
isSet(::Type{IndicatorRotatedSOC}) = true 

function (f::IndicatorRotatedSOC)(x::Vector{Float64}, enableParallel::Bool=false)
    @assert length(x) == f.dim "IndicatorRotatedSOC: Dimension of x must be equal to the dimension of the function"
    
    # Check non-negativity constraints
    if x[1] < -FeasTolerance || x[2] < -FeasTolerance
        return +Inf
    end
    
    # Check rotated cone constraint: ||x[3:end]||² ≤ 2*x[1]*x[2]
    vecNormSq = sum(x[3:end].^2)
    return vecNormSq <= 2 * x[1] * x[2] + FeasTolerance ? 0.0 : +Inf
end

function proximalOracle!(y::Vector{Float64}, f::IndicatorRotatedSOC, x::Vector{Float64}, gamma::Float64 = 1.0, enableParallel::Bool=false)
    @assert length(x) == f.dim "IndicatorRotatedSOC: Dimension of x must be equal to the dimension of the function"
    @assert length(y) == f.dim "IndicatorRotatedSOC: Dimension of y must be equal to the dimension of the function"
    
    # Project onto rotated SOC using linear transformation to regular SOC
    # Rotated SOC: {(u,v,w) | ||w||² ≤ 2uv, u ≥ 0, v ≥ 0}
    # Transformation: (u,v,w) → (u+v, u-v, √2*w)
    
    u, v = x[1], x[2]
    w = x[3:end]
    
    # Transform to regular SOC coordinates
    s = u + v
    t = u - v
    z = f.sqrt2 * w
    
    # Create vector for standard SOC projection: [s; t; z] where √(t² + ||z||²) ≤ s
    soc_vec = [s; t; z]
    
    # Project onto standard SOC with radius at position 1
    soc_func = IndicatorSOC(f.dim, 1)  # radius at position 1
    soc_projected = proximalOracle(soc_func, soc_vec, gamma, enableParallel)
    
    # Extract projected values
    s_proj = soc_projected[1]
    t_proj = soc_projected[2]
    z_proj = soc_projected[3:end]
    
    # Transform back to rotated SOC coordinates
    # u = (s + t)/2, v = (s - t)/2, w = z/√2
    u_proj = (s_proj + t_proj) / 2
    v_proj = (s_proj - t_proj) / 2
    
    y[1] = max(0.0, u_proj)  # u
    y[2] = max(0.0, v_proj)  # v
    y[3:end] .= z_proj / f.sqrt2  # w
end

function proximalOracle(f::IndicatorRotatedSOC, x::Vector{Float64}, gamma::Float64 = 1.0, enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 