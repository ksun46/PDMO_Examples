""" 
    IndicatorNonnegativeOrthant()

Indicator function of the nonnegative orthant ℝⁿ₊.

# Mathematical Definition
The indicator function of the nonnegative orthant:
```
I_{ℝⁿ₊}(x) = 0     if xᵢ ≥ 0 for all i ∈ {1,...,n}
I_{ℝⁿ₊}(x) = +∞    otherwise
```

This represents the constraint that all components of x must be non-negative.

# Constructor
```julia
IndicatorNonnegativeOrthant()
```
No parameters needed - the constraint applies to any dimensional input.

# Properties
- **Smooth**: No, not differentiable at the boundary (xᵢ = 0)
- **Convex**: Yes, the nonnegative orthant is a convex cone
- **Proximal**: Yes, proximal operator is element-wise projection
- **Set Indicator**: Yes, represents the constraint set ℝⁿ₊

# Mathematical Properties
- **Constraint set**: ℝⁿ₊ = {x ∈ ℝⁿ : xᵢ ≥ 0 ∀i}
- **Geometric interpretation**: First orthant in ℝⁿ (all coordinates non-negative)
- **Proximal operator**: Element-wise projection: [prox_f(x)]ᵢ = max(xᵢ, 0)
- **Conjugate function**: Sum of negative parts: f*(y) = ∑ᵢ max(-yᵢ, 0)

# Geometric Interpretation
- **Orthant**: One of 2ⁿ orthants in ℝⁿ (the "positive" one)
- **Boundary**: Coordinate hyperplanes where xᵢ = 0
- **Interior**: Points where xᵢ > 0 for all i
- **Convex cone**: Closed under positive linear combinations

# Implementation Details
- **Dimension-agnostic**: Works with any input dimension
- **Tolerance-aware**: Uses FeasTolerance for numerical feasibility checks
- **Parallel support**: Implements parallel computation for large vectors
- **Memory efficient**: In-place operations, no temporary allocations

# Examples
```julia
# Create constraint function
f = IndicatorNonnegativeOrthant()

# Check feasibility
x_feasible = [1.0, 2.0, 0.0]  # All components ≥ 0
val = f(x_feasible)  # Returns 0.0

x_infeasible = [1.0, -1.0, 2.0]  # Contains negative component
val = f(x_infeasible)  # Returns +∞

# Project onto nonnegative orthant
x = [2.0, -3.0, 0.5, -1.0]
x_proj = proximalOracle(f, x)  # Returns [2.0, 0.0, 0.5, 0.0]

# In-place projection (more efficient)
x = [2.0, -3.0, 0.5, -1.0]
y = similar(x)
proximalOracle!(y, f, x)  # y = [2.0, 0.0, 0.5, 0.0]

# Works with any dimension
x_1d = [-5.0]
proj_1d = proximalOracle(f, x_1d)  # Returns [0.0]

x_high_dim = randn(1000)  # Random 1000-dimensional vector
proj_high = proximalOracle(f, x_high_dim)  # Projects all components

# Parallel computation for large vectors
x_large = randn(10000)
proj_parallel = proximalOracle(f, x_large, 1.0, true)  # enableParallel=true
```

# Algorithm Applications
- **Non-negativity constraints**: Enforce xᵢ ≥ 0 in optimization
- **Projected gradient descent**: Project gradient steps onto feasible region
- **ADMM**: Enforce non-negativity in alternating minimization
- **Support vector machines**: Non-negative dual variables
- **Non-negative matrix factorization**: Constrain factor matrices
- **Portfolio optimization**: Non-negative asset weights
- **Image processing**: Non-negative pixel intensities
- **Compressed sensing**: Sparse non-negative signal recovery

# Optimization Context
Commonly appears in constrained optimization:
```julia
minimize f(x)
subject to x ≥ 0  # Component-wise non-negativity
```

This constraint is often handled via:
1. **Projected methods**: Project iterates onto ℝⁿ₊
2. **Penalty methods**: Add barrier/penalty terms
3. **Primal-dual methods**: Use dual variables for constraints

# Projection Properties
The projection P_{ℝⁿ₊}(x) has several important properties:
- **Idempotent**: P(P(x)) = P(x)
- **Non-expansive**: ||P(x) - P(y)|| ≤ ||x - y||
- **Separable**: [P(x)]ᵢ depends only on xᵢ
- **Monotone**: xᵢ ≤ yᵢ ⟹ [P(x)]ᵢ ≤ [P(y)]ᵢ

# Performance Characteristics
- **Function evaluation**: O(n) complexity
- **Proximal operator**: O(n) complexity
- **Memory usage**: O(1) additional storage
- **Parallel scaling**: Excellent (independent components)

# Special Cases and Extensions
- **Box constraints**: Combine with upper bounds for IndicatorBox
- **Simplex constraints**: Add ∑xᵢ = 1 constraint
- **Cone constraints**: Extension to more general convex cones
- **Integer constraints**: Combine with integrality requirements

# Numerical Considerations
- **Tolerance handling**: Uses FeasTolerance for boundary points
- **Floating-point precision**: Robust to small numerical errors
- **Parallel threshold**: Uses parallelization for vectors > 1000 elements
- **Zero preservation**: Exactly preserves zero values (no numerical drift)

# Relationship to Other Constraints
- **IndicatorBox**: Generalization with both lower and upper bounds
- **IndicatorBallL2**: Different geometry (curved vs. polyhedral boundary)
- **IndicatorSOC**: Generalization to second-order cone constraints
- **Linear inequalities**: Subset of general linear inequality constraints

"""
struct IndicatorNonnegativeOrthant <: AbstractFunction 
    IndicatorNonnegativeOrthant() = new()
end 

isProximal(::Type{IndicatorNonnegativeOrthant}) = true 
isConvex(::Type{IndicatorNonnegativeOrthant}) = true 
isSet(::Type{IndicatorNonnegativeOrthant}) = true 

function (f::IndicatorNonnegativeOrthant)(x::NumericVariable, enableParallel::Bool=false)
    if any(x .< -FeasTolerance)
        return +Inf
    end
    return 0.0 
end 


function proximalOracle!(y::NumericVariable, f::IndicatorNonnegativeOrthant, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("IndicatorNonnegativeOrthant: proximal oracle does not support in-place operations for scalar inputs.")
    end
    if gamma < 0.0
        error("IndicatorNonnegativeOrthant: proximal oracle encountered gamma = $gamma < 0.")
    end

    @assert(size(x) == size(y), "IndicatorNonnegativeOrthant: input dimension mismatch.")

    if enableParallel && length(x) > 1000  # some threshold
        Threads.@threads for i in eachindex(x)
            @inbounds y[i] = max(x[i], 0.0)
        end
    else
        y .= max.(x, 0.0)
    end

end 

function proximalOracle(f::IndicatorNonnegativeOrthant, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        return max(x, 0.0)
    end 
    
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 
