""" 
    FrobeniusNormSquare(A::Matrix{Float64}, b::Union{Matrix{Float64}, Vector{Float64}}, numberRows::Int64, numberColumns::Int64, coe::Float64=0.5)

Represents the squared Frobenius norm of the residual AX - b.

# Mathematical Definition
```
f(X) = coe * ||AX - b||²_F
```

where:
- A is an L × m coefficient matrix
- X is an m × n variable matrix (or vector if n=1)
- b is an L × n target matrix (or L-dimensional vector if n=1)
- ||·||_F denotes the Frobenius norm: ||M||²_F = ∑ᵢⱼ M²ᵢⱼ = trace(M'M)

# Arguments
- `A::Matrix{Float64}`: Coefficient matrix (L × m)
- `b::Union{Matrix{Float64}, Vector{Float64}}`: Target matrix (L × n) or vector (L)
- `numberRows::Int64`: Number of rows in X (m)
- `numberColumns::Int64`: Number of columns in X (n), set to 1 if b is a vector
- `coe::Float64=0.5`: Positive scaling coefficient

# Properties
- **Smooth**: Yes, quadratic functions are infinitely differentiable
- **Convex**: Yes, squared Frobenius norm is convex
- **Proximal**: Yes, has explicit proximal operator via linear system solution

# Mathematical Properties
- **Gradient**: ∇f(X) = 2 * coe * A'(AX - b)
- **Proximal Operator**: Solution to (I + 2γ*coe*A'A)Y = X + 2γ*coe*A'b

# Internal Structure
The struct pre-computes and caches:
- `ATransA = A'A`: Gram matrix for efficient repeated computations
- Factorization of the proximal system matrix for fast linear solves
- Buffer arrays to minimize memory allocations

# Implementation Details
- **Automatic problem detection**: Handles both vector (n=1) and matrix (n>1) cases
- **Efficient factorization**: Uses Cholesky when possible, falls back to LU
- **Factorization caching**: Reuses factorization when γ parameter doesn't change
- **Memory management**: Pre-allocated buffers for all intermediate computations

# Examples
```julia
# Least squares problem: minimize ||Ax - b||²
A = rand(10, 5)  # 10 constraints, 5 variables
b = rand(10)     # Target vector
f = FrobeniusNormSquare(A, b, 5, 1, 0.5)  # Note: coe=0.5 gives (1/2)||Ax-b||²
x = rand(5)
val = f(x)  # Function value
grad = gradientOracle(f, x)  # Gradient: A'(Ax - b)

# Matrix least squares: minimize ||AX - B||²_F
A = rand(10, 5)    # Linear operator
B = rand(10, 3)    # Target matrix
f = FrobeniusNormSquare(A, B, 5, 3, 1.0)
X = rand(5, 3)
val = f(X)  # Function value
grad = gradientOracle(f, X)  # Gradient matrix

# Proximal operator (useful in optimization algorithms)
f = FrobeniusNormSquare(A, b, 5, 1, 1.0)
x_current = rand(5)
γ = 0.1  # Proximal parameter
x_prox = proximalOracle(f, x_current, γ)  # Proximal step
```

# Algorithm Applications
- **Least squares regression**: Direct formulation of ||Ax - b||²
- **Ridge regression**: Add L2 regularization
- **Matrix completion**: Frobenius norm data fitting term
- **Image denoising**: Data fidelity term in variational methods
- **System identification**: Parameter estimation problems
- **Proximal gradient methods**: Smooth term in composite optimization
- **ADMM**: Quadratic penalty terms

# Optimization Context
This function commonly appears in:
```julia
# Composite optimization problem
minimize f(x) + g(x)
# where f(x) = (1/2)||Ax - b||² (smooth)
# and g(x) is some regularizer (possibly non-smooth)
```

Algorithms like proximal gradient descent alternate between:
1. Gradient step on f: x̃ = x - α∇f(x)
2. Proximal step on g: x⁺ = prox_g(x̃)

# Performance Characteristics
- **Function evaluation**: O(mn + Ln) operations
- **Gradient computation**: O(mn + Ln) operations  
- **Proximal operator**: O(m³) for factorization + O(m²n) for solve
- **Memory usage**: O(m²) for factorization + O(mn + Ln) for buffers

# Numerical Considerations
- **Condition number**: Well-conditioned when A'A is well-conditioned
- **Factorization choice**: Cholesky (faster) vs LU (more robust)
- **Scaling**: Consider rescaling A and b for numerical stability
- **Caching**: Factorization is cached and reused when γ doesn't change
"""
mutable struct FrobeniusNormSquare <: AbstractFunction
    A::Matrix{Float64}
    b::Union{Matrix{Float64}, Vector{Float64}}
    ATransA::Matrix{Float64}
    numberRows::Int64
    numberColumns::Int64
    coe::Float64
    bufferResidual::Union{Matrix{Float64}, Vector{Float64}}
    bufferSystem::Matrix{Float64}
    bufferRHS::Union{Matrix{Float64}, Vector{Float64}}
    factorization::Any  # Store factorization for efficient proximal operator
    last_coe::Float64
    isVectorProblem::Bool

    function FrobeniusNormSquare(A::Matrix{Float64}, b::Union{Matrix{Float64}, Vector{Float64}}, numberRows::Int64, numberColumns::Int64, coe::Float64=0.5)
        if coe <= 0.0
            error("FrobeniusNormSquare: coe must be positive.")
        end
        
        isVectorProblem = isa(b, Vector{Float64})
        
        if isVectorProblem
            if size(A, 1) != length(b)
                error("FrobeniusNormSquare: A rows must match b length when b is a vector.")
            end
            if numberColumns != 1
                error("FrobeniusNormSquare: numberColumns must be 1 when b is a vector.")
            end
        else
            if size(A, 1) != size(b, 1)
                error("FrobeniusNormSquare: A and b must have the same row size.")
            end
            if size(b, 2) != numberColumns
                error("FrobeniusNormSquare: size(b, 2) != numberColumns.")
            end
        end
        
        if size(A, 2) != numberRows
            error("FrobeniusNormSquare: size(A, 2) != numberRows")
        end

        ATransA = A' * A
        
        if isVectorProblem
            bufferResidual = zeros(length(b))
            bufferRHS = zeros(numberRows)
        else
            bufferResidual = zeros(size(b))
            bufferRHS = zeros(numberRows, numberColumns)
        end
        
        bufferSystem = zero(ATransA)
        
        # Initialize without factorization - we'll compute it in the first proximal call
        return new(A, b, ATransA, numberRows, numberColumns, coe, bufferResidual, bufferSystem, bufferRHS, nothing, -1.0, isVectorProblem)
    end
end 

isSmooth(::Type{FrobeniusNormSquare}) = true 
isProximal(::Type{FrobeniusNormSquare}) = true 
isConvex(::Type{FrobeniusNormSquare}) = true 

function (f::FrobeniusNormSquare)(x::NumericVariable, enableParallel::Bool=false)
    if f.isVectorProblem
        @assert(length(x) == f.numberRows, "FrobeniusNormSquare: input dimension mismatch.")
        f.bufferResidual .= f.A * x .- f.b
    else
        @assert(size(x) == (f.numberRows, f.numberColumns), "FrobeniusNormSquare: input dimension mismatch.")
        f.bufferResidual .= f.A * x .- f.b
    end
    return f.coe * dot(f.bufferResidual, f.bufferResidual)
end

function gradientOracle!(y::NumericVariable, f::FrobeniusNormSquare, x::NumericVariable, enableParallel::Bool=false)
    if f.isVectorProblem
        @assert(length(x) == f.numberRows, "FrobeniusNormSquare: input dimension mismatch.")
        @assert(length(y) == f.numberRows, "FrobeniusNormSquare: output dimension mismatch.")
        
        # Compute Ax - b
        f.bufferResidual .= f.A * x .- f.b
        
        # y = 2 * coe * A'(Ax - b)
        y .= 2 * f.coe * (f.A' * f.bufferResidual)
    else
        @assert(size(x) == (f.numberRows, f.numberColumns), "FrobeniusNormSquare: input dimension mismatch.")
        @assert(size(y) == (f.numberRows, f.numberColumns), "FrobeniusNormSquare: output dimension mismatch.")
        
        # Compute Ax - b
        f.bufferResidual .= f.A * x .- f.b
        
        # y = 2 * coe * A'(Ax - b)
        mul!(y, f.A', f.bufferResidual)
        y .*= 2 * f.coe
    end
end

function gradientOracle(f::FrobeniusNormSquare, x::NumericVariable, enableParallel::Bool=false)
    if f.isVectorProblem
        y = zeros(f.numberRows)
    else
        y = zeros(f.numberRows, f.numberColumns)
    end
    gradientOracle!(y, f, x, enableParallel)
    return y
end

function proximalOracle!(y::NumericVariable, f::FrobeniusNormSquare, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    @assert(gamma > 0.0, "FrobeniusNormSquare: gamma must be positive.")
    
    if f.isVectorProblem
        @assert(length(x) == f.numberRows, "FrobeniusNormSquare: input dimension mismatch.")
        @assert(length(y) == f.numberRows, "FrobeniusNormSquare: output dimension mismatch.")
    else
        @assert(size(x) == (f.numberRows, f.numberColumns), "FrobeniusNormSquare: input dimension mismatch.")
        @assert(size(y) == (f.numberRows, f.numberColumns), "FrobeniusNormSquare: output dimension mismatch.")
    end
    
    coe = 2 * gamma * f.coe 
    
    # Set up linear system: (I + 2γcoe * A'A)y = x + 2γcoe * A'b
    # We only need to recompute the factorization if gamma changes
    if f.factorization === nothing || f.last_coe != coe
        f.bufferSystem .= coe .* f.ATransA
        for i in 1:f.numberRows
            f.bufferSystem[i,i] += 1.0 
        end
        f.factorization = try
            cholesky(f.bufferSystem)
        catch
            factorize(f.bufferSystem)
        end
        f.last_coe = coe
    end
    
    # Compute right-hand side: x + 2γcoe * A'b
    f.bufferRHS .= x .+ coe .* (f.A' * f.b)
    
    # Solve the system
    ldiv!(y, f.factorization, f.bufferRHS)
end 

function proximalOracle(f::FrobeniusNormSquare, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if f.isVectorProblem
        y = zeros(f.numberRows)
    else
        y = zeros(f.numberRows, f.numberColumns)
    end
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end


