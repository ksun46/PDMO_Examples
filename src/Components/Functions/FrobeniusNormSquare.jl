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

# Numerical Considerations
- **Condition number**: Well-conditioned when A'A is well-conditioned
- **Factorization choice**: Cholesky (faster) vs LU (more robust)
- **Regularization**: Add small diagonal term if A'A is singular
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


