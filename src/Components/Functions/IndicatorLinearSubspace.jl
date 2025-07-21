"""
    IndicatorLinearSubspace(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64})

Represents the indicator function of a linear subspace defined by the equation Ax = b.

The indicator function is defined as:
```math
I_{Ax=b}(x) = \\begin{cases}
0 & \\text{if } Ax = b \\\\
\\infty & \\text{otherwise}
\\end{cases}
```

# Arguments
- `A::SparseMatrixCSC{Float64, Int64}`: The constraint matrix
- `b::Vector{Float64}`: The right-hand side vector

# Fields
- `A`: The constraint matrix
- `b`: The right-hand side vector
- `U`: Left singular vectors from SVD decomposition
- `S`: Singular values from SVD decomposition
- `V`: Right singular vectors from SVD decomposition
- `rank`: Numerical rank of matrix A
- `isFullRank`: Boolean indicating if A has full row rank
- `projectionMatrix`: Pre-computed matrix for projection:
  * If full rank: stores (AA')^{-1}
  * If rank deficient: stores A^+ (pseudoinverse)

# Notes
- SVD decomposition and projection matrices are computed once during initialization for efficiency
- The proximal operator (projection) is computed differently based on whether A has full row rank:
  * For full rank: y = x - A'(AA')^{-1}(Ax - b)
  * For rank deficient: y = x - A^+(Ax - b), where A^+ is the pseudoinverse
- Numerical rank is determined using a tolerance based on machine epsilon

# Example
```julia
A = sparse([1.0 2.0; 3.0 4.0])
b = [1.0, 2.0]
f = IndicatorLinearSubspace(A, b)
x = [0.0, 0.0]
proj = proximalOracle(f, x)  # Projects x onto the subspace Ax = b
```
"""
struct IndicatorLinearSubspace <: AbstractFunction 
    A::SparseMatrixCSC{Float64, Int64}
    b::Vector{Float64}
    U::Matrix{Float64}  
    S::Vector{Float64}  
    V::Matrix{Float64}  
    rank::Int64        
    isFullRank::Bool   
    projectionMatrix::Matrix{Float64}  
    residual::Vector{Float64}  
    temp::Vector{Float64}     

    function IndicatorLinearSubspace(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64})
        m, n = size(A)
        @assert(m == length(b), "IndicatorLinearSubspace: Dimension mismatch.")
        @assert(m > 0 && n > 0, "IndicatorLinearSubspace: Empty matrix not allowed.")
        @assert(m <= n, "IndicatorLinearSubspace: More columns than rows.")

        # Compute SVD once during initialization
        F = svd(Matrix(A))
        tol = eps(Float64) * maximum(F.S) * sqrt(max(m,n))  # Standard numerical tolerance
        rank = sum(F.S .> tol)
        isFullRank = (rank == m)
        
        # Pre-compute projection matrix based on rank
        if isFullRank
            # For full rank case: compute (AA')^{-1}
            projectionMatrix = F.U[:, 1:m] * Diagonal(1.0 ./ F.S[1:m].^2) * F.U[:, 1:m]'
        else
            # For rank deficient case: compute full pseudoinverse
            Sinv = zeros(length(F.S))
            @inbounds for i in 1:length(F.S)
                Sinv[i] = F.S[i] > tol ? 1.0/F.S[i] : 0.0
            end
            projectionMatrix = F.V * Diagonal(Sinv) * F.U'
        end
        
        # Initialize buffers
        residual = zeros(m)
        temp = zeros(m)  # Always m-dimensional for both cases
        
        new(A, b, F.U, F.S, F.V, rank, isFullRank, projectionMatrix, residual, temp)
    end
end

# Override traits for IndicatorLinearSubspace
isProximal(::Type{IndicatorLinearSubspace}) = true 
isConvex(::Type{IndicatorLinearSubspace}) = true 
isSet(::Type{IndicatorLinearSubspace}) = true 

# function value
function (f::IndicatorLinearSubspace)(x::NumericVariable, enableParallel::Bool=false)
    @assert(size(f.A, 2) == length(x), "IndicatorLinearSubspace: input dimension mismatch.")
    
    mul!(f.residual, f.A, x)
    f.residual .-= f.b
    return norm(f.residual, 2) <= FeasTolerance ? 0.0 : Inf  # Explicitly use 2-norm
end 

# proximal oracle
function proximalOracle!(y::NumericVariable, f::IndicatorLinearSubspace, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("IndicatorLinearSubspace: proximal oracle does not support in-place operations for scalar inputs.")
    end
    if gamma < 0.0
        error("IndicatorLinearSubspace: proximal oracle encountered gamma = $gamma < 0.")
    end
    @assert(length(x) == size(f.A, 2) == length(y), "IndicatorLinearSubspace: input dimension mismatch.")

    # Compute residual: Ax - b
    mul!(f.residual, f.A, x)
    f.residual .-= f.b
    
    if f.isFullRank  # Full row rank case
        # y = x - A'(AA')^{-1}(Ax - b)
        mul!(f.temp, f.projectionMatrix, f.residual)
        mul!(y, f.A', f.temp, -1.0, 0.0)
        y .+= x
    else  # Rank deficient case
        # y = x - A^+(Ax - b)
        mul!(f.temp, f.projectionMatrix, f.residual)
        copyto!(y, x)
        y .-= f.temp
    end
end

function proximalOracle(f::IndicatorLinearSubspace, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false) 
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

function testIndicatorLinearSubspace(m, n)
    for _ in 1:10 
        myA = sparse(randn(m, n))
        b = randn(m)
        f = IndicatorLinearSubspace(myA, b)
        x = randn(n)
        y = proximalOracle(f, x)
        @assert(norm(myA * y - b, 2) < 1e-6, "IndicatorLinearSubspace: proximal oracle failed.")
    end
end
