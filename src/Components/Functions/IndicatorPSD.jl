""" 
    IndicatorPSD(dim::Int64)

Represents the indicator function of the positive semidefinite cone.

The indicator function of the positive semidefinite cone is defined as:
```math
I_{\\mathcal{S}_+^n}(X) = \\begin{cases}
0 & \\text{if } X \\in \\mathcal{S}_+^n \\\\
\\infty & \\text{otherwise}
\\end{cases}
```

where ``\\mathcal{S}_+^n = \\{X \\in \\mathbb{R}^{n \\times n} : X = X^T, X \\succeq 0\\}`` is the cone of 
symmetric positive semidefinite matrices.

# Mathematical Properties
- **Convex**: The PSD cone is convex, making this a convex indicator function
- **Closed**: The PSD cone is closed, ensuring well-defined projections
- **Self-dual**: The PSD cone is self-dual under the trace inner product
- **Proximal**: The proximal operator corresponds to projection onto the PSD cone

# Arguments
- `dim::Int64`: The dimension n of the n×n matrix space (must be ≥ 1)

# Fields
- `dim::Int64`: Matrix dimension (n×n)
- `X::Matrix{Float64}`: Internal buffer for dense matrix computations

# Function Properties
- `isProximal(IndicatorPSD)`: `true` - admits efficient proximal operator
- `isConvex(IndicatorPSD)`: `true` - indicator of convex cone
- `isSet(IndicatorPSD)`: `true` - indicator function of a set

# Proximal Operator
The proximal operator (projection onto PSD cone) is computed via:
1. **Symmetrization**: Ensure input matrix is symmetric: ``\\bar{X} = (X + X^T)/2``
2. **Eigendecomposition**: Compute ``\\bar{X} = Q\\Lambda Q^T``
3. **Projection**: Set negative eigenvalues to zero: ``\\Lambda_+ = \\max(\\Lambda, 0)``
4. **Reconstruction**: Return ``\\text{proj}(X) = Q\\Lambda_+ Q^T``
"""
struct IndicatorPSD <: AbstractFunction 
    dim::Int64  # matrix dimension (n×n)
    X::Matrix{Float64} # buffer for dense matrix 
    function IndicatorPSD(dim::Int64)
        @assert dim >= 1 "Dimension must be at least 1"
        new(dim, zeros(dim, dim))
    end
end

isProximal(::Type{IndicatorPSD}) = true 
isConvex(::Type{IndicatorPSD}) = true 
isSet(::Type{IndicatorPSD}) = true 

"""
Check if matrix is approximately symmetric within tolerance
"""
function isApproxSymmetric(X::AbstractMatrix{Float64}, tol::Float64)
    size(X, 1) == size(X, 2) || return false
    for i in 1:size(X,1)
        for j in (i+1):size(X,2)
            # Check both (i,j) and (j,i) elements
            if abs(X[i,j] - X[j,i]) > tol
                return false
            end
        end
    end
    return true
end

"""
    (f::IndicatorPSD)(x::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, enableParallel::Bool=false)

Evaluate the indicator function for a matrix input (dense or sparse).
"""
function (f::IndicatorPSD)(x::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, enableParallel::Bool=false)
    @assert size(x) == (f.dim, f.dim) "IndicatorPSD: Input matrix must be dim × dim"
    
    # Use stricter tolerance for symmetry check
    if isApproxSymmetric(x, ZeroTolerance) == false # Stricter tolerance than FeasTolerance
        error("IndicatorPSD: Input matrix must be symmetric")
    end
    
    # Check positive semidefiniteness using eigenvalues
    f.X .= x
    eigvals = LinearAlgebra.eigvals(Symmetric(f.X))
    return minimum(eigvals) >= -FeasTolerance ? 0.0 : +Inf
end

"""
    proximalOracle!(y::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, 
                    f::IndicatorPSD, 
                    x::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, 
                    gamma::Float64 = 1.0, 
                    enableParallel::Bool=false)

Project onto the PSD cone. Works with both dense and sparse matrices.
The projection is performed by:
1. Symmetrizing the matrix
2. Computing eigendecomposition
3. Setting negative eigenvalues to zero
4. Reconstructing the matrix
"""
function proximalOracle!(y::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, 
                        f::IndicatorPSD, 
                        x::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, 
                        gamma::Float64 = 1.0, 
                        enableParallel::Bool=false)
    @assert size(x) == (f.dim, f.dim) "IndicatorPSD: Input matrix must be dim × dim"
    @assert size(y) == (f.dim, f.dim) "IndicatorPSD: Output matrix must be dim × dim"
    
    # Convert to dense and symmetrize
    f.X .= x #onvert to dense if sparse
    f.X .= (f.X .+ f.X') / 2
    
    # Eigendecomposition
    F = eigen(Symmetric(f.X))
    
    # Zero out negative eigenvalues
    λ = max.(F.values, 0)
    
    # Reconstruct matrix: V * Diagonal(λ) * V'
    result = F.vectors * Diagonal(λ) * F.vectors'
    
    # Ensure perfect symmetry in the result
    result = (result + result') / 2
    
    if isa(y, SparseMatrixCSC)
        # If output should be sparse, convert the result to sparse
        y .= sparse(result)
    else
        # If output should be dense, keep it dense
        y .= result
    end
end

"""
    proximalOracle(f::IndicatorPSD, 
                  x::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, 
                  gamma::Float64 = 1.0, 
                  enableParallel::Bool=false)

Non-mutating version of the proximal operator.
Returns same type as input (sparse or dense).
"""
function proximalOracle(f::IndicatorPSD, 
                       x::Union{Matrix{Float64}, SparseMatrixCSC{Float64}}, 
                       gamma::Float64 = 1.0, 
                       enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 