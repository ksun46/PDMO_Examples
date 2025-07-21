import Arpack
"""
    LinearMappingMatrix(A::SparseMatrixCSC{Float64, Int64})

Constructs a linear mapping defined by a sparse matrix `A`.

The mapping is defined as:

    y = A * x

where `x` can be a vector or a matrix. If `x` is a matrix, the mapping is applied to each column.

# Fields
- `A`: A `SparseMatrixCSC{Float64, Int64}` representing the linear operator.
- `inputDim`: The number of columns of `A` (the dimension of input vectors).
- `outputDim`: The number of rows of `A` (the dimension of output vectors).
"""
struct LinearMappingMatrix <: AbstractMapping
    A::SparseMatrixCSC{Float64, Int64}
    inputDim::Int64
    outputDim::Int64

    function LinearMappingMatrix(A::SparseMatrixCSC{Float64, Int64})
        new(A, size(A, 2), size(A, 1))
    end
end

"""
    (L::LinearMappingMatrix)(x::AbstractVecOrMat{Float64}, ret::AbstractVecOrMat{Float64}, add::Bool = false)

Apply the matrix-based linear mapping to input `x` and store the result in `ret`.

# Arguments
- `x::AbstractVecOrMat{Float64}`: Input vector or matrix to which the mapping is applied.
- `ret::AbstractVecOrMat{Float64}`: Output vector or matrix to store the result.
- `add::Bool`: If `true`, adds the result to `ret` instead of overwriting it. Default is `false`.

# Implementation Details
The function computes `ret = L.A * x` (or adds to `ret` if `add` is `true`) using the efficient `mul!`
function for matrix-vector or matrix-matrix multiplication. 

For vectors, the input dimension must match `L.inputDim` and output dimension must match `L.outputDim`.
For matrices, the input must have `L.inputDim` rows, and the output must have `L.outputDim` rows and
the same number of columns as the input.

# Errors
- Throws an error if `x` is neither a vector nor a matrix of Float64.
- Throws an assertion error if the dimensions of inputs and outputs don't match requirements.
"""
function (L::LinearMappingMatrix)(x::AbstractVecOrMat{Float64}, ret::AbstractVecOrMat{Float64}, add::Bool = false)
    if isa(x, AbstractVector)
        @assert(length(x) == L.inputDim, "LinearMappingMatrix: input vector must have length $(L.inputDim).")
        @assert(length(ret) == L.outputDim, "LinearMappingMatrix: output vector must have length $(L.outputDim).")
        if add == false 
            mul!(ret, L.A, x)
        else 
            mul!(ret, L.A, x, 1.0, 1.0)
        end 
    elseif isa(x, AbstractMatrix)
        @assert(size(x, 1) == L.inputDim, "LinearMappingMatrix: input matrix must have $(L.inputDim) rows.")
        @assert(size(ret, 1) == L.outputDim && size(ret, 2) == size(x, 2), 
            "LinearMappingMatrix: output matrix must have size ($(L.outputDim), $(size(x,2))).")
        if add == false
            mul!(ret, L.A, x)
        else
            mul!(ret, L.A, x, 1.0, 1.0)
        end
    else
        error("LinearMappingMatrix: input must be a vector or a matrix of Float64.")
    end
end

"""
    (L::LinearMappingMatrix)(x::AbstractVecOrMat{Float64})

Apply the matrix-based linear mapping to input `x` and return the result as a new array.

# Arguments
- `x::AbstractVecOrMat{Float64}`: Input vector or matrix to which the mapping is applied.

# Returns
- A new vector or matrix containing the result of applying the mapping to `x`.

# Implementation Details
The function allocates an appropriate output array and delegates to the in-place version
of the operator. For vectors, the output has length `L.outputDim`. For matrices, the output
has `L.outputDim` rows and the same number of columns as the input.

# Errors
- Throws an error if `x` is neither a vector nor a matrix of Float64.
"""
function (L::LinearMappingMatrix)(x::AbstractVecOrMat{Float64})
    if isa(x, AbstractVector)
        ret = Vector{Float64}(undef, L.outputDim)
        L(x, ret)
        return ret
    elseif isa(x, AbstractMatrix)
        ret = Matrix{Float64}(undef, L.outputDim, size(x, 2))
        L(x, ret)
        return ret
    else
        error("LinearMappingMatrix: input must be a vector or a matrix of Float64.")
    end
end

"""
    adjoint!(L::LinearMappingMatrix, y::AbstractVecOrMat{Float64}, ret::AbstractVecOrMat{Float64}, add::Bool = false)

Apply the adjoint of the matrix-based linear mapping to input `y` and store the result in `ret`.

# Arguments
- `y::AbstractVecOrMat{Float64}`: Input vector or matrix to which the adjoint mapping is applied.
- `ret::AbstractVecOrMat{Float64}`: Output vector or matrix to store the result.
- `add::Bool`: If `true`, adds the result to `ret` instead of overwriting it. Default is `false`.

# Implementation Details
The function computes `ret = L.A' * y` (or adds to `ret` if `add` is `true`) using the efficient `mul!`
function for matrix-vector or matrix-matrix multiplication with the transposed matrix.

For vectors, the input dimension must match `L.outputDim` and output dimension must match `L.inputDim`.
For matrices, the input must have `L.outputDim` rows, and the output must have `L.inputDim` rows and
the same number of columns as the input.

# Errors
- Throws an error if `y` is neither a vector nor a matrix of Float64.
- Throws an assertion error if the dimensions of inputs and outputs don't match requirements.
"""
function adjoint!(L::LinearMappingMatrix, y::AbstractVecOrMat{Float64}, ret::AbstractVecOrMat{Float64}, add::Bool = false)
    if isa(y, AbstractVector)
        @assert(length(y) == L.outputDim, "LinearMappingMatrix: input vector must have length $(L.outputDim).")
        @assert(length(ret) == L.inputDim, "LinearMappingMatrix: output vector must have length $(L.inputDim).")
        if add == false 
            mul!(ret, L.A', y)
        else 
            mul!(ret, L.A', y, 1.0, 1.0)
        end 
    elseif isa(y, AbstractMatrix)
        @assert(size(y, 1) == L.outputDim, "LinearMappingMatrix: input matrix must have $(L.outputDim) rows.")
        @assert(size(ret, 1) == L.inputDim && size(ret, 2) == size(y, 2), 
            "LinearMappingMatrix: output matrix must have size ($(L.inputDim), $(size(y,2))).")
        if add == false
            mul!(ret, L.A', y)
        else
            mul!(ret, L.A', y, 1.0, 1.0)
        end
    else
        error("LinearMappingMatrix: input must be a vector or a matrix of Float64.")
    end
end

"""
    adjoint(L::LinearMappingMatrix, y::AbstractVecOrMat{Float64})

Apply the adjoint of the matrix-based linear mapping to input `y` and return the result as a new array.

# Arguments
- `y::AbstractVecOrMat{Float64}`: Input vector or matrix to which the adjoint mapping is applied.

# Returns
- A new vector or matrix containing the result of applying the adjoint mapping to `y`.

# Implementation Details
The function allocates an appropriate output array and delegates to the in-place version
of the adjoint operator. For vectors, the output has length `L.inputDim`. For matrices, the output
has `L.inputDim` rows and the same number of columns as the input.

# Errors
- Throws an error if `y` is neither a vector nor a matrix of Float64.
"""
function adjoint(L::LinearMappingMatrix, y::AbstractVecOrMat{Float64})
    if isa(y, AbstractVector)
        ret = Vector{Float64}(undef, L.inputDim)
        adjoint!(L, y, ret)
        return ret
    elseif isa(y, AbstractMatrix)
        ret = Matrix{Float64}(undef, L.inputDim, size(y, 2))
        adjoint!(L, y, ret)
        return ret
    else
        error("LinearMappingMatrix: input must be a vector or a matrix of Float64.")
    end
end 

"""
    createAdjointMapping(L::LinearMappingMatrix)

Create a new mapping that represents the adjoint of the given matrix-based linear mapping.

# Arguments
- `L::LinearMappingMatrix`: The mapping for which to create an adjoint.

# Returns
- A new `LinearMappingMatrix` containing the transposed matrix `L.A'`.

# Implementation Details
For a matrix-based linear mapping defined by matrix A, the adjoint mapping is defined
by the transposed matrix A'. This function creates a new mapping with the transposed matrix.
"""
function createAdjointMapping(L::LinearMappingMatrix)
    return LinearMappingMatrix(sparse(L.A'))
end 

"""
    operatorNorm2(L::LinearMappingMatrix)

Compute the operator norm (largest singular value) of the matrix-based linear mapping.

# Arguments
- `L::LinearMappingMatrix`: The mapping for which to compute the operator norm.

# Returns
- The operator norm of the matrix `L.A`. For small matrices, this is the exact value.
  For large matrices, this is a fast upper bound estimate.

# Implementation Details
Uses a hybrid approach based on matrix size:
- For small matrices (nnz < 10,000): Uses Arpack.svds for exact computation
- For large matrices: Uses sqrt(||A||_1 * ||A||_∞) as a fast, robust upper bound

This approach provides accuracy for small problems while avoiding convergence 
issues for large sparse matrices.
"""
function operatorNorm2(L::LinearMappingMatrix)
    # Use matrix size to determine approach
    # Threshold based on number of non-zeros to handle sparse matrices well
    nnz_threshold = 10000
    
    # Fast estimate computation (used as fallback or for large matrices)
    function fast_estimate()
        norm_1 = norm(L.A, 1)      # Maximum absolute column sum
        norm_inf = norm(L.A, Inf)  # Maximum absolute row sum
        return sqrt(norm_1 * norm_inf)
    end
    
    if nnz(L.A) < nnz_threshold
        # Small matrix: try exact SVD computation
        try
            result = Arpack.svds(L.A, nsv=1)
            svd_obj = result[1]
            return svd_obj.S[1]
        catch
            @warn "operatorNorm2: SVD computation failed for small matrix. Using fast estimate."
        end
    end 

    norm_1 = norm(L.A, 1)      # Maximum absolute column sum
    norm_inf = norm(L.A, Inf)  # Maximum absolute row sum
    return sqrt(norm_1 * norm_inf)
end 



