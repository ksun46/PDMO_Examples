"""
    LinearMappingExtraction(dim::Tuple, coe::Float64, indexStart::Int64, indexEnd::Int64)

Constructs a linear mapping that extracts a subset of the input array along the first dimension
and applies a scaling coefficient. The mapping is defined as:

    L(x) = coe * x[indexStart:indexEnd, :, ...]

where `dim` specifies the dimensions of the input array.

# Arguments
- `dim::Tuple`: The dimensions of the full input array.
- `coe::Float64`: The scaling coefficient.
- `indexStart::Int64`: The starting index for extraction along the first dimension.
- `indexEnd::Int64`: The ending index for extraction along the first dimension.

An assertion is made that `1 ≤ indexStart ≤ indexEnd ≤ dim[1]`.

For example, suppose we have the following constraint in the original problem:

    A1(x1) + A2(x2) + A3(x3) = b, 

where x1, x2, x3 are the variables, the pipeline will break the constriant into 
   
    A1(x1) - z1 = 0 
    A2(x2) - z2 = 0 
    A3(x3) - z3 = 0
    z in {[z1,z2,z3]: z1 + z2 + z3 = b}

Then this mapping extracts z_i from z, i.e., A_i(x_i) + L_i(z) = 0, where L_i(z) = -z_i 
"""
struct LinearMappingExtraction <: AbstractMapping 
    dim::Tuple 
    coe::Float64 
    indexStart::Int64 
    indexEnd::Int64

    function LinearMappingExtraction(dim::Tuple, coe::Float64, indexStart::Int64, indexEnd::Int64)
        @assert(length(dim) >= 1 && indexStart >= 1 && indexEnd <= dim[1] && indexStart <= indexEnd, "LinearMappingExtraction: indices out of range for the given dimension.")
        @assert(coe != 0.0, "LinearMappingExtraction: scaling coefficient must be non-zero.")
        new(dim, coe, indexStart, indexEnd)
    end
end

"""
    (L::LinearMappingExtraction)(x::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the extraction mapping to input `x` and store the result in `ret`.

# Arguments
- `x::NumericVariable`: Input array with dimensions matching `L.dim`.
- `ret::NumericVariable`: Output array to store the result, with dimensions `(L.indexEnd - L.indexStart + 1, size(x)[2:end]...)`.
- `add::Bool`: If `true`, adds the result to `ret` instead of overwriting it. Default is `false`.

# Implementation Details
The function extracts the slice `x[indexStart:indexEnd, :, ...]`, multiplies by the coefficient `coe`,
and either assigns to or adds to `ret` depending on the `add` parameter.
"""
function (L::LinearMappingExtraction)(x::NumericVariable, ret::NumericVariable, add::Bool = false)
    @assert(size(x) == L.dim, "LinearMappingExtraction: input dimensions do not match the specified dim.")
    # The expected shape for the output: (indexEnd-indexStart+1, size(x)[2:end]...)
    expectedShape = (L.indexEnd - L.indexStart + 1, Base.tail(size(x))...)
    @assert(size(ret) == expectedShape, "LinearMappingExtraction: output array has incorrect dimensions.")
    
    # Handle vector case separately to avoid tuple indexing issues with sparse vectors
    if length(L.dim) == 1
        if add == false
            if L.coe == 1.0
                ret .= x[L.indexStart:L.indexEnd]
            else
                ret .= L.coe .* x[L.indexStart:L.indexEnd]
            end
        else
            if L.coe == 1.0
                ret .+= x[L.indexStart:L.indexEnd]
            else
                ret .+= L.coe .* x[L.indexStart:L.indexEnd]
            end
        end
    else
        # For multi-dimensional arrays, use tuple indexing
        slice = (L.indexStart:L.indexEnd, ntuple(_ -> Colon(), ndims(x)-1)...)
        if add == false 
            if L.coe == 1.0
                ret .= x[slice]
            else
                ret .= L.coe .* x[slice]
            end
        else 
            if L.coe == 1.0
                ret .+= x[slice]
            else
                ret .+= L.coe .* x[slice]
            end
        end
    end
end

"""
    (L::LinearMappingExtraction)(x::AbstractArray{Float64})

Apply the extraction mapping to input `x` and return the result as a new array.

# Arguments
- `x::AbstractArray{Float64}`: Input array with dimensions matching `L.dim`.

# Returns
- An array containing the extracted and scaled slice of `x`.

# Implementation Details
The function extracts the slice `x[indexStart:indexEnd, :, ...]`, multiplies by the coefficient `coe`,
and returns the result as a new array.
"""
function (L::LinearMappingExtraction)(x::AbstractArray{Float64})
    @assert(size(x) == L.dim, "LinearMappingExtraction: input dimensions do not match the specified dim.")
    if length(L.dim) == 1
        if L.coe == 1.0
            return x[L.indexStart:L.indexEnd]
        else
            return L.coe .* x[L.indexStart:L.indexEnd]
        end
    else
        slice = (L.indexStart:L.indexEnd, ntuple(_ -> Colon(), ndims(x)-1)...)
        if L.coe == 1.0
            return x[slice]
        else
            return L.coe .* x[slice]
        end
    end
end

"""
    adjoint!(L::LinearMappingExtraction, y::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the adjoint of the extraction mapping to input `y` and store the result in `ret`.

# Arguments
- `y::NumericVariable`: Input array with dimensions matching the extraction slice.
- `ret::NumericVariable`: Output array to store the result, with dimensions matching `L.dim`.
- `add::Bool`: If `true`, adds the result to `ret` instead of overwriting it. Default is `false`.

# Implementation Details
For the extraction mapping, the adjoint is an embedding operator: it takes an array of the extracted shape,
scales it by `coe`, and places it into an array of the full input dimensions (with zeros elsewhere).
If `add` is `false`, the output array is first zeroed before the embedding.
"""
function adjoint!(L::LinearMappingExtraction, y::NumericVariable, ret::NumericVariable, add::Bool = false)
    @assert(size(ret) == L.dim, "LinearMappingExtraction (adjoint!): output array must have dimensions equal to dim.")
    # The expected shape for y must be the same as the extracted slice.
    expected_shape = (L.indexEnd - L.indexStart + 1, Base.tail(L.dim)...)
    @assert(size(y) == expected_shape, "LinearMappingExtraction (adjoint!): input array y has incorrect dimensions.")
    
    if length(L.dim) == 1
        if add == false
            ret .= 0.0
            if L.coe == 1.0
                ret[L.indexStart:L.indexEnd] .= y
            else
                ret[L.indexStart:L.indexEnd] .= L.coe .* y
            end
        else
            if L.coe == 1.0
                ret[L.indexStart:L.indexEnd] .+= y
            else
                ret[L.indexStart:L.indexEnd] .+= L.coe .* y
            end
        end
    else
        if add == false 
            ret .= 0.0
            slice = (L.indexStart:L.indexEnd, ntuple(_ -> Colon(), length(L.dim)-1)...)
            if L.coe == 1.0
                ret[slice] .= y
            else
                ret[slice] .= L.coe .* y
            end
        else 
            slice = (L.indexStart:L.indexEnd, ntuple(_ -> Colon(), length(L.dim)-1)...)
            if L.coe == 1.0
                ret[slice] .+= y
            else
                ret[slice] .+= L.coe .* y
            end
        end
    end
end

"""
    adjoint(L::LinearMappingExtraction, y::NumericVariable)

Apply the adjoint of the extraction mapping to input `y` and return the result as a new array.

# Arguments
- `y::NumericVariable`: Input array with dimensions matching the extraction slice.

# Returns
- An array with dimensions matching `L.dim` containing the embedded and scaled input,
  with zeros elsewhere.

# Implementation Details
Allocates a new array of zeros with dimensions `L.dim` and delegates to `adjoint!` for the actual computation.
Uses sparse arrays for dimensions ≤ 2 to optimize memory usage.
"""
function adjoint(L::LinearMappingExtraction, y::NumericVariable)
    ret = length(L.dim) <= 2 ? spzeros(L.dim) : zeros(L.dim)
    adjoint!(L, y, ret)
    return ret
end

"""
    operatorNorm2(L::LinearMappingExtraction)

Compute the operator norm (largest singular value) of the extraction mapping.

# Returns
- The absolute value of the scaling coefficient `coe`.

# Implementation Details
For an extraction mapping, the operator norm is simply the absolute value of the scaling coefficient,
since the mapping only scales the extracted values.
"""
function operatorNorm2(L::LinearMappingExtraction)
    return L.coe < 0.0 ? -L.coe : L.coe 
end 
