"""
    IndicatorSumOfNVariables

Represents the indicator function for the constraint

    x₁ + x₂ + … + xₙ = b

where b (`rhs`) is of type `NumericVariable` (i.e. a scalar, vector, matrix, or tensor)
and the variable `x` is assumed to be a concatenation of `numberVariables` blocks,
each having the same number of elements (and shape when reshaped) as `rhs`.
"""

struct IndicatorSumOfNVariables <: AbstractFunction 
    numberVariables::Int64 
    rhs::NumericVariable

    IndicatorSumOfNVariables(numberVariables::Int64, rhs::NumericVariable) = new(numberVariables, rhs)
end 

# Override traits for IndicatorSumOfNVariables
isProximal(::Type{IndicatorSumOfNVariables}) = true 
isConvex(::Type{IndicatorSumOfNVariables}) = true
isSet(::Type{IndicatorSumOfNVariables}) = true
isSupportedByJuMP(f::Type{<:IndicatorSumOfNVariables}) = true 


"""
    (f::IndicatorSumOfNVariables)(x::NumericVariable, enableParallel::Bool=false) -> Float64

Evaluates the indicator function for the constraint

    x₁ + x₂ + … + xₙ = rhs

If `rhs` is a scalar, then `x` is expected to be a vector of length `numberVariables`
and the constraint is sum(x) ≈ rhs (within `FeasTolerance`).
If `rhs` is not a scalar, then `x` is assumed to be an array whose first dimension is
size(rhs, 1) * numberVariables and the remaining dimensions match `rhs`.
In that case, the function reshapes each block (along the first dimension) to the shape of `rhs`,
sums them elementwise, and compares the result with `rhs`.
Returns 0.0 if the constraint is satisfied (within tolerance) and `Inf` otherwise.
"""
function (f::IndicatorSumOfNVariables)(x::NumericVariable, enableParallel::Bool=false)
    if isa(f.rhs, Number)
        # Scalar case: x is a vector of length numberVariables.
        if length(x) != f.numberVariables
            error("IndicatorSumOfNVariables: Expected x to have length $(f.numberVariables), got length $(length(x)).")
        end
        res = sum(x) - f.rhs
        return abs(res) < FeasTolerance ? 0.0 : Inf
    else
        # Non-scalar case: x is an array with first dimension equal to size(rhs,1) * numberVariables.
        dims_rhs = size(f.rhs)
        if size(x, 1) != dims_rhs[1] * f.numberVariables
            error("IndicatorSumOfNVariables: Expected x to have first dimension $(dims_rhs[1] * f.numberVariables), got $(size(x, 1)).")
        end
        # Check remaining dimensions (if any)
        for d in 2:length(dims_rhs)
            if size(x, d) != dims_rhs[d]
                error("IndicatorSumOfNVariables: Expected dimension $d of x to equal $(dims_rhs[d]), got $(size(x, d)).")
            end
        end
        res = -f.rhs 
        for k in 1:f.numberVariables
            idx_range = ((k-1)*dims_rhs[1] + 1):(k*dims_rhs[1])
            # Use colon for remaining dimensions.
            block = view(x, idx_range, ntuple(i -> Colon(), ndims(x)-1)...)
            res .+= block
        end
        return sum(abs, res) < FeasTolerance ? 0.0 : Inf
    end
end

"""
    proximalOracle!(y::NumericVariable, f::IndicatorSumOfNVariables, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)

Computes the proximal operator (i.e. the projection) of the indicator function in-place,
storing the result in `y`.

For the scalar case (when `rhs` is a Number), it is assumed that `x` is a vector of length
`numberVariables`, and the projection onto { x : sum(x) = rhs } subtracts the uniform shift

    shift = (sum(x) - rhs) / numberVariables

from each entry.
For the non-scalar case, `x` is assumed to be an array whose first dimension is
size(rhs, 1) * numberVariables and remaining dimensions match `rhs`.
The function computes the elementwise residual

    res = (sum of blocks) - rhs

and then computes

    shift = res / numberVariables

which is subtracted from every block.
No additional dimension checking is performed for performance reasons.
"""
function proximalOracle!(y::NumericVariable, f::IndicatorSumOfNVariables, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if isa(f.rhs, Number)
        error("IndicatorSumOfNVariables: gradient oracle does not support in-place operations for scalar inputs.")
    end
        
    dims_rhs = size(f.rhs)
    nblock = f.numberVariables

    if size(x, 1) != dims_rhs[1] * nblock
        error("IndicatorSumOfNVariables: Expected x to have first dimension $(dims_rhs[1] * nblock), got $(size(x, 1)).")
    end
    
    for d in 2:length(dims_rhs)
        if size(x, d) != dims_rhs[d]
            error("IndicatorSumOfNVariables: Expected dimension $d of x to equal $(dims_rhs[d]), got $(size(x, d)).")
        end
    end
    
    # Compute the residual: res = (sum over blocks) - rhs.
    res = -f.rhs
    for k in 1:nblock
        idx_range = ((k-1)*dims_rhs[1] + 1):(k*dims_rhs[1])
        block = view(x, idx_range, ntuple(i -> Colon(), ndims(x)-1)...)
        res .+= block
    end

    # compute the shift
    shift = res ./ f.numberVariables
    
    # Apply the projection: subtract shift from each block.
    for k in 1:nblock
        idx_range = ((k-1)*dims_rhs[1] + 1):(k*dims_rhs[1])
        block_x = view(x, idx_range, ntuple(i -> Colon(), ndims(x)-1)...)
        block_y = view(y, idx_range, ntuple(i -> Colon(), ndims(x)-1)...)
        copy!(block_y, block_x .- shift)
    end
end

"""
    proximalOracle(f::IndicatorSumOfNVariables, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false) -> NumericVariable

Computes and returns the proximal operator (projection) of the indicator function,
i.e. the projection of `x` onto the set

    x₁ + x₂ + … + xₙ = rhs.
"""
function proximalOracle(f::IndicatorSumOfNVariables, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if isa(f.rhs, Number)
        if length(x) != f.numberVariables
            error("IndicatorSumOfNVariables: Expected x to have length $(f.numberVariables), got length $(length(x)).")
        end

        res = sum(x) - f.rhs
        shift = res / f.numberVariables

        y = similar(x)
        for i in eachindex(x)
            y[i] = x[i] - shift
        end
        return y
    else        
        y = similar(x)
        proximalOracle!(y, f, x, gamma, enableParallel)
        return y
    end
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorSumOfNVariables, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    # Create unconstrained variables
    dim = length(var)
    
    subvectorDim = length(g.rhs)
    numberVariables = g.numberVariables
    @assert dim == subvectorDim * numberVariables "JuMPAddProximableFunction of $(typeof(g)): Dimension must equal length(rhs) * numberVariables"
        
    # Add constraints: sum over blocks for each element
    JuMP.@constraint(model, [k in 1:subvectorDim],
        sum(var[(idx-1) * subvectorDim + k] for idx in 1:numberVariables) == g.rhs[k])
    
    return nothing  # Sum of N variables constraints don't contribute to objective
end