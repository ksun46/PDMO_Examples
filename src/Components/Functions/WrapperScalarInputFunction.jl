"""
    WrapperScalarInputFunction(originalFunction::AbstractFunction)

Wrapper that adapts scalar-input functions to work with vector interfaces.

This wrapper takes a function that operates on scalars and adapts it to work with
1-dimensional vectors, enabling integration with vector-based optimization algorithms.

# Arguments
- `originalFunction::AbstractFunction`: Function that operates on scalar inputs

# Properties
- **Smooth**: Inherits from original function
- **Convex**: Inherits from original function  
- **Proximal**: Inherits from original function
- **Set**: Inherits from original function

# Mathematical Properties
All properties are inherited from the original function, with input/output adaptation:
- Function evaluation: f([x]) = originalFunction(x)
- Gradient: ∇f([x]) = [∇originalFunction(x)]
- Proximal operator: prox_f([x]) = [prox_originalFunction(x)]

# Input/Output Format
- **Input**: Vector of length 1, e.g., [x]
- **Output**: Scalar for function evaluation, vector of length 1 for gradient/proximal

# Examples
```julia
# Adapt a scalar indicator function to vector interface
# Original function: f(x) = I_{[0,1]}(x) for scalar x
original_f = IndicatorBox(0.0, 1.0)  # Scalar version
vector_f = WrapperScalarInputFunction(original_f)

# Usage with vector input
x = [0.5]  # Vector of length 1
val = vector_f(x)  # Returns 0.0
grad = gradientOracle(vector_f, x)  # Returns [0.0] (if smooth)
prox = proximalOracle(vector_f, x)  # Returns [0.5] (projection)

# Invalid usage
x = [0.5, 1.0]  # Vector of length 2
# val = vector_f(x)  # ERROR: input must be vector of length 1
```

# Use Cases
- Adapting scalar functions for vector-based algorithms
- Interfacing with optimization solvers that expect vector inputs
- Building composite functions with mixed scalar/vector components
- Legacy code integration

# Limitations
- Only works with vector inputs of length 1
- Adds slight computational overhead due to wrapping
- Error checking is performed at runtime
- Original function must properly handle scalar inputs

# Implementation Notes
The wrapper performs the following adaptations:
1. Extracts scalar from length-1 vector input
2. Calls original function with scalar
3. Wraps scalar result back into vector format (for gradients/proximal)
4. Delegates all trait checks to the original function
"""
struct WrapperScalarInputFunction <: AbstractFunction 
    originalFunction::AbstractFunction
    function WrapperScalarInputFunction(f::AbstractFunction)
        new(f)
    end 
end 

# Delegate the traits to the original function. 
# Note: the traits checkers for WrapperScalarInputFunction check the instance instead of the type. 
isProximal(w::WrapperScalarInputFunction) = isProximal(typeof(w.originalFunction))
isSmooth(w::WrapperScalarInputFunction) = isSmooth(typeof(w.originalFunction))
isConvex(w::WrapperScalarInputFunction) = isConvex(typeof(w.originalFunction))
isSet(w::WrapperScalarInputFunction) = isSet(typeof(w.originalFunction))

function (f::WrapperScalarInputFunction)(x::NumericVariable, enableParallel::Bool=false)
    @assert(isa(x, Vector{Float64}) && length(x) == 1, "WrapperScalarInputFunction: input must be a vector of length 1.")
    return f.originalFunction(x[1], enableParallel)
end 

function gradientOracle!(y::NumericVariable, f::WrapperScalarInputFunction, x::NumericVariable, enableParallel::Bool=false)
    @assert(isa(x, Vector{Float64}) && length(x) == 1, "WrapperScalarInputFunction: gradientOracle! encountered non-vector input.")
    @assert(isa(y, Vector{Float64}) && length(y) == 1, "WrapperScalarInputFunction: gradientOracle! encountered non-vector output.")
    if isSmooth(typeof(f.originalFunction))
        y[1] = gradientOracle(f.originalFunction, x[1], enableParallel)
    else
        error("WrapperScalarInputFunction: original function is not smooth.")
    end
end 

function gradientOracle(f::WrapperScalarInputFunction, x::NumericVariable, enableParallel::Bool=false)
    y = Vector{Float64}(undef, 1)
    gradientOracle!(y, f, x, enableParallel)
    return y
end 


function proximalOracle!(y::NumericVariable, f::WrapperScalarInputFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)
    @assert(isa(x, Vector{Float64}) && length(x) == 1, "WrapperScalarInputFunction: proximalOracle! encountered non-vector input.")
    @assert(isa(y, Vector{Float64}) && length(y) == 1, "WrapperScalarInputFunction: proximalOracle! encountered non-vector output.")
    if isProximal(typeof(f.originalFunction))
        y[1] = proximalOracle(f.originalFunction, x[1], gamma, enableParallel)
    else
        error("WrapperScalarInputFunction: original function is not proximal.")
    end
end 

function proximalOracle(f::WrapperScalarInputFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)
    y = Vector{Float64}(undef, 1)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end  


