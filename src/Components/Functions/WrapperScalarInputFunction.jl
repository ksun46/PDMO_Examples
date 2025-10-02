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
isProximal(w::WrapperScalarInputFunction) = isProximal(w.originalFunction)
isSmooth(w::WrapperScalarInputFunction) = isSmooth(w.originalFunction)
isConvex(w::WrapperScalarInputFunction) = isConvex(w.originalFunction)
isSet(w::WrapperScalarInputFunction) = isSet(w.originalFunction)
isSupportedByJuMP(w::WrapperScalarInputFunction) = isSupportedByJuMP(w.originalFunction)

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

# JuMP support
function JuMPAddProximableFunction(g::WrapperScalarInputFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    # WrapperScalarInputFunction only works with dimension 1
    @assert length(var) == 1 "WrapperScalarInputFunction: dimension must be 1, got $(length(var))"
    
    # Check if original function supports JuMP
    if isSupportedByJuMP(g.originalFunction) == false
        error("WrapperScalarInputFunction: original function $(typeof(g.originalFunction)) does not support JuMP")
    end
    
    # Check if original function is proximable (constraint-related)
    if isProximal(g.originalFunction) == false
        error("WrapperScalarInputFunction: JuMPAddProximableFunction called but original function is not proximal")
    end
    
    # Delegate to original function's JuMP implementation
    return JuMPAddProximableFunction(g.originalFunction, model, var)
end

function JuMPAddSmoothFunction(f::WrapperScalarInputFunction, model::JuMP.Model, 
    var::Vector{<:JuMP.VariableRef})
    
    # WrapperScalarInputFunction only works with dimension 1
    @assert length(var) == 1 "WrapperScalarInputFunction: variable dimension must be 1, got $(length(var))"
    
    # Check if original function supports JuMP
    if isSupportedByJuMP(f.originalFunction) == false
        error("WrapperScalarInputFunction: original function $(typeof(f.originalFunction)) does not support JuMP")
    end
    
    # Check if original function is smooth (objective-related)
    if isSmooth(f.originalFunction) == false
        error("WrapperScalarInputFunction: JuMPAddSmoothFunction called but original function is not smooth")
    end

    # Delegate to original function's JuMP implementation
    return JuMPAddSmoothFunction(f.originalFunction, model, var)
end

