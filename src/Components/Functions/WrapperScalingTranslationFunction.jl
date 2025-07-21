"""
    WrapperScalingTranslationFunction

A wrapper that represents a transformed function of the form f(x) = g(coe·x + translation),
where g is the original function, coe is a positive scaling coefficient, and translation
is an additive shift.

This transformation is useful for:
- Scaling and shifting existing functions
- Implementing variable transformations in optimization
- Creating scaled versions of constraint functions
- Handling affine transformations of variables

# Fields
- `originalFunction::AbstractFunction`: The original function g
- `coe::Float64`: The scaling coefficient (must be positive)
- `translation::NumericVariable`: The translation vector/scalar
- `buffer::NumericVariable`: Internal buffer for computations

# Mathematical Properties
For f(x) = g(coe·x + translation):
- Function evaluation: f(x) = g(coe·x + translation)
- Gradient (if g is smooth): ∇f(x) = coe · ∇g(coe·x + translation)
- Proximal operator: prox_{γf}(z) = (prox_{γ·coe², g}(translation + coe·z) - translation) / coe

# Examples
```julia
# Create a scaled and shifted L2 ball: ||2x + [1,1]||₂ ≤ 1
g = IndicatorBallL2(1.0)
f = WrapperScalingTranslationFunction(g, 2.0, [1.0, 1.0])

# Create a shifted quadratic: (1/2)(x - 2)²
g = QuadraticFunction(1.0, 0.0, 0.0)  # (1/2)x²
f = WrapperScalingTranslationFunction(g, 1.0, -2.0)  # (1/2)(x + (-2))² = (1/2)(x - 2)²
```
"""
struct WrapperScalingTranslationFunction <: AbstractFunction 
    originalFunction::AbstractFunction 
    coe::Float64 
    translation::NumericVariable 
    buffer::NumericVariable 
    
    """
        WrapperScalingTranslationFunction(originalFunction::AbstractFunction, coe::Float64, translation::NumericVariable)

    Construct a wrapper function representing f(x) = originalFunction(coe·x + translation).

    # Arguments
    - `originalFunction::AbstractFunction`: The function g to be transformed
    - `coe::Float64`: The scaling coefficient (must be positive)
    - `translation::NumericVariable`: The translation vector/scalar

    # Throws
    - `ErrorException`: If coe ≤ 0

    # Examples
    ```julia
    # Scaled L1 norm: f(x) = ||2x||₁
    g = ElementwiseL1Norm()
    f = WrapperScalingTranslationFunction(g, 2.0, 0.0)

    # Shifted and scaled box constraint: -1 ≤ 3x + 2 ≤ 1
    g = IndicatorBox(-1.0, 1.0)
    f = WrapperScalingTranslationFunction(g, 3.0, 2.0)
    ```
    """
    function WrapperScalingTranslationFunction(originalFunction::AbstractFunction, coe::Float64, translation::NumericVariable)
        if coe <= 0.0
            error("WrapperScalingTranslationFunction: coe must be positive")
        end 
        
        # Create buffer with appropriate type based on translation
        if isa(translation, Number)
            buffer = 0.0  # Scalar buffer for scalar translation
        else
            buffer = similar(translation)  # Array buffer for array translation
        end
        
        new(originalFunction, coe, translation, buffer)
    end 
end 

# Trait checkers - delegate to original function
"""
    isProximal(f::WrapperScalingTranslationFunction)

Check if the wrapped function has a proximal operator.
Delegates to the original function's proximal property.
"""
isProximal(f::WrapperScalingTranslationFunction) = isProximal(f.originalFunction)

"""
    isSmooth(f::WrapperScalingTranslationFunction)

Check if the wrapped function is smooth (differentiable).
Delegates to the original function's smoothness property.
"""
isSmooth(f::WrapperScalingTranslationFunction) = isSmooth(f.originalFunction)

"""
    isConvex(f::WrapperScalingTranslationFunction)

Check if the wrapped function is convex.
Delegates to the original function's convexity property.
"""
isConvex(f::WrapperScalingTranslationFunction) = isConvex(f.originalFunction)

"""
    isSet(f::WrapperScalingTranslationFunction)

Check if the wrapped function is an indicator function of a set.
Delegates to the original function's set property.
"""
isSet(f::WrapperScalingTranslationFunction) = isSet(f.originalFunction)

"""
    (f::WrapperScalingTranslationFunction)(x::NumericVariable, enableParallel::Bool=false)

Evaluate the transformed function f(x) = g(coe·x + translation).

# Arguments
- `x::NumericVariable`: Input point
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Returns
- Function value at the transformed input

# Throws
- `ErrorException`: If input and translation dimensions don't match
"""
function (f::WrapperScalingTranslationFunction)(x::NumericVariable, enableParallel::Bool=false)
    if size(x) != size(f.translation)
        error("WrapperScalingTranslationFunction: input and translation must have the same size")
    end 
    return f.originalFunction(f.coe * x + f.translation, enableParallel)
end 

"""
    gradientOracle!(y::NumericVariable, f::WrapperScalingTranslationFunction, x::NumericVariable, enableParallel::Bool=false)

In-place computation of the gradient ∇f(x) = coe · ∇g(coe·x + translation).

# Mathematical Background
For f(x) = g(coe·x + translation), the chain rule gives:
∇f(x) = ∇g(coe·x + translation) · coe

# Arguments
- `y::NumericVariable`: Output buffer for the gradient (modified in-place)
- `f::WrapperScalingTranslationFunction`: The wrapped function
- `x::NumericVariable`: Point at which to evaluate the gradient
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Throws
- `ErrorException`: If original function is not smooth, or dimension mismatches occur
"""
function gradientOracle!(y::NumericVariable, f::WrapperScalingTranslationFunction, x::NumericVariable, enableParallel::Bool=false)
    if isSmooth(typeof(f.originalFunction)) == false
        error("WrapperScalingTranslationFunction: original function is not smooth")
    end 
    if isa(x, Number)
        error("WrapperScalingTranslationFunction: gradientOracle! does not support scalar input")
    end 
    if size(x) != size(f.translation)
        error("WrapperScalingTranslationFunction: input and translation must have the same size")
    end 
    if size(x) != size(y)
        error("WrapperScalingTranslationFunction: input and output must have the same size")
    end 
    y .= f.coe .* gradientOracle(f.originalFunction, f.coe * x + f.translation, enableParallel)
end 

"""
    gradientOracle(f::WrapperScalingTranslationFunction, x::NumericVariable, enableParallel::Bool=false)

Compute the gradient ∇f(x) = coe · ∇g(coe·x + translation).

# Arguments
- `f::WrapperScalingTranslationFunction`: The wrapped function
- `x::NumericVariable`: Point at which to evaluate the gradient
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Returns
- `NumericVariable`: The gradient vector/matrix

# Throws
- `ErrorException`: If original function is not smooth
"""
function gradientOracle(f::WrapperScalingTranslationFunction, x::NumericVariable, enableParallel::Bool=false)
    if isSmooth(typeof(f.originalFunction)) == false
        error("WrapperScalingTranslationFunction: original function is not smooth")
    end 
    if isa(x, Number) && isa(f.translation, Number)
        return f.coe * gradientOracle(f.originalFunction, f.coe * x + f.translation, enableParallel)
    end 

    y = similar(x)
    gradientOracle!(y, f, x, enableParallel)
    return y
end 

"""
    proximalOracle!(y::NumericVariable, f::WrapperScalingTranslationFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)

In-place computation of the proximal operator of f(x) = g(coe·x + translation).

# Mathematical Background
For f(x) = g(coe·x + translation), the proximal operator is:
prox_{γf}(z) = argmin_x { g(coe·x + translation) + (1/(2γ))||x - z||² }

Through variable substitution u = coe·x + translation, this becomes:
prox_{γf}(z) = (prox_{γ·coe², g}(translation + coe·z) - translation) / coe

# Algorithm
1. Transform input: buffer = coe·x + translation
2. Apply original proximal: prox_{γ·coe², g}(buffer)
3. Transform back: (result - translation) / coe

# Arguments
- `y::NumericVariable`: Output buffer for the result (modified in-place)
- `f::WrapperScalingTranslationFunction`: The wrapped function
- `x::NumericVariable`: Input point for the proximal operator
- `gamma::Float64`: Proximal parameter
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Throws
- `ErrorException`: If original function doesn't have proximal operator, or dimension mismatches occur
"""
function proximalOracle!(y::NumericVariable, f::WrapperScalingTranslationFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)
    if isProximal(typeof(f.originalFunction)) == false
        error("WrapperScalingTranslationFunction: original function is not proximal")
    end 
    if isa(x, Number)
        error("WrapperScalingTranslationFunction: proximalOracle! does not support scalar input")
    end 
    if size(x) != size(f.translation)
        error("WrapperScalingTranslationFunction: input and translation must have the same size")
    end 
    if size(x) != size(y)
        error("WrapperScalingTranslationFunction: input and output must have the same size")
    end 
    
    # Transform input: buffer = coe * x + translation
    f.buffer .= f.coe .* x .+ f.translation 

    # Apply original proximal with scaled gamma: prox_{γ·coe², g}(buffer)
    proximalOracle!(y, f.originalFunction, f.buffer, gamma * f.coe^2, enableParallel)

    # Transform result back to original space
    y .-= f.translation   # Subtract translation
    y ./= f.coe          # Scale by 1/coe
end 

"""
    proximalOracle(f::WrapperScalingTranslationFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)

Compute the proximal operator of f(x) = g(coe·x + translation).

Returns: prox_{γf}(x) = (prox_{γ·coe², g}(translation + coe·x) - translation) / coe

# Arguments
- `f::WrapperScalingTranslationFunction`: The wrapped function
- `x::NumericVariable`: Input point for the proximal operator
- `gamma::Float64`: Proximal parameter
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Returns
- `NumericVariable`: Result of the proximal operator

# Examples
```julia
# Proximal operator of scaled L2 ball: f(x) = I_{||2x||₂ ≤ 1}(x)
g = IndicatorBallL2(1.0)
f = WrapperScalingTranslationFunction(g, 2.0, 0.0)
result = proximalOracle(f, [1.0, 1.0], 1.0)  # Projects onto scaled ball
```

# Throws
- `ErrorException`: If original function doesn't have proximal operator
"""
function proximalOracle(f::WrapperScalingTranslationFunction, x::NumericVariable, gamma::Float64, enableParallel::Bool=false)
    if isProximal(typeof(f.originalFunction)) == false
        error("WrapperScalingTranslationFunction: original function is not proximal")
    end 
    if isa(x, Number)
        prox_center = f.coe * x + f.translation 
        y = proximalOracle(f.originalFunction, prox_center, gamma * f.coe^2, enableParallel)
        return (y - f.translation) * (1.0 / f.coe)
    end 
    
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 