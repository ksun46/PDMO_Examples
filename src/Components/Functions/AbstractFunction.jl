"""
    AbstractFunction

Abstract base type for all functions in the Bipartization optimization framework.

This type serves as the foundation for implementing mathematical functions that can be used
in optimization algorithms. It defines a common interface for function evaluation, 
gradients, and proximal operators.

# Interface Requirements

To implement a new function type, you must:

1. **Define a concrete struct** that inherits from `AbstractFunction`
2. **Implement function evaluation** by overriding the call operator
3. **Implement trait methods** to specify function properties
4. **Implement oracles** for smooth/proximal functions as needed

# Core Interface Methods

## Function Evaluation
```julia
(f::YourFunction)(x::NumericVariable, enableParallel::Bool=false) -> Float64
```

## Trait Methods (override as needed)
```julia
isSmooth(::Type{YourFunction}) = true/false     # Has gradient?
isProximal(::Type{YourFunction}) = true/false   # Has proximal operator?
isConvex(::Type{YourFunction}) = true/false     # Is convex?
isSet(::Type{YourFunction}) = true/false        # Is indicator function?
```

## Gradient Oracle (if isSmooth = true)
```julia
gradientOracle!(grad::NumericVariable, f::YourFunction, x::NumericVariable, enableParallel::Bool=false)
gradientOracle(f::YourFunction, x::NumericVariable, enableParallel::Bool=false)
```

## Proximal Oracle (if isProximal = true)
```julia
proximalOracle!(y::NumericVariable, f::YourFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
proximalOracle(f::YourFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
```

# Implementation Example

```julia
# Define a simple quadratic function: f(x) = (1/2)||x||²
struct SimpleQuadratic <: AbstractFunction
    # No fields needed for this simple case
end

# Specify traits
isSmooth(::Type{SimpleQuadratic}) = true
isConvex(::Type{SimpleQuadratic}) = true
isProximal(::Type{SimpleQuadratic}) = true

# Function evaluation
function (f::SimpleQuadratic)(x::NumericVariable, enableParallel::Bool=false)
    return 0.5 * sum(x.^2)
end

# Gradient oracle
function gradientOracle!(grad::NumericVariable, f::SimpleQuadratic, x::NumericVariable, enableParallel::Bool=false)
    grad .= x
end

# Proximal oracle
function proximalOracle!(y::NumericVariable, f::SimpleQuadratic, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    y .= x ./ (1.0 + gamma)
end
```

# Built-in Function Types

The framework provides many built-in function types:

- **Basic Functions**: `Zero`, `AffineFunction`, `QuadraticFunction`
- **Norms**: `ElementwiseL1Norm`, `FrobeniusNormSquare`, `MatrixNuclearNorm`
- **Indicators**: `IndicatorBox`, `IndicatorBallL2`, `IndicatorSOC`, `IndicatorPSD`
- **Wrappers**: `WrapperScalingTranslationFunction`, `WrapperScalarInputFunction`
- **User-defined**: `UserDefinedSmoothFunction`, `UserDefinedProximalFunction`

# Design Principles

1. **Modularity**: Each function type is self-contained
2. **Efficiency**: Support for in-place operations and parallel computation
3. **Flexibility**: Trait-based design allows algorithm selection
4. **Extensibility**: Easy to add new function types
5. **Type Safety**: Strong typing helps catch errors early


"""
abstract type AbstractFunction end 

""" 
    NumericVariable

Type alias for function arguments in the Bipartization framework.

This type represents the domain of functions, encompassing both scalar and array inputs.
It provides a unified interface for working with optimization variables of different
dimensions and structures.

# Definition
```julia
NumericVariable = Union{Float64, AbstractArray{Float64, N} where N}
```

# Supported Types
- **Scalar**: `Float64` - Single real number
- **Vector**: `Vector{Float64}` - 1D array of real numbers  
- **Matrix**: `Matrix{Float64}` - 2D array of real numbers
- **Tensor**: `Array{Float64, N}` - N-dimensional array of real numbers

# Examples
```julia
# Scalar variable
x_scalar::NumericVariable = 3.14

# Vector variable
x_vector::NumericVariable = [1.0, 2.0, 3.0]

# Matrix variable
x_matrix::NumericVariable = [1.0 2.0; 3.0 4.0]

# 3D tensor variable
x_tensor::NumericVariable = rand(2, 3, 4)
```

# Usage in Functions
Functions that accept `NumericVariable` arguments can handle different input types:

```julia
function (f::MyFunction)(x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Float64)
        # Handle scalar case
        return x^2
    else
        # Handle array case
        return sum(x.^2)
    end
end
```

# Design Rationale
- **Flexibility**: Supports various optimization variable structures
- **Type Safety**: Ensures only numeric types are used
- **Performance**: Avoids unnecessary type conversions
- **Compatibility**: Works with Julia's array ecosystem
"""
const NumericVariable = Union{Float64, AbstractArray{Float64, N} where N}

""" 
    isProximal(f::AbstractFunction) -> Bool
    isProximal(::Type{<:AbstractFunction}) -> Bool

Trait checker for the proximal operator capability.

Returns `true` if the function has an implemented proximal operator, `false` otherwise.
Functions with proximal operators can be used in proximal algorithms like ADMM,
forward-backward splitting, and Douglas-Rachford splitting.

# Mathematical Background
The proximal operator of a function f is defined as:
```
prox_{γf}(x) = argmin_z { f(z) + (1/(2γ))||z - x||² }
```

# Implementation Requirements
If `isProximal(f) = true`, then the function must implement:
- `proximalOracle!(y, f, x, γ)`: In-place proximal operator
- `proximalOracle(f, x, γ)`: Allocating proximal operator (has default implementation)

# Examples
```julia
# Check if a function has proximal operator
f = ElementwiseL1Norm()
isProximal(f)  # Returns true

# Use in algorithm selection
if isProximal(f)
    # Use proximal algorithm
    result = proximalOracle(f, x, γ)
else
    # Use different algorithm approach
    result = gradientOracle(f, x)
end
```

# Built-in Proximal Functions
- `ElementwiseL1Norm`: Soft thresholding
- `IndicatorBox`: Projection onto box constraints
- `IndicatorBallL2`: Projection onto L2 ball
- `Zero`: Identity operator
- Many indicator functions: projections onto constraint sets
"""
isProximal(::Type{<:AbstractFunction}) = false
isProximal(::T) where T <: AbstractFunction = isProximal(T)

""" 
    isSmooth(f::AbstractFunction) -> Bool
    isSmooth(::Type{<:AbstractFunction}) -> Bool

Trait checker for the smooth (differentiable) property.

Returns `true` if the function is smooth (differentiable) everywhere in its domain,
`false` otherwise. Smooth functions can be used in gradient-based optimization
algorithms like gradient descent, Newton's method, and quasi-Newton methods.

# Mathematical Background
A function f is smooth if it is continuously differentiable, meaning:
- The gradient ∇f(x) exists at every point x in the domain
- The gradient is continuous

# Implementation Requirements
If `isSmooth(f) = true`, then the function must implement:
- `gradientOracle!(grad, f, x)`: In-place gradient computation
- `gradientOracle(f, x)`: Allocating gradient computation (has default implementation)

# Examples
```julia
# Check if a function is smooth
f = QuadraticFunction(Q, q, r)
isSmooth(f)  # Returns true

# Use in algorithm selection
if isSmooth(f)
    # Use gradient-based algorithm
    grad = gradientOracle(f, x)
    x_new = x - α * grad  # Gradient descent step
else
    # Use derivative-free algorithm
    x_new = proximalOracle(f, x, γ)  # If proximal is available
end
```

# Built-in Smooth Functions
- `QuadraticFunction`: Gradient is linear
- `AffineFunction`: Gradient is constant
- `ComponentwiseExponentialFunction`: Gradient is exponential
- `Zero`: Gradient is zero
- `FrobeniusNormSquare`: Gradient is linear in matrix case

# Non-smooth Functions
- `ElementwiseL1Norm`: Not differentiable at zero
- Most indicator functions: Not differentiable on boundaries
- `MatrixNuclearNorm`: Not differentiable when singular values are zero
"""
isSmooth(::Type{<:AbstractFunction}) = false
isSmooth(::T) where T <: AbstractFunction = isSmooth(T)

""" 
    isConvex(f::AbstractFunction) -> Bool
    isConvex(::Type{<:AbstractFunction}) -> Bool

Trait checker for the convex property.

Returns `true` if the function is convex, `false` otherwise. Convex functions
have many desirable properties for optimization, including the guarantee that
any local minimum is also a global minimum.

# Mathematical Background
A function f is convex if for all x, y in its domain and λ ∈ [0,1]:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

# Properties of Convex Functions
- Any local minimum is a global minimum
- The set of global minimizers forms a convex set
- First-order optimality conditions are sufficient
- Many efficient optimization algorithms are guaranteed to converge

# Examples
```julia
# Check if a function is convex
f = ElementwiseL1Norm()
isConvex(f)  # Returns true

# Use in algorithm selection
if isConvex(f)
    # Use convex optimization algorithm
    # Global optimality guarantees apply
else
    # Use general nonlinear optimization
    # Local optimality only
end
```

# Built-in Convex Functions
- **Norms**: `ElementwiseL1Norm`, `FrobeniusNormSquare`
- **Indicators**: All indicator functions of convex sets
- **Basic**: `Zero`, `AffineFunction`
- **Quadratic**: `QuadraticFunction` (if positive semidefinite)
- **Exponential**: `ComponentwiseExponentialFunction`

# Non-convex Functions
- Some user-defined functions
- `MatrixNuclearNorm` with different weights
- Functions with non-convex constraints

# Algorithm Implications
- Convex functions enable global convergence guarantees
- Specialized convex optimization algorithms can be used
- Duality theory applies for convex functions
- Efficient solution methods are available
"""
isConvex(::Type{<:AbstractFunction}) = false 
isConvex(::T) where T <: AbstractFunction = isConvex(T)

""" 
    isSet(f::AbstractFunction) -> Bool
    isSet(::Type{<:AbstractFunction}) -> Bool

Trait checker for indicator functions of sets.

Returns `true` if the function is the indicator function of a set, `false` otherwise.
Indicator functions are fundamental in constrained optimization and represent
constraints as functions that are 0 inside the constraint set and ∞ outside.

# Mathematical Background
An indicator function of a set S is defined as:
```
I_S(x) = 0    if x ∈ S
I_S(x) = +∞   if x ∉ S
```

# Properties of Indicator Functions
- Always convex if the underlying set is convex
- The proximal operator is the projection onto the set
- Used to represent constraints in optimization problems
- Enable conversion between constrained and unconstrained formulations

# Examples
```julia
# Check if a function is an indicator function
f = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
isSet(f)  # Returns true

# Use in constraint handling
if isSet(f)
    # This represents a constraint
    # Proximal operator is projection onto the set
    projected_x = proximalOracle(f, x)
else
    # This is a regular objective function
    function_value = f(x)
end
```

# Built-in Indicator Functions
- **Box constraints**: `IndicatorBox`
- **Ball constraints**: `IndicatorBallL2`
- **Cone constraints**: `IndicatorSOC`, `IndicatorRotatedSOC`
- **Matrix constraints**: `IndicatorPSD`
- **Linear constraints**: `IndicatorLinearSubspace`, `IndicatorHyperplane`
- **Orthant constraints**: `IndicatorNonnegativeOrthant`
- **Custom constraints**: `IndicatorSumOfNVariables`

# Relationship to Proximal Operators
For indicator functions, the proximal operator is the projection:
```
prox_{γI_S}(x) = Proj_S(x) = argmin_{y∈S} ||y - x||²
```

# Algorithm Applications
- **Constrained optimization**: Represent feasible regions
- **Projection methods**: Direct projection onto constraint sets
- **Penalty methods**: Soft constraint handling
- **ADMM**: Splitting methods for constrained problems
"""
isSet(T::Type{<:AbstractFunction}) = false
isSet(::T) where T <: AbstractFunction = isSet(T)

"""
    (f::AbstractFunction)(x::NumericVariable, enableParallel::Bool=false) -> Float64

Function call operator for evaluating the function at a given point.

This is the primary interface for computing function values. Every concrete function 
type must implement this method to define how the function is evaluated.

# Arguments
- `f::AbstractFunction`: The function object to evaluate
- `x::NumericVariable`: The point at which to evaluate the function
- `enableParallel::Bool=false`: Whether to enable parallel computation (when supported)

# Returns
- `Float64`: The function value f(x)

# Implementation Requirements
Every concrete function type must override this method. The implementation should:
1. Validate input dimensions and types
2. Compute the function value efficiently
3. Handle edge cases appropriately
4. Optionally support parallel computation

# Examples
```julia
# Basic usage
f = ElementwiseL1Norm(0.5)
x = [1.0, -2.0, 3.0]
val = f(x)  # Equivalent to f(x, false)

# With parallel computation
val_parallel = f(x, true)  # May use parallel algorithms if supported

# Different input types
f_matrix = FrobeniusNormSquare(A, b, m, n)
X = rand(m, n)
val_matrix = f_matrix(X)  # Works with matrix inputs
```

# Common Patterns
- **Indicator functions**: Return 0.0 if x is in the constraint set, +∞ otherwise
- **Norm functions**: Return ||x||_p for various norms
- **Regularization**: Return penalty terms for optimization
- **Composite functions**: Combine multiple function evaluations

# Performance Notes
- Implementations should minimize memory allocations
- Use in-place operations when possible
- Consider vectorization for array operations
- Parallel computation should be used for large-scale problems

# Error Handling
- Throw descriptive errors for dimension mismatches
- Use appropriate tolerances for numerical comparisons
- Handle special cases (NaN, Inf, empty inputs) gracefully
"""
function (f::AbstractFunction)(x::NumericVariable, enableParallel::Bool=false)
    error("Function evaluation not implemented for $(typeof(f))")
end

"""
    proximalOracle!(y::NumericVariable, f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)

In-place computation of the proximal operator for function f.

The proximal operator is a fundamental concept in convex optimization and is defined as:
```
prox_{γf}(x) = argmin_z { f(z) + (1/(2γ))||z - x||² }
```

This function computes the result in-place, storing it in the pre-allocated output `y`.

# Arguments
- `y::NumericVariable`: Pre-allocated output buffer (modified in-place)
- `f::AbstractFunction`: The function for which to compute the proximal operator
- `x::NumericVariable`: The input point
- `gamma::Float64=1.0`: The proximal parameter γ > 0
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Mathematical Background
The proximal operator generalizes the concept of projection onto a set:
- For indicator functions: prox_{γI_S}(x) = Proj_S(x) (projection onto set S)
- For L1 norm: prox_{γ||·||₁}(x) = soft_threshold(x, γ) (soft thresholding)
- For quadratic functions: Has explicit closed-form solution

# Implementation Requirements
Functions with `isProximal(f) = true` must implement this method. The implementation should:
1. Validate input dimensions and parameter values
2. Compute the proximal operator efficiently
3. Store the result in the pre-allocated output `y`
4. Handle edge cases and numerical stability
5. Optionally support parallel computation

# Examples
```julia
# Basic usage with indicator function
f = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
x = [2.0, -2.0]
y = similar(x)
proximalOracle!(y, f, x, 1.0)  # Projects onto box, y = [1.0, -1.0]

# L1 norm with soft thresholding
f = ElementwiseL1Norm(0.5)
x = [2.0, -3.0, 0.3]
y = similar(x)
proximalOracle!(y, f, x, 1.0)  # y = [1.5, -2.5, 0.0]

# Matrix functions
f = IndicatorPSD(3)
X = rand(3, 3)
Y = similar(X)
proximalOracle!(Y, f, X, 1.0)  # Projects onto PSD cone
```

# Performance Considerations
- Pre-allocate output buffer to avoid memory allocation
- Use in-place operations within the implementation
- Consider cache efficiency for large arrays
- Parallel computation can be beneficial for large-scale problems

# Common Proximal Operators
- **Indicator functions**: Projection onto constraint sets
- **L1 norm**: Soft thresholding operator
- **L2 ball**: Projection onto ball (scaling if outside)
- **Zero function**: Identity operator
- **Quadratic functions**: Require solving linear systems

# Error Handling
- Input validation for dimensions and parameter ranges
- Numerical stability considerations
- Appropriate handling of edge cases (zero gamma, boundary cases)
"""
function proximalOracle!(y::NumericVariable, f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    error("proximalOracle! not implemented for $(typeof(f))")
end

"""
    proximalOracle(f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false) -> NumericVariable

Computes the proximal operator for function f, returning a new array.

This is the allocating version of the proximal operator that creates and returns a new
array containing the result. For performance-critical applications, consider using
the in-place version `proximalOracle!` instead.

# Arguments
- `f::AbstractFunction`: The function for which to compute the proximal operator
- `x::NumericVariable`: The input point
- `gamma::Float64=1.0`: The proximal parameter γ > 0
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Returns
- `NumericVariable`: The result prox_{γf}(x), same type and size as input `x`

# Mathematical Definition
Computes: prox_{γf}(x) = argmin_z { f(z) + (1/(2γ))||z - x||² }

# Implementation Notes
The default implementation:
1. Allocates a new array `y = similar(x)`
2. Calls `proximalOracle!(y, f, x, gamma, enableParallel)`
3. Returns the result

Concrete function types may override this method for more efficient implementations,
but most can rely on the default implementation.

# Examples
```julia
# Basic usage - returns new array
f = ElementwiseL1Norm(0.5)
x = [2.0, -3.0, 0.3]
result = proximalOracle(f, x, 1.0)  # Returns [1.5, -2.5, 0.0]

# Chaining operations
f1 = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
f2 = ElementwiseL1Norm(0.1)
x = [2.0, -2.0]
result = proximalOracle(f2, proximalOracle(f1, x))  # Compose operations

# Different input types
f_matrix = IndicatorPSD(3)
X = rand(3, 3)
Y = proximalOracle(f_matrix, X)  # Returns projected matrix

# Scalar inputs (for appropriate functions)
f_scalar = ElementwiseL1Norm(1.0)
x_scalar = 2.0
result_scalar = proximalOracle(f_scalar, x_scalar)  # Returns 1.0
```

# Performance Considerations
- **Memory allocation**: Creates new array on each call
- **Prefer in-place version**: Use `proximalOracle!` for better performance
- **Temporary arrays**: Consider pre-allocating arrays for repeated calls
- **Memory pressure**: May cause garbage collection in tight loops

# Algorithm Applications
- **Proximal gradient methods**: Forward-backward splitting
- **ADMM**: Alternating direction method of multipliers
- **Douglas-Rachford**: Splitting methods
- **Primal-dual methods**: Condat-Vu, Chambolle-Pock algorithms

# Common Use Cases
- **Constraint projection**: Project onto feasible sets
- **Regularization**: Apply regularization operators
- **Denoising**: Soft thresholding for sparse signals
- **Matrix completion**: Project onto low-rank or PSD constraints


"""
function proximalOracle(f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

"""
    gradientOracle!(grad::NumericVariable, f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false)

In-place computation of the gradient of function f at point x.

This function computes the gradient (or subgradient for non-smooth functions) and stores
the result in the pre-allocated output buffer `grad`.

# Arguments
- `grad::NumericVariable`: Pre-allocated output buffer for gradient (modified in-place)
- `f::AbstractFunction`: The function for which to compute the gradient
- `x::NumericVariable`: The point at which to evaluate the gradient
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Mathematical Background
For smooth functions, the gradient is defined as:
```
∇f(x) = lim_{h→0} [f(x+h) - f(x)] / h
```

For vector/matrix functions, this becomes the vector/matrix of partial derivatives.

# Implementation Requirements
Functions with `isSmooth(f) = true` must implement this method. The implementation should:
1. Validate input dimensions match between `x` and `grad`
2. Compute the gradient efficiently
3. Store the result in the pre-allocated output `grad`
4. Handle numerical stability issues
5. Optionally support parallel computation

# Examples
```julia
# Basic usage with quadratic function
Q = [2.0 0.0; 0.0 2.0]
q = [1.0, 1.0]
f = QuadraticFunction(sparse(Q), q, 0.0)
x = [1.0, 2.0]
grad = similar(x)
gradientOracle!(grad, f, x)  # grad = Q*x + q = [3.0, 5.0]

# Affine function (constant gradient)
A = [2.0, 3.0]
f = AffineFunction(A, 0.0)
x = [1.0, 1.0]
grad = similar(x)
gradientOracle!(grad, f, x)  # grad = A = [2.0, 3.0]

# Matrix function
A = rand(10, 5)
b = rand(10, 3)
f = FrobeniusNormSquare(A, b, 5, 3)
X = rand(5, 3)
grad = similar(X)
gradientOracle!(grad, f, X)  # grad = 2*A'*(A*X - b)
```

# Performance Considerations
- **Memory efficiency**: Uses pre-allocated buffer to avoid allocations
- **In-place operations**: Implementation should minimize temporary arrays
- **Vectorization**: Take advantage of BLAS/LAPACK when possible
- **Parallel computation**: Can be beneficial for large-scale problems
- **Cache efficiency**: Consider memory access patterns

# Common Gradient Patterns
- **Linear functions**: Gradient is constant (independent of x)
- **Quadratic functions**: Gradient is linear in x
- **Least squares**: Gradient involves matrix-vector products
- **Exponential functions**: Gradient involves exponential evaluations
- **Composite functions**: Use chain rule

# Numerical Considerations
- **Finite precision**: Be aware of numerical errors
- **Scaling**: Consider function scaling for numerical stability
- **Condition numbers**: Well-conditioned problems have more stable gradients
- **Overflow/underflow**: Handle extreme values appropriately

# Algorithm Applications
- **Gradient descent**: Steepest descent optimization
- **Newton's method**: Second-order optimization
- **Quasi-Newton methods**: BFGS, L-BFGS
- **Conjugate gradient**: Iterative linear system solvers
- **Trust region methods**: Model-based optimization

# Error Handling
- Validate that `x` and `grad` have compatible dimensions
- Check for numerical issues (NaN, Inf)
- Provide meaningful error messages for dimension mismatches
- Handle edge cases gracefully
"""
function gradientOracle!(grad::NumericVariable, f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false)
    error("gradientOracle! not implemented for $(typeof(f))")
end

"""
    gradientOracle(f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false) -> NumericVariable

Computes the gradient of function f at point x, returning a new array.

This is the allocating version of the gradient computation that creates and returns a new
array containing the gradient. For performance-critical applications, consider using
the in-place version `gradientOracle!` instead.

# Arguments
- `f::AbstractFunction`: The function for which to compute the gradient
- `x::NumericVariable`: The point at which to evaluate the gradient
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Returns
- `NumericVariable`: The gradient ∇f(x), same type and size as input `x`

# Mathematical Definition
For smooth functions: ∇f(x) where each component is ∂f/∂xᵢ
For vector functions: Returns the Jacobian or gradient matrix

# Implementation Notes
The default implementation:
1. Allocates a new array `grad = similar(x)`
2. Calls `gradientOracle!(grad, f, x, enableParallel)`
3. Returns the result

Concrete function types may override this method for more efficient implementations,
but most can rely on the default implementation.

# Examples
```julia
# Basic usage with quadratic function
Q = sparse([2.0 0.0; 0.0 2.0])
q = [1.0, 1.0]
f = QuadraticFunction(Q, q, 0.0)
x = [1.0, 2.0]
grad = gradientOracle(f, x)  # Returns [3.0, 5.0]

# Gradient descent step
f = QuadraticFunction(Q, q, 0.0)
x = [1.0, 2.0]
α = 0.1  # Step size
grad = gradientOracle(f, x)
x_new = x - α * grad  # Gradient descent update

# Matrix function gradients
A = rand(10, 5)
b = rand(10, 3)
f = FrobeniusNormSquare(A, b, 5, 3)
X = rand(5, 3)
grad_X = gradientOracle(f, X)  # Returns 5×3 gradient matrix

# Chaining with function evaluation
f = ComponentwiseExponentialFunction([1.0, 2.0])
x = [0.0, 1.0]
val = f(x)  # Function value
grad = gradientOracle(f, x)  # Gradient at same point
```

# Performance Considerations
- **Memory allocation**: Creates new array on each call
- **Prefer in-place version**: Use `gradientOracle!` for better performance
- **Temporary arrays**: Consider pre-allocating arrays for repeated calls
- **Memory pressure**: May cause garbage collection in tight loops
- **Automatic differentiation**: Consider AD tools for complex functions

# Algorithm Applications
- **Steepest descent**: x_{k+1} = x_k - α∇f(x_k)
- **Momentum methods**: Incorporate previous gradient information
- **Adam optimizer**: Adaptive gradient methods
- **Line search**: Determine optimal step sizes
- **Quasi-Newton**: Approximate Hessian from gradients

# Common Use Cases
- **Optimization**: First-order optimization algorithms
- **Machine learning**: Backpropagation and gradient-based training
- **Sensitivity analysis**: Study function behavior
- **Root finding**: Newton's method for systems of equations
- **Numerical integration**: Gradient-based quadrature

# Numerical Considerations
- **Gradient magnitude**: Large gradients may indicate poor scaling
- **Numerical derivatives**: Consider finite difference approximations
- **Automatic differentiation**: Tools like ForwardDiff.jl or ReverseDiff.jl
- **Condition number**: Well-conditioned problems have stable gradients

# Error Handling
- Functions must be smooth (`isSmooth(f) = true`)
- Input validation for dimensions and types
- Handling of numerical issues (NaN, Inf)
- Meaningful error messages for debugging


"""
function gradientOracle(f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false)
    grad = similar(x)
    gradientOracle!(grad, f, x, enableParallel)
    return grad
end


# Include and export all function implementations
include("AffineFunction.jl")
include("ComponentwiseExponentialFunction.jl")
include("ElementwiseL1Norm.jl")
include("FrobeniusNormSquare.jl")
include("IndicatorBallL2.jl")
include("IndicatorBox.jl")
include("IndicatorHyperplane.jl")
include("IndicatorLinearSubspace.jl")
include("IndicatorNonnegativeOrthant.jl")
include("IndicatorPSD.jl")
include("IndicatorRotatedSOC.jl")
include("IndicatorSOC.jl")
include("IndicatorSumOfNVariables.jl")
include("MatrixNuclearNorm.jl")
include("QuadraticFunction.jl")
include("UserDefinedProximalFunction.jl")
include("UserDefinedSmoothFunction.jl")
include("WeightedMatrixL1Norm.jl")
include("WrapperScalarInputFunction.jl")
include("WrapperScalingTranslationFunction.jl")
include("Zero.jl")

include("AbstractFunctionUtil.jl")

