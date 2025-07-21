"""
    estimateLipschitzConstant(f::AbstractFunction, x::NumericVariable; maxTrials::Int=50, 
                             minStepSize::Float64=1e-6, maxStepSize::Float64=1.0) -> Float64

Estimate the Lipschitz constant of the gradient ∇f at point x using multiple sampling strategies.

This function provides robust estimation of the Lipschitz constant L such that 
||∇f(x) - ∇f(y)|| ≤ L||x - y|| for points near x. It handles special cases with exact
solutions and uses multiple numerical strategies for general functions.

# Arguments
- `f::AbstractFunction`: The function for which to estimate the Lipschitz constant
- `x::NumericVariable`: The point around which to estimate the constant
- `maxTrials::Int=50`: Maximum number of sampling trials for estimation
- `minStepSize::Float64=1e-6`: Minimum step size for finite differences
- `maxStepSize::Float64=1.0`: Maximum step size for finite differences

# Returns
- `Float64`: Estimated Lipschitz constant (conservative upper bound)

# Algorithm
The function uses a multi-strategy approach:

## Exact Cases
- **QuadraticFunction**: Returns 2||Q|| where Q is the Hessian matrix
- **AffineFunction/Zero**: Returns 0 (constant gradient)

## Numerical Estimation (3 strategies)
1. **Random Directions**: Sample random unit directions with multiple scales
2. **Coordinate Directions**: Test axis-aligned perturbations (for problems ≤ 1000 dimensions)  
3. **Adaptive Scaling**: Use gradient magnitude to inform step size selection

## Statistical Robustness
- Collects estimates from all strategies
- Removes outliers using percentile-based filtering
- Returns conservative estimate (90th percentile or 1.2× median)
- Handles edge cases with appropriate fallbacks

# Examples
```julia
# For a quadratic function f(x) = 0.5x'Qx
L = estimateLipschitzConstant(f, x0)  # Returns 2*opnorm(Q)

# For a general smooth function
L = estimateLipschitzConstant(f, x0, maxTrials=100)  # More thorough sampling
```

# Notes
- The estimate is intentionally conservative to ensure algorithm stability
- For high-dimensional problems, coordinate sampling is limited for efficiency
- Multiple strategies provide robustness across different function types
- Handles both scalar and vector inputs appropriately
"""
function estimateLipschitzConstant(f::AbstractFunction, x::NumericVariable;
    maxTrials::Int=50, minStepSize::Float64=1e-6, maxStepSize::Float64=1.0)
    
    # Handle special cases with exact solutions
    if isa(f, QuadraticFunction)
        return 2.0 * opnorm(Array(f.Q))
    end
    
    if isa(f, AffineFunction) || isa(f, Zero)
        return 0.0 
    end 

    # Improved estimation using multiple strategies
    grad1 = gradientOracle(f, x) 
    
    if isa(x, Number)
        # For scalar functions, use simple finite difference
        ret = 0.0
        for _ in 1:maxTrials
            h = minStepSize + (maxStepSize - minStepSize) * rand()
            grad_right = gradientOracle(f, x + h)
            grad_left = gradientOracle(f, x - h)
            ret = max(ret, abs(grad_right - grad_left) / (2*h))
        end
        return 1.1 * ret
    end
    
    # For vector/matrix functions, use multiple estimation strategies
    grad2 = similar(grad1)
    perturbed = similar(x)
    direction = similar(x)
    
    estimates = Float64[]
    
    # Strategy 1: Random directions with multiple scales
    for trial in 1:div(maxTrials, 3)
        copyto!(perturbed, x)
        randn!(direction)  # Fill with random normal values
        direction_norm = norm(direction)
        
        if direction_norm > 1e-12  # Avoid zero directions
            # Normalize direction and try multiple step sizes
            direction ./= direction_norm
            
            for scale in [1e-4, 1e-3, 1e-2, 1e-1]
                step_size = scale * (1.0 + 0.5 * randn())  # Add some randomness
                step_size = clamp(step_size, minStepSize, maxStepSize)
                
                # perturbed = x + step_size * direction
                copyto!(perturbed, x)
                axpy!(step_size, direction, perturbed)
                gradientOracle!(grad2, f, perturbed)
                
                grad_diff_norm = norm(grad2 .- grad1)
                if grad_diff_norm > 1e-12
                    estimate = grad_diff_norm / step_size
                    push!(estimates, estimate)
                end
            end
        end
    end
    
    # Strategy 2: Coordinate directions (for high-dimensional problems)
    if length(x) <= 1000  # Only for reasonably sized problems
        coordinate_trials = min(div(maxTrials, 3), length(x))
        coords_to_try = randperm(length(x))[1:coordinate_trials]
        
        for coord_idx in coords_to_try
            copyto!(perturbed, x)
            
            for step_size in [1e-4, 1e-3, 1e-2]
                perturbed[coord_idx] += step_size
                gradientOracle!(grad2, f, perturbed)
                
                grad_diff_norm = norm(grad2 .- grad1)
                if grad_diff_norm > 1e-12
                    estimate = grad_diff_norm / step_size
                    push!(estimates, estimate)
                end
                
                perturbed[coord_idx] = x[coord_idx]  # Reset
            end
        end
    end
    
    # Strategy 3: Adaptive step size based on gradient magnitude
    remaining_trials = maxTrials - div(maxTrials, 3) * 2
    base_step = min(0.01 * norm(x) / (norm(grad1) + 1e-8), maxStepSize)
    
    for trial in 1:remaining_trials
        copyto!(perturbed, x)
        randn!(direction)
        direction_norm = norm(direction)
        
        if direction_norm > 1e-12
            direction ./= direction_norm
            step_size = base_step * (0.1 + 1.9 * rand())  # Random scaling between 0.1 and 2.0
            step_size = clamp(step_size, minStepSize, maxStepSize)
            
            # perturbed = x + step_size * direction
            copyto!(perturbed, x)
            axpy!(step_size, direction, perturbed)
            gradientOracle!(grad2, f, perturbed)
            
            grad_diff_norm = norm(grad2 .- grad1)
            if grad_diff_norm > 1e-12
                estimate = grad_diff_norm / step_size
                push!(estimates, estimate)
            end
        end
    end
    
    # Robust statistical estimation
    if isempty(estimates)
        @warn "No valid Lipschitz estimates found, using conservative default"
        return 1.0
    end
    
    # Remove outliers and compute robust estimate
    sort!(estimates)
    n = length(estimates)
    
    if n >= 10
        # Use 90th percentile to be conservative but not overly so
        percentile_90 = estimates[min(n, max(1, round(Int, 0.9 * n)))]
        # Also consider the median for stability
        median_est = estimates[div(n, 2)]
        final_estimate = max(percentile_90, 1.2 * median_est)
    elseif n >= 3
        # For smaller samples, use maximum of the top estimates
        top_third = estimates[max(1, div(2*n, 3)):end]
        final_estimate = 1.3 * maximum(top_third)
    else
        # Very few estimates, be conservative
        final_estimate = 1.5 * maximum(estimates)
    end
    
    return final_estimate
end


"""
    proximalOracleOfConjugate!(y::NumericVariable, f::AbstractFunction, x::NumericVariable, 
                              gamma::Float64=1.0, enableParallel::Bool=false)

In-place computation of the proximal operator of the convex conjugate f* using Moreau's identity.

This function computes y = prox_{γf*}(x) where f* is the convex conjugate of f.
It uses the Moreau identity: prox_{γf*}(x) = x - γ * prox_{f/γ}(x/γ).

# Arguments
- `y::NumericVariable`: Output variable (modified in-place)
- `f::AbstractFunction`: The function whose conjugate proximal operator to compute
- `x::NumericVariable`: Input point
- `gamma::Float64=1.0`: Proximal parameter (must be positive)
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Effects
- Modifies `y` in-place to contain prox_{γf*}(x)
- Requires `f` to have a proximal oracle implementation

# Mathematical Background
The Moreau identity relates the proximal operators of a function and its conjugate:
- prox_{γf*}(x) + γ * prox_{f/γ}(x/γ) = x

This identity allows computing the proximal operator of f* when only the proximal
operator of f is available, which is common in primal-dual algorithms.

# Errors
- Throws error if `f` does not have a proximal oracle
- Throws error if `x` and `y` have different sizes
- Throws error if `gamma ≤ 0`
- Throws error for scalar inputs (use scalar version of proximal operators)

# Applications
- Primal-dual splitting algorithms (ADMM, Condat-Vu)
- Computing dual proximal steps when only primal proximal is available
- Implementing algorithms that require conjugate function evaluations

# Example
```julia
# For constraint f = indicator of convex set C
# prox_{γf*}(x) computes projection onto γC
proximalOracleOfConjugate!(y, f, x, gamma)
```
"""
function proximalOracleOfConjugate!(y::NumericVariable, f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if (isa(f, WrapperScalarInputFunction) && isProximal(f) == false) ||
       (isa(f, WrapperScalarInputFunction) == false && isProximal(typeof(f)) == false)
        error("proximalOracleOfConjugate! requires f to have a proximal oracle")
    end 
    
    if size(x) != size(y)
        error("proximalOracleOfConjugate!: x and y must have the same size")
    end 

    if gamma <= 0.0
        error("proximalOracleOfConjugate!: gamma must be positive")
    end 

    if isa(x, Number)
        error("proximalOracleOfConjugate!: does not support scalar input")
    end 

    scaled_x = x ./ gamma
    proximalOracle!(y, f, scaled_x, 1.0/gamma, enableParallel)
    axpby!(1.0, x, -gamma, y)
end


"""
    proximalOracleOfConjugate(f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, 
                             enableParallel::Bool=false) -> NumericVariable

Compute the proximal operator of the convex conjugate f* using Moreau's identity.

This function returns prox_{γf*}(x) where f* is the convex conjugate of f.
It uses the Moreau identity: prox_{γf*}(x) = x - γ * prox_{f/γ}(x/γ).

# Arguments
- `f::AbstractFunction`: The function whose conjugate proximal operator to compute
- `x::NumericVariable`: Input point
- `gamma::Float64=1.0`: Proximal parameter (must be positive)  
- `enableParallel::Bool=false`: Whether to enable parallel computation

# Returns
- `NumericVariable`: Result prox_{γf*}(x), same type and size as input `x`

# Mathematical Background
The proximal operator of the convex conjugate is fundamental in convex optimization:
- For f*(y) = sup_x [⟨x,y⟩ - f(x)], the conjugate function
- prox_{γf*}(x) = argmin_y [½||y-x||² + γf*(y)]

The Moreau identity provides an efficient way to compute this without explicitly
constructing the conjugate function, requiring only the proximal operator of f.

# Errors
- Throws error if `f` does not have a proximal oracle
- Throws error if `gamma ≤ 0`

# Applications
- **Dual variable updates** in primal-dual algorithms
- **Constraint handling** where f is an indicator function
- **Regularization** in optimization problems
- **Image processing** and signal processing applications

# Examples
```julia
# For L1 norm f(x) = ||x||₁
# The conjugate f*(y) = indicator of ||y||∞ ≤ 1
y = proximalOracleOfConjugate(l1_norm, x, gamma)  # Projects onto ℓ∞ ball

# For indicator of convex set f = δ_C
# The conjugate f*(y) = σ_C(y) (support function)
y = proximalOracleOfConjugate(indicator_C, x, gamma)  # Scaled projection
```


"""
function proximalOracleOfConjugate(f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if (isa(f, WrapperScalarInputFunction) && isProximal(f) == false) ||
       (isa(f, WrapperScalarInputFunction) == false && isProximal(typeof(f)) == false)
        error("proximalOracleOfConjugate requires f to have a proximal oracle")
    end 

    if gamma <= 0.0
        error("proximalOracleOfConjugate: gamma must be positive")
    end 

    if isa(x, Number)
        return x - gamma * proximalOracle(f, x/gamma, 1.0/gamma, enableParallel)
    end 

    y = similar(x)
    proximalOracleOfConjugate!(y, f, x, gamma, enableParallel)
    return y
end
