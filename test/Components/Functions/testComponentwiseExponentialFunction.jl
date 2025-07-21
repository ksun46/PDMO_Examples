using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "ComponentwiseExponentialFunction Tests" begin
    @testset "Constructor" begin
        # Test construction with coefficient vector
        coeffs = [1.0, 2.0, 0.5]
        f = ComponentwiseExponentialFunction(coeffs)
        @test f isa ComponentwiseExponentialFunction
        @test f isa AbstractFunction
        @test f.coefficients == coeffs
        
        # Test construction with dimension
        n = 5
        f_dim = ComponentwiseExponentialFunction(n)
        @test length(f_dim.coefficients) == n
        @test f_dim.coefficients == ones(n)
        
        # Test error cases
        @test_throws ErrorException ComponentwiseExponentialFunction(Float64[])  # Empty coefficients
        @test_throws ErrorException ComponentwiseExponentialFunction([-1.0, 1.0])  # Negative coefficient
        @test_throws ErrorException ComponentwiseExponentialFunction([1.0, -0.1])  # Negative coefficient
        
        # Test with zero coefficients (should be allowed)
        f_zero = ComponentwiseExponentialFunction([0.0, 1.0, 0.0])
        @test f_zero.coefficients == [0.0, 1.0, 0.0]
        
        # Test with very small positive coefficients
        f_small = ComponentwiseExponentialFunction([1e-10, 1.0])
        @test f_small.coefficients == [1e-10, 1.0]
    end

    @testset "Function Traits" begin
        @test isConvex(ComponentwiseExponentialFunction) == true
        @test isSmooth(ComponentwiseExponentialFunction) == true
        @test isProximal(ComponentwiseExponentialFunction) == false  # No proximal operator implemented
        @test isSet(ComponentwiseExponentialFunction) == false
    end

    @testset "Function Evaluation" begin
        # Test simple case: f(x) = sum(coeffs .* exp.(x))
        coeffs = [1.0, 2.0, 0.5]
        f = ComponentwiseExponentialFunction(coeffs)
        
        x = [0.0, 0.0, 0.0]  # exp(0) = 1
        expected = sum(coeffs .* exp.(x))  # 1*1 + 2*1 + 0.5*1 = 3.5
        @test f(x) ≈ expected
        
        # Test with non-zero input
        x2 = [1.0, -1.0, 2.0]
        expected2 = coeffs[1] * exp(1.0) + coeffs[2] * exp(-1.0) + coeffs[3] * exp(2.0)
        @test f(x2) ≈ expected2
        
        # Test with zero coefficients
        coeffs_zero = [0.0, 1.0, 0.0]
        f_zero = ComponentwiseExponentialFunction(coeffs_zero)
        x3 = [10.0, 2.0, -5.0]
        expected3 = 0.0 * exp(10.0) + 1.0 * exp(2.0) + 0.0 * exp(-5.0)
        @test f_zero(x3) ≈ expected3
        
        # Test with unit coefficients
        f_unit = ComponentwiseExponentialFunction(3)
        x4 = [1.0, 2.0, 3.0]
        expected4 = exp(1.0) + exp(2.0) + exp(3.0)
        @test f_unit(x4) ≈ expected4
    end

    @testset "Gradient Oracle" begin
        # Test gradient: ∇f(x) = coeffs .* exp.(x)
        coeffs = [2.0, 1.5, 0.8]
        f = ComponentwiseExponentialFunction(coeffs)
        
        x = [1.0, -1.0, 0.5]
        expected_grad = coeffs .* exp.(x)
        
        # Test non-mutating gradient
        grad = gradientOracle(f, x)
        @test grad ≈ expected_grad
        @test size(grad) == size(x)
        
        # Test in-place gradient
        grad_inplace = similar(x)
        gradientOracle!(grad_inplace, f, x)
        @test grad_inplace ≈ expected_grad
        
        # Test gradient at zero
        x_zero = zeros(3)
        grad_zero = gradientOracle(f, x_zero)
        @test grad_zero ≈ coeffs  # exp(0) = 1
        
        # Test finite difference validation
        h = 1e-8
        grad_numerical = similar(x)
        for i in 1:length(x)
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            grad_numerical[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        end
        @test grad ≈ grad_numerical atol=1e-6
        
        # Test with zero coefficients
        coeffs_mixed = [0.0, 2.0, 0.0]
        f_mixed = ComponentwiseExponentialFunction(coeffs_mixed)
        x_test = [5.0, 1.0, -2.0]
        grad_mixed = gradientOracle(f_mixed, x_test)
        expected_mixed = [0.0, 2.0 * exp(1.0), 0.0]
        @test grad_mixed ≈ expected_mixed
    end

    @testset "Edge Cases" begin
        # Test with single element
        f_single = ComponentwiseExponentialFunction([3.0])
        x_single = [2.0]
        @test f_single(x_single) ≈ 3.0 * exp(2.0)
        
        grad_single = gradientOracle(f_single, x_single)
        @test grad_single ≈ [3.0 * exp(2.0)]
        
        # Test with very large input values
        coeffs_large = [1.0, 1.0]
        f_large = ComponentwiseExponentialFunction(coeffs_large)
        x_large = [10.0, -10.0]  # Mix of large positive and negative
        
        val_large = f_large(x_large)
        @test isfinite(val_large)
        @test val_large > 0.0
        
        grad_large = gradientOracle(f_large, x_large)
        @test all(isfinite.(grad_large))
        @test all(grad_large .> 0.0)
        
        # Test with very small input values
        x_small = [-50.0, -50.0]  # Very negative values
        val_small = f_large(x_small)
        @test val_small ≈ 2.0 * exp(-50.0)  # Should be very small but positive
        @test val_small > 0.0
        
        # Test with mixed large and small coefficients
        coeffs_mixed = [1e-6, 1e6]
        f_mixed = ComponentwiseExponentialFunction(coeffs_mixed)
        x_mixed = [0.0, 0.0]
        expected_mixed = 1e-6 * 1.0 + 1e6 * 1.0
        @test f_mixed(x_mixed) ≈ expected_mixed
        
        # Test numerical stability
        x_extreme = [100.0, -100.0, 0.0]
        coeffs_extreme = [1e-10, 1.0, 1e10]
        f_extreme = ComponentwiseExponentialFunction(coeffs_extreme)
        
        val_extreme = f_extreme(x_extreme)
        @test isfinite(val_extreme)
        
        grad_extreme = gradientOracle(f_extreme, x_extreme)
        @test all(isfinite.(grad_extreme))
    end

    @testset "Mathematical Properties" begin
        coeffs = [1.0, 2.0, 0.5]
        f = ComponentwiseExponentialFunction(coeffs)
        
        # Test convexity: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
        x1 = [1.0, -1.0, 2.0]
        x2 = [-1.0, 2.0, 0.0]
        λ = 0.3
        
        x_combo = λ * x1 + (1 - λ) * x2
        f_combo = f(x_combo)
        f_convex_combo = λ * f(x1) + (1 - λ) * f(x2)
        
        @test f_combo ≤ f_convex_combo + 1e-10
        
        # Test that function is always positive
        x_random = randn(3)
        @test f(x_random) > 0.0
        
        # Test that gradient is always positive (since coeffs ≥ 0 and exp > 0)
        grad_random = gradientOracle(f, x_random)
        @test all(grad_random .≥ 0.0)
        
        # Test monotonicity in each component (when coefficient > 0)
        x_base = [0.0, 0.0, 0.0]
        for i in 1:length(coeffs)
            if coeffs[i] > 0
                x_plus = copy(x_base)
                x_plus[i] += 1.0
                @test f(x_plus) > f(x_base)
            end
        end
        
        # Test that gradient equals derivative of function value
        x_test = [0.5, -0.5, 1.0]
        grad_test = gradientOracle(f, x_test)
        
        # For exponential function, ∇f(x) = coeffs .* exp.(x)
        # And f(x) = sum(coeffs .* exp.(x))
        # So ∂f/∂x_i = coeffs[i] * exp(x[i]) = grad_test[i]
        for i in 1:length(x_test)
            expected_partial = coeffs[i] * exp(x_test[i])
            @test grad_test[i] ≈ expected_partial
        end
        
        # Test scaling property: if we scale coefficients, function scales proportionally
        scale_factor = 3.0
        coeffs_scaled = scale_factor * coeffs
        f_scaled = ComponentwiseExponentialFunction(coeffs_scaled)
        
        x_test_scale = randn(3)
        @test f_scaled(x_test_scale) ≈ scale_factor * f(x_test_scale)
        
        grad_scaled = gradientOracle(f_scaled, x_test_scale)
        grad_original = gradientOracle(f, x_test_scale)
        @test grad_scaled ≈ scale_factor * grad_original
    end
end 