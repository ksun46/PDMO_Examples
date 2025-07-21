using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "QuadraticFunction Tests" begin
    @testset "Constructor" begin
        # Test basic construction
        n = 3
        Q = sparse(1.0 * I(n))  # Ensure Float64 type
        q = randn(n)
        r = randn()
        
        f = QuadraticFunction(Q, q, r)
        @test f isa QuadraticFunction
        @test f isa AbstractFunction
        @test f.Q == Q
        @test f.q == q
        @test f.r == r
        
        # Test dimension constructor
        f2 = QuadraticFunction(5)
        @test f2 isa QuadraticFunction
        @test size(f2.Q) == (5, 5)
        @test length(f2.q) == 5
        @test f2.r == 0.0
        @test nnz(f2.Q) == 0  # Should be zero matrix
        @test f2.q == zeros(5)
        
        # Test dimension mismatch errors
        Q_wrong = sparse(1.0 * I(3))  # Ensure Float64 type
        q_wrong = randn(4)  # Wrong dimension
        @test_throws ErrorException QuadraticFunction(Q_wrong, q_wrong, 0.0)
        
        # Test non-square matrix error
        Q_nonsquare = sprandn(3, 4, 0.5)
        q_match = randn(3)
        @test_throws ErrorException QuadraticFunction(Q_nonsquare, q_match, 0.0)
    end

    @testset "Function Traits" begin
        @test isSmooth(QuadraticFunction) == true
        @test isConvex(QuadraticFunction) == true
        @test isProximal(QuadraticFunction) == false
        @test isSet(QuadraticFunction) == false
    end

    @testset "Function Evaluation" begin
        # Test simple quadratic: f(x) = x'*I*x + 0*x + 0 = ||x||^2
        n = 3
        Q = sparse(1.0 * I(n))  # Ensure Float64 type
        q = zeros(n)
        r = 0.0
        f = QuadraticFunction(Q, q, r)
        
        x = [1.0, 2.0, 3.0]
        expected = dot(x, x)  # ||x||^2 = 1 + 4 + 9 = 14
        @test f(x) ≈ expected
        
        # Test general quadratic: f(x) = x'*Q*x + q'*x + r
        Q2 = sparse([2.0 1.0; 1.0 3.0])
        q2 = [1.0, -2.0]
        r2 = 5.0
        f2 = QuadraticFunction(Q2, q2, r2)
        
        x2 = [2.0, 1.0]
        # f(x) = [2,1]'*[2 1; 1 3]*[2,1] + [1,-2]'*[2,1] + 5
        #      = [2,1]*[5,5] + [1,-2]*[2,1] + 5
        #      = 10 + 5 + 0 + 5 = 20
        expected2 = dot(Q2 * x2, x2) + dot(q2, x2) + r2
        @test f2(x2) ≈ expected2
        
        # Test with zero vector
        x_zero = zeros(n)
        @test f(x_zero) ≈ r
        
        # Test with negative values
        x_neg = [-1.0, -2.0, -3.0]
        expected_neg = dot(Q * x_neg, x_neg) + dot(q, x_neg) + r
        @test f(x_neg) ≈ expected_neg
    end

    @testset "Gradient Oracle" begin
        # Test gradient: ∇f(x) = (Q + Q')*x + q
        n = 3
        Q = sparse([2.0 1.0 0.0; 0.0 3.0 1.0; 1.0 0.0 2.0])
        q = [1.0, -1.0, 2.0]
        r = 0.0
        f = QuadraticFunction(Q, q, r)
        
        x = [1.0, 2.0, 3.0]
        expected_grad = (Q + Q') * x + q
        
        # Test non-mutating gradient
        grad = gradientOracle(f, x)
        @test grad ≈ expected_grad
        @test size(grad) == size(x)
        
        # Test in-place gradient
        grad_inplace = similar(x)
        gradientOracle!(grad_inplace, f, x)
        @test grad_inplace ≈ expected_grad
        
        # Test gradient at zero
        x_zero = zeros(n)
        grad_zero = gradientOracle(f, x_zero)
        @test grad_zero ≈ q
        
        # Test gradient for symmetric matrix (Q = Q')
        Q_sym = sparse([2.0 1.0; 1.0 3.0])
        q_sym = [1.0, -2.0]
        f_sym = QuadraticFunction(Q_sym, q_sym, 0.0)
        
        x_sym = [2.0, 1.0]
        expected_grad_sym = 2 * Q_sym * x_sym + q_sym  # Since Q = Q', (Q + Q') = 2Q
        grad_sym = gradientOracle(f_sym, x_sym)
        @test grad_sym ≈ expected_grad_sym
    end

    @testset "Edge Cases" begin
        # Test with very small dimensions
        f_small = QuadraticFunction(1)
        x_small = [2.0]
        @test f_small(x_small) ≈ 0.0  # Zero quadratic
        @test gradientOracle(f_small, x_small) ≈ [0.0]
        
        # Test with large sparse matrix
        n_large = 100
        Q_large = sprandn(n_large, n_large, 0.01)  # Very sparse
        q_large = randn(n_large)
        r_large = randn()
        f_large = QuadraticFunction(Q_large, q_large, r_large)
        
        x_large = randn(n_large)
        val_large = f_large(x_large)
        @test val_large isa Float64
        
        grad_large = gradientOracle(f_large, x_large)
        @test length(grad_large) == n_large
        
        # Test with zero matrix and vector
        n = 4
        Q_zero = spzeros(n, n)
        q_zero = zeros(n)
        r_zero = 3.0
        f_zero = QuadraticFunction(Q_zero, q_zero, r_zero)
        
        x = randn(n)
        @test f_zero(x) ≈ r_zero  # Should just return the constant
        @test gradientOracle(f_zero, x) ≈ zeros(n)  # Gradient should be zero
        
        # Test numerical stability with very small/large values
        Q_small = sparse(1e-10 * I(2))  # Ensure Float64 type
        q_small = [1e-10, -1e-10]
        f_small_vals = QuadraticFunction(Q_small, q_small, 1e-10)
        
        x_test = [1e5, -1e5]
        val_test = f_small_vals(x_test)
        @test isfinite(val_test)
        
        grad_test = gradientOracle(f_small_vals, x_test)
        @test all(isfinite.(grad_test))
    end

    @testset "Mathematical Properties" begin
        # Test that gradient is correct by finite differences
        n = 3
        Q = sparse(randn(n, n))
        Q = Q + Q'  # Make symmetric for easier testing
        q = randn(n)
        r = randn()
        f = QuadraticFunction(Q, q, r)
        
        x = randn(n)
        grad_analytical = gradientOracle(f, x)
        
        # Finite difference approximation
        h = 1e-8
        grad_numerical = similar(x)
        for i in 1:n
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            grad_numerical[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        end
        
        @test grad_analytical ≈ grad_numerical atol=1e-6
        
        # Test convexity property: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
        # For quadratic functions with PSD Q, this should hold
        Q_psd = sparse(1.0 * I(n))  # Identity is PSD, ensure Float64 type
        f_convex = QuadraticFunction(Q_psd, q, r)
        
        x1 = randn(n)
        x2 = randn(n)
        λ = 0.3
        
        x_combo = λ * x1 + (1 - λ) * x2
        f_combo = f_convex(x_combo)
        f_convex_combo = λ * f_convex(x1) + (1 - λ) * f_convex(x2)
        
        @test f_combo ≤ f_convex_combo + 1e-10  # Allow small numerical error
    end
end 