using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "FrobeniusNormSquare Tests" begin
    @testset "Constructor" begin
        # Test vector case construction
        A = randn(5, 3)
        b = randn(5)
        f_vec = FrobeniusNormSquare(A, b, 3, 1, 1.0)
        @test f_vec isa FrobeniusNormSquare
        @test f_vec isa AbstractFunction
        @test f_vec.A == A
        @test f_vec.b == b
        @test f_vec.numberRows == 3
        @test f_vec.numberColumns == 1
        @test f_vec.coe == 1.0
        @test f_vec.isVectorProblem == true
        
        # Test matrix case construction
        A_mat = randn(4, 2)
        b_mat = randn(4, 3)
        f_mat = FrobeniusNormSquare(A_mat, b_mat, 2, 3, 2.0)
        @test f_mat.A == A_mat
        @test f_mat.b == b_mat
        @test f_mat.numberRows == 2
        @test f_mat.numberColumns == 3
        @test f_mat.coe == 2.0
        @test f_mat.isVectorProblem == false
        
        # Test default coefficient
        f_default = FrobeniusNormSquare(A, b, 3, 1)
        @test f_default.coe == 0.5
        
        # Test error cases
        @test_throws ErrorException FrobeniusNormSquare(A, b, 3, 1, -1.0)  # Negative coefficient
        @test_throws ErrorException FrobeniusNormSquare(A, randn(4), 3, 1, 1.0)  # Dimension mismatch
        @test_throws ErrorException FrobeniusNormSquare(A, b, 2, 1, 1.0)  # Wrong numberRows
        @test_throws ErrorException FrobeniusNormSquare(A, b, 3, 2, 1.0)  # Wrong numberColumns for vector
        
        # Matrix dimension errors
        @test_throws ErrorException FrobeniusNormSquare(A_mat, randn(3, 3), 2, 3, 1.0)  # Row mismatch
        @test_throws ErrorException FrobeniusNormSquare(A_mat, randn(4, 2), 2, 3, 1.0)  # Column mismatch
    end

    @testset "Function Traits" begin
        @test isSmooth(FrobeniusNormSquare) == true
        @test isProximal(FrobeniusNormSquare) == true
        @test isConvex(FrobeniusNormSquare) == true
        @test isSet(FrobeniusNormSquare) == false
    end

    @testset "Function Evaluation - Vector Case" begin
        # Simple case: f(x) = ||Ax - b||_F^2
        A = [1.0 0.0; 0.0 1.0; 1.0 1.0]  # 3x2 matrix
        b = [1.0, 2.0, 3.0]
        f = FrobeniusNormSquare(A, b, 2, 1, 1.0)
        
        x = [1.0, 2.0]
        residual = A * x - b  # [0, 0, 0]
        expected = dot(residual, residual)
        @test f(x) ≈ expected
        
        # Test with different coefficient
        f_coeff = FrobeniusNormSquare(A, b, 2, 1, 2.0)
        @test f_coeff(x) ≈ 2.0 * expected
        
        # Test with non-zero residual
        x2 = [0.0, 0.0]
        residual2 = A * x2 - b  # [-1, -2, -3]
        expected2 = dot(residual2, residual2)  # 1 + 4 + 9 = 14
        @test f(x2) ≈ expected2
    end

    @testset "Function Evaluation - Matrix Case" begin
        # Matrix case: f(X) = ||AX - B||_F^2
        A = [1.0 0.0; 0.0 1.0]  # 2x2 identity
        B = [1.0 2.0; 3.0 4.0]  # 2x2 target
        f = FrobeniusNormSquare(A, B, 2, 2, 1.0)
        
        X = [1.0 2.0; 3.0 4.0]  # Should give zero residual
        @test f(X) ≈ 0.0
        
        X2 = zeros(2, 2)
        residual = A * X2 - B  # -B
        expected = norm(residual)^2  # Use default norm which is Frobenius for matrices
        @test f(X2) ≈ expected
    end

    @testset "Gradient Oracle - Vector Case" begin
        A = randn(4, 3)
        b = randn(4)
        coe = 1.5
        f = FrobeniusNormSquare(A, b, 3, 1, coe)
        
        x = randn(3)
        
        # Analytical gradient: 2 * coe * A'(Ax - b)
        residual = A * x - b
        expected_grad = 2 * coe * (A' * residual)
        
        # Test non-mutating gradient
        grad = gradientOracle(f, x)
        @test grad ≈ expected_grad
        @test size(grad) == size(x)
        
        # Test in-place gradient
        grad_inplace = similar(x)
        gradientOracle!(grad_inplace, f, x)
        @test grad_inplace ≈ expected_grad
        
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
    end

    @testset "Gradient Oracle - Matrix Case" begin
        A = randn(3, 2)
        B = randn(3, 4)
        coe = 0.8
        f = FrobeniusNormSquare(A, B, 2, 4, coe)
        
        X = randn(2, 4)
        
        # Analytical gradient: 2 * coe * A'(AX - B)
        residual = A * X - B
        expected_grad = 2 * coe * (A' * residual)
        
        grad = gradientOracle(f, X)
        @test grad ≈ expected_grad
        @test size(grad) == size(X)
        
        # Test in-place gradient
        grad_inplace = similar(X)
        gradientOracle!(grad_inplace, f, X)
        @test grad_inplace ≈ expected_grad
    end

    @testset "Proximal Oracle - Vector Case" begin
        # Simple case where we can verify the solution
        A = [1.0 0.0; 0.0 1.0]  # 2x2 identity
        b = [1.0, 2.0]
        coe = 1.0
        f = FrobeniusNormSquare(A, b, 2, 1, coe)
        
        x = [0.0, 0.0]
        γ = 1.0
        
        # For this case, the proximal operator solves:
        # (I + 2γcoe * A'A)y = x + 2γcoe * A'b
        # (I + 2*1*1 * I)y = [0,0] + 2*1*1 * [1,2]
        # 3*I*y = [2,4]
        # y = [2/3, 4/3]
        expected = [2.0/3.0, 4.0/3.0]
        
        # Test non-mutating proximal
        prox = proximalOracle(f, x, γ)
        @test prox ≈ expected
        @test size(prox) == size(x)
        
        # Test in-place proximal
        prox_inplace = similar(x)
        proximalOracle!(prox_inplace, f, x, γ)
        @test prox_inplace ≈ expected
        
        # Test error for negative gamma
        @test_throws AssertionError proximalOracle(f, x, -1.0)
    end

    @testset "Proximal Oracle - Matrix Case" begin
        # Simple matrix case
        A = Matrix(1.0I, 2, 2)  # Identity matrix
        B = ones(2, 2)
        coe = 0.5
        f = FrobeniusNormSquare(A, B, 2, 2, coe)
        
        X = zeros(2, 2)
        γ = 1.0
        
        prox = proximalOracle(f, X, γ)
        @test size(prox) == size(X)
        
        # Verify that the proximal operator reduces the objective
        obj_original = f(X)
        obj_prox = f(prox)
        @test obj_prox ≤ obj_original
        
        # Test in-place version
        prox_inplace = similar(X)
        proximalOracle!(prox_inplace, f, X, γ)
        @test prox_inplace ≈ prox
    end

    @testset "Edge Cases" begin
        # Test with very small dimensions
        A_small = reshape([1.0], 1, 1)
        b_small = [2.0]
        f_small = FrobeniusNormSquare(A_small, b_small, 1, 1, 1.0)
        
        x_small = [1.0]
        @test f_small(x_small) ≈ 1.0  # (1*1 - 2)^2 = 1
        
        grad_small = gradientOracle(f_small, x_small)
        @test length(grad_small) == 1
        
        prox_small = proximalOracle(f_small, x_small, 1.0)
        @test length(prox_small) == 1
        
        # Test with zero coefficient (minimum allowed)
        A = randn(3, 2)
        b = randn(3)
        f_min_coe = FrobeniusNormSquare(A, b, 2, 1, 1e-10)
        
        x = randn(2)
        val = f_min_coe(x)
        @test val ≥ 0.0
        @test isfinite(val)
        
        # Test with large matrices
        A_large = randn(50, 20)
        b_large = randn(50)
        f_large = FrobeniusNormSquare(A_large, b_large, 20, 1, 1.0)
        
        x_large = randn(20)
        val_large = f_large(x_large)
        @test isfinite(val_large)
        
        grad_large = gradientOracle(f_large, x_large)
        @test length(grad_large) == 20
        @test all(isfinite.(grad_large))
        
        # Test numerical stability
        A_cond = [1e6 0.0; 0.0 1e-6]  # Ill-conditioned matrix
        b_cond = [1.0, 1.0]
        f_cond = FrobeniusNormSquare(A_cond, b_cond, 2, 1, 1.0)
        
        x_cond = [1e-6, 1e6]
        val_cond = f_cond(x_cond)
        @test isfinite(val_cond)
        
        prox_cond = proximalOracle(f_cond, x_cond, 1.0)
        @test all(isfinite.(prox_cond))
    end

    @testset "Mathematical Properties" begin
        A = randn(4, 3)
        b = randn(4)
        coe = 2.0
        f = FrobeniusNormSquare(A, b, 3, 1, coe)
        
        # Test convexity: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
        x1 = randn(3)
        x2 = randn(3)
        λ = 0.3
        
        x_combo = λ * x1 + (1 - λ) * x2
        f_combo = f(x_combo)
        f_convex_combo = λ * f(x1) + (1 - λ) * f(x2)
        
        @test f_combo ≤ f_convex_combo + 1e-10
        
        # Test that proximal operator is a contraction
        # ||prox_γf(x) - prox_γf(y)|| ≤ ||x - y||
        γ = 0.5
        prox1 = proximalOracle(f, x1, γ)
        prox2 = proximalOracle(f, x2, γ)
        
        @test norm(prox1 - prox2) ≤ norm(x1 - x2) + 1e-10
        
        # Test proximal operator optimality condition
        # For the proximal operator, we should have:
        # prox = argmin_y { f(y) + (1/2γ)||y - x||^2 }
        x_test = randn(3)
        γ_test = 1.0
        prox_test = proximalOracle(f, x_test, γ_test)
        
        # The gradient of the proximal objective at the solution should be zero
        # ∇f(prox) + (1/γ)(prox - x) = 0
        grad_at_prox = gradientOracle(f, prox_test)
        expected_zero = grad_at_prox + (1/γ_test) * (prox_test - x_test)
        @test norm(expected_zero) < 1e-6
    end
end 