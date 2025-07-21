using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "AffineFunction Tests" begin
    @testset "Constructor" begin
        # Test vector coefficient construction
        A = randn(5)
        r = randn()
        f = AffineFunction(A, r)
        @test f isa AffineFunction
        @test f isa AbstractFunction
        @test f.A == A
        @test f.r == r
        
        # Test scalar coefficient construction
        A_scalar = 3.0
        r_scalar = -2.0
        f_scalar = AffineFunction(A_scalar, r_scalar)
        @test f_scalar.A == A_scalar
        @test f_scalar.r == r_scalar
        
        # Test matrix coefficient construction
        A_matrix = randn(3, 4)
        f_matrix = AffineFunction(A_matrix, 1.0)
        @test f_matrix.A == A_matrix
        @test f_matrix.r == 1.0
        
        # Test default r value
        f_default = AffineFunction(A)
        @test f_default.A == A
        @test f_default.r == 0.0
    end

    @testset "Function Traits" begin
        @test isProximal(AffineFunction) == true
        @test isSmooth(AffineFunction) == true
        @test isConvex(AffineFunction) == true
        @test isSet(AffineFunction) == false
    end

    @testset "Function Evaluation" begin
        # Test scalar case: f(x) = A*x + r
        A_scalar = 2.0
        r_scalar = 3.0
        f_scalar = AffineFunction(A_scalar, r_scalar)
        
        x_scalar = 5.0
        expected_scalar = A_scalar * x_scalar + r_scalar  # 2*5 + 3 = 13
        @test f_scalar(x_scalar) ≈ expected_scalar
        
        # Test vector case: f(x) = A'*x + r (dot product)
        A_vec = [1.0, 2.0, 3.0]
        r_vec = -1.0
        f_vec = AffineFunction(A_vec, r_vec)
        
        x_vec = [2.0, 1.0, 4.0]
        expected_vec = dot(A_vec, x_vec) + r_vec  # 1*2 + 2*1 + 3*4 - 1 = 15
        @test f_vec(x_vec) ≈ expected_vec
        
        # Test matrix case: f(X) = <A, X> + r (Frobenius inner product)
        A_mat = [1.0 2.0; 3.0 4.0]
        r_mat = 2.0
        f_mat = AffineFunction(A_mat, r_mat)
        
        X_mat = [2.0 1.0; 0.0 3.0]
        expected_mat = dot(A_mat, X_mat) + r_mat  # 1*2 + 2*1 + 3*0 + 4*3 + 2 = 18
        @test f_mat(X_mat) ≈ expected_mat
        
        # Test zero cases
        @test f_vec(zeros(3)) ≈ r_vec
        @test f_mat(zeros(2, 2)) ≈ r_mat
    end

    @testset "Gradient Oracle" begin
        # Test scalar case: gradient is just A
        A_scalar = 3.0
        f_scalar = AffineFunction(A_scalar, 1.0)
        
        @test gradientOracle(f_scalar, 5.0) ≈ A_scalar
        @test gradientOracle(f_scalar, -2.0) ≈ A_scalar
        
        # Test vector case: gradient is A
        A_vec = [1.0, -2.0, 3.0]
        f_vec = AffineFunction(A_vec, 0.0)
        
        x_vec = randn(3)
        grad_vec = gradientOracle(f_vec, x_vec)
        @test grad_vec ≈ A_vec
        @test size(grad_vec) == size(x_vec)
        
        # Test in-place gradient for vectors
        grad_inplace = similar(x_vec)
        gradientOracle!(grad_inplace, f_vec, x_vec)
        @test grad_inplace ≈ A_vec
        
        # Test matrix case: gradient is A
        A_mat = randn(2, 3)
        f_mat = AffineFunction(A_mat, 0.0)
        
        X_mat = randn(2, 3)
        grad_mat = gradientOracle(f_mat, X_mat)
        @test grad_mat ≈ A_mat
        @test size(grad_mat) == size(X_mat)
        
        # Test in-place gradient for matrices
        grad_mat_inplace = similar(X_mat)
        gradientOracle!(grad_mat_inplace, f_mat, X_mat)
        @test grad_mat_inplace ≈ A_mat
        
        # Test error for scalar in-place
        @test_throws ErrorException gradientOracle!(0.0, f_scalar, 5.0)
    end

    @testset "Proximal Oracle" begin
        # Test scalar case: prox_γf(x) = x - γ*A
        A_scalar = 2.0
        f_scalar = AffineFunction(A_scalar, 1.0)
        
        x_scalar = 5.0
        γ = 0.5
        expected_prox_scalar = x_scalar - γ * A_scalar  # 5 - 0.5*2 = 4
        @test proximalOracle(f_scalar, x_scalar, γ) ≈ expected_prox_scalar
        
        # Test vector case: prox_γf(x) = x - γ*A
        A_vec = [1.0, -1.0, 2.0]
        f_vec = AffineFunction(A_vec, 0.0)
        
        x_vec = [3.0, 2.0, 1.0]
        γ_vec = 0.2
        expected_prox_vec = x_vec - γ_vec * A_vec
        prox_vec = proximalOracle(f_vec, x_vec, γ_vec)
        @test prox_vec ≈ expected_prox_vec
        @test size(prox_vec) == size(x_vec)
        
        # Test in-place proximal for vectors
        prox_inplace = similar(x_vec)
        proximalOracle!(prox_inplace, f_vec, x_vec, γ_vec)
        @test prox_inplace ≈ expected_prox_vec
        
        # Test matrix case
        A_mat = [1.0 0.0; -1.0 2.0]
        f_mat = AffineFunction(A_mat, 0.0)
        
        X_mat = [2.0 3.0; 1.0 4.0]
        γ_mat = 0.1
        expected_prox_mat = X_mat - γ_mat * A_mat
        prox_mat = proximalOracle(f_mat, X_mat, γ_mat)
        @test prox_mat ≈ expected_prox_mat
        
        # Test in-place proximal for matrices
        prox_mat_inplace = similar(X_mat)
        proximalOracle!(prox_mat_inplace, f_mat, X_mat, γ_mat)
        @test prox_mat_inplace ≈ expected_prox_mat
        
        # Test error for negative gamma
        @test_throws ErrorException proximalOracle(f_vec, x_vec, -0.1)
        @test_throws ErrorException proximalOracle!(prox_inplace, f_vec, x_vec, -0.1)
        @test_throws ErrorException proximalOracle(f_scalar, x_scalar, -0.1)
        
        # Test error for scalar in-place
        @test_throws ErrorException proximalOracle!(0.0, f_scalar, 5.0, 1.0)
    end

    @testset "Edge Cases" begin
        # Test with zero coefficient
        A_zero = zeros(3)
        f_zero = AffineFunction(A_zero, 5.0)
        
        x = randn(3)
        @test f_zero(x) ≈ 5.0  # Should just return r
        @test gradientOracle(f_zero, x) ≈ A_zero
        @test proximalOracle(f_zero, x, 1.0) ≈ x  # prox should be identity
        
        # Test with zero r
        A_nonzero = [1.0, 2.0]
        f_zero_r = AffineFunction(A_nonzero, 0.0)
        
        x_test = [3.0, 4.0]
        @test f_zero_r(x_test) ≈ dot(A_nonzero, x_test)
        
        # Test with very large values
        A_large = [1e6, -1e6]
        f_large = AffineFunction(A_large, 1e6)
        
        x_large = [1e-6, 1e-6]
        val_large = f_large(x_large)
        @test isfinite(val_large)
        
        grad_large = gradientOracle(f_large, x_large)
        @test grad_large ≈ A_large
        
        # Test with very small gamma
        γ_small = 1e-10
        prox_small = proximalOracle(f_large, x_large, γ_small)
        expected_small = x_large - γ_small * A_large
        @test prox_small ≈ expected_small
        
        # Test with sparse matrices
        A_sparse = sprandn(5, 5, 0.3)
        f_sparse = AffineFunction(A_sparse, 0.0)
        
        X_sparse = sprandn(5, 5, 0.3)
        val_sparse = f_sparse(X_sparse)
        @test val_sparse isa Float64
        
        grad_sparse = gradientOracle(f_sparse, X_sparse)
        @test grad_sparse ≈ A_sparse
        
        # Test with empty arrays
        A_empty = Float64[]
        f_empty = AffineFunction(A_empty, 2.0)
        
        x_empty = Float64[]
        @test f_empty(x_empty) ≈ 2.0
        @test gradientOracle(f_empty, x_empty) ≈ A_empty
        @test proximalOracle(f_empty, x_empty, 1.0) ≈ x_empty
    end

    @testset "Mathematical Properties" begin
        # Test linearity: f(αx + βy) = αf(x) + βf(y) - (α+β-1)r
        # For affine functions: f(x) = A'x + r
        A = [1.0, 2.0, -1.0]
        r = 3.0
        f = AffineFunction(A, r)
        
        x1 = [1.0, 2.0, 3.0]
        x2 = [2.0, -1.0, 1.0]
        α = 0.3
        β = 0.7
        
        # For affine functions, we have: f(αx + βy) = αf(x) + βf(y) - (α+β-1)r
        combo = α * x1 + β * x2
        f_combo = f(combo)
        f_linear_combo = α * f(x1) + β * f(x2) - (α + β - 1) * r
        @test f_combo ≈ f_linear_combo
        
        # Test gradient consistency with finite differences
        h = 1e-8
        x = randn(3)
        grad_analytical = gradientOracle(f, x)
        
        grad_numerical = similar(x)
        for i in 1:3
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            grad_numerical[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        end
        
        @test grad_analytical ≈ grad_numerical atol=1e-6
        
        # Test proximal operator property
        # For affine functions, prox_γf(x) = x - γ∇f(x) = x - γA
        x_test = randn(3)
        γ_test = 0.1
        
        prox_result = proximalOracle(f, x_test, γ_test)
        expected_prox = x_test - γ_test * gradientOracle(f, x_test)
        @test prox_result ≈ expected_prox
    end
end 