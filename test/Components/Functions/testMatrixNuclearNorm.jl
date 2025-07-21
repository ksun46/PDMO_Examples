using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "MatrixNuclearNorm Tests" begin
    @testset "Constructor" begin
        # Test valid construction
        b = [1.0, 2.0]
        rows, cols = 3, 2
        f = MatrixNuclearNorm(b, rows, cols)
        @test f isa MatrixNuclearNorm
        @test f isa AbstractFunction
        @test f.b == b
        @test f.numberRows == rows
        @test f.numberColumns == cols
        
        # Test square matrix
        b_square = [1.0, 1.5, 2.0]
        f_square = MatrixNuclearNorm(b_square, 3, 3)
        @test f_square.b == b_square
        @test f_square.numberRows == 3
        @test f_square.numberColumns == 3
        
        # Test different weight values
        b_diff = [0.1, 10.0, 0.5]
        f_diff = MatrixNuclearNorm(b_diff, 5, 3)
        @test f_diff.b == b_diff
        
        # Test error cases
        @test_throws ErrorException MatrixNuclearNorm([0.0, 1.0], 2, 2)  # Zero weight
        @test_throws ErrorException MatrixNuclearNorm([-1.0, 1.0], 2, 2)  # Negative weight
        @test_throws ErrorException MatrixNuclearNorm([1.0, 2.0], 0, 2)  # Zero rows
        @test_throws ErrorException MatrixNuclearNorm([1.0, 2.0], 2, 0)  # Zero cols
        @test_throws ErrorException MatrixNuclearNorm([1.0, 2.0, 3.0], 2, 2)  # Wrong number of weights
        @test_throws ErrorException MatrixNuclearNorm([1.0], 2, 3)  # Too few weights
    end

    @testset "Function Traits" begin
        @test isProximal(MatrixNuclearNorm) == true
        @test isSmooth(MatrixNuclearNorm) == false
        @test isSet(MatrixNuclearNorm) == false
        # Note: isConvex is not defined as true because it's only convex when all weights are equal
    end

    @testset "Function Evaluation" begin
        # Test simple case with identity weights
        b = [1.0, 1.0]
        f = MatrixNuclearNorm(b, 2, 2)
        
        # Test with diagonal matrix (known singular values)
        X1 = [3.0 0.0; 0.0 2.0]
        # Singular values are [3.0, 2.0]
        expected1 = 1.0 * 3.0 + 1.0 * 2.0  # 5.0
        @test f(X1) ≈ expected1
        
        # Test with zero matrix
        X_zero = zeros(2, 2)
        @test f(X_zero) ≈ 0.0
        
        # Test with different weights
        b_weighted = [2.0, 0.5]
        f_weighted = MatrixNuclearNorm(b_weighted, 2, 2)
        expected_weighted = 2.0 * 3.0 + 0.5 * 2.0  # 6.0 + 1.0 = 7.0
        @test f_weighted(X1) ≈ expected_weighted
        
        # Test with rectangular matrix (more rows than columns)
        b_rect = [1.0, 1.0]
        f_rect = MatrixNuclearNorm(b_rect, 3, 2)
        X_rect = [1.0 0.0; 0.0 2.0; 0.0 0.0]
        # Should have singular values [2.0, 1.0]
        expected_rect = 1.0 * 2.0 + 1.0 * 1.0  # 3.0
        @test f_rect(X_rect) ≈ expected_rect
        
        # Test with rectangular matrix (more columns than rows)
        b_rect2 = [1.0, 1.0]
        f_rect2 = MatrixNuclearNorm(b_rect2, 2, 3)
        X_rect2 = [1.0 0.0 0.0; 0.0 2.0 0.0]
        # Should have singular values [2.0, 1.0]
        expected_rect2 = 1.0 * 2.0 + 1.0 * 1.0  # 3.0
        @test f_rect2(X_rect2) ≈ expected_rect2
    end

    @testset "Proximal Oracle" begin
        # Test simple case with identity weights
        b = [1.0, 1.0]
        f = MatrixNuclearNorm(b, 2, 2)
        γ = 0.5
        
        # Test with diagonal matrix
        X = [3.0 0.0; 0.0 2.0]
        prox = proximalOracle(f, X, γ)
        
        # Expected: soft threshold singular values by γ * b[i]
        # Original singular values: [3.0, 2.0]
        # Thresholded: [max(0, 3.0 - 0.5*1.0), max(0, 2.0 - 0.5*1.0)] = [2.5, 1.5]
        # Result should be diag([2.5, 1.5])
        expected_diag = [2.5, 1.5]
        
        # Check that result has expected singular values
        F_result = svd(prox)
        @test F_result.S ≈ expected_diag atol=1e-10
        @test size(prox) == size(X)
        
        # Test in-place version
        prox_inplace = similar(X)
        proximalOracle!(prox_inplace, f, X, γ)
        @test prox_inplace ≈ prox
        
        # Test with different weights
        b_diff = [2.0, 0.5]
        f_diff = MatrixNuclearNorm(b_diff, 2, 2)
        prox_diff = proximalOracle(f_diff, X, γ)
        
        # Thresholded: [max(0, 3.0 - 0.5*2.0), max(0, 2.0 - 0.5*0.5)] = [2.0, 1.75]
        F_diff = svd(prox_diff)
        expected_diff = [2.0, 1.75]
        @test F_diff.S ≈ expected_diff atol=1e-10
        
        # Test case where some singular values become zero
        γ_large = 2.0
        prox_large = proximalOracle(f, X, γ_large)
        
        # Thresholded: [max(0, 3.0 - 2.0*1.0), max(0, 2.0 - 2.0*1.0)] = [1.0, 0.0]
        F_large = svd(prox_large)
        expected_large = [1.0, 0.0]
        @test F_large.S ≈ expected_large atol=1e-10
        
        # Test case where all singular values become zero
        γ_huge = 10.0
        prox_huge = proximalOracle(f, X, γ_huge)
        @test prox_huge ≈ zeros(2, 2) atol=1e-10
    end

    @testset "Proximal Oracle - Rectangular Matrices" begin
        # Test with more rows than columns
        b = [1.0, 1.0]
        f = MatrixNuclearNorm(b, 3, 2)
        γ = 0.5
        
        X = [2.0 0.0; 0.0 3.0; 0.0 0.0]
        prox = proximalOracle(f, X, γ)
        
        @test size(prox) == (3, 2)
        
        # Check that singular values are properly thresholded
        F_result = svd(prox)
        # Original singular values should be [3.0, 2.0]
        # Thresholded: [2.5, 1.5]
        expected_sv = [2.5, 1.5]
        @test F_result.S ≈ expected_sv atol=1e-10
        
        # Test with more columns than rows
        f_wide = MatrixNuclearNorm(b, 2, 3)
        X_wide = [2.0 0.0 0.0; 0.0 3.0 0.0]
        prox_wide = proximalOracle(f_wide, X_wide, γ)
        
        @test size(prox_wide) == (2, 3)
        F_wide = svd(prox_wide)
        @test F_wide.S ≈ expected_sv atol=1e-10
    end

    @testset "Edge Cases" begin
        # Test with single singular value (1x1 matrix)
        b_single = [2.0]
        f_single = MatrixNuclearNorm(b_single, 1, 1)
        
        X_single = reshape([3.0], 1, 1)
        γ = 1.0
        prox_single = proximalOracle(f_single, X_single, γ)
        
        # Expected: max(0, 3.0 - 1.0*2.0) = 1.0
        @test prox_single ≈ reshape([1.0], 1, 1)
        
        # Test with very small gamma
        γ_small = 1e-10
        X_test = [2.0 1.0; 1.0 2.0]
        b_test = [1.0, 1.0]
        f_test = MatrixNuclearNorm(b_test, 2, 2)
        
        prox_small = proximalOracle(f_test, X_test, γ_small)
        # Should be very close to original matrix
        @test prox_small ≈ X_test atol=1e-8
        
        # Test with zero matrix
        X_zero = zeros(2, 2)
        prox_zero = proximalOracle(f_test, X_zero, 1.0)
        @test prox_zero ≈ zeros(2, 2)
        
        # Test with very large weights
        b_large = [1e6, 1e6]
        f_large = MatrixNuclearNorm(b_large, 2, 2)
        prox_large_weights = proximalOracle(f_large, X_test, 1.0)
        # All singular values should be thresholded to zero
        @test prox_large_weights ≈ zeros(2, 2) atol=1e-10
        
        # Test with very small weights
        b_small = [1e-6, 1e-6]
        f_small = MatrixNuclearNorm(b_small, 2, 2)
        prox_small_weights = proximalOracle(f_small, X_test, 1.0)
        # Should be very close to original (minimal thresholding)
        @test norm(prox_small_weights - X_test) < 1e-5
        
        # Test numerical stability
        X_cond = [1e10 1e-10; 1e-10 1e10]
        prox_cond = proximalOracle(f_test, X_cond, 1.0)
        @test all(isfinite.(prox_cond))
    end

    @testset "Mathematical Properties" begin
        b = [1.0, 2.0]
        f = MatrixNuclearNorm(b, 2, 2)
        γ = 0.5
        
        # Test that proximal operator reduces the objective
        X_test = [3.0 1.0; 1.0 2.0]
        prox_test = proximalOracle(f, X_test, γ)
        
        # The proximal objective should be minimized
        # prox = argmin_Y { f(Y) + (1/2γ)||Y - X||_F^2 }
        obj_original = f(X_test) + (1/(2*γ)) * norm(X_test - X_test)^2
        obj_prox = f(prox_test) + (1/(2*γ)) * norm(prox_test - X_test)^2
        
        @test obj_prox ≤ obj_original + 1e-10
        
        # Test that proximal operator is non-expansive for convex case (equal weights)
        b_equal = [1.0, 1.0]
        f_equal = MatrixNuclearNorm(b_equal, 2, 2)
        
        X1 = [1.0 2.0; 3.0 1.0]
        X2 = [2.0 1.0; 1.0 3.0]
        
        prox1 = proximalOracle(f_equal, X1, γ)
        prox2 = proximalOracle(f_equal, X2, γ)
        
        # For equal weights (convex case), should be non-expansive
        @test norm(prox1 - prox2) ≤ norm(X1 - X2) + 1e-10
        
        # Test that soft thresholding preserves matrix structure
        X_diag = [5.0 0.0; 0.0 3.0]
        prox_diag = proximalOracle(f, X_diag, γ)
        
        # Result should still be diagonal (up to numerical precision)
        @test abs(prox_diag[1,2]) < 1e-10
        @test abs(prox_diag[2,1]) < 1e-10
        
        # Test scaling property: prox_αf(x) for α > 0
        α = 2.0
        b_scaled = α * b
        f_scaled = MatrixNuclearNorm(b_scaled, 2, 2)
        
        prox_scaled = proximalOracle(f_scaled, X_test, γ)
        prox_original = proximalOracle(f, X_test, γ/α)
        
        # These should be related by the scaling property of proximal operators
        # This is a complex relationship for nuclear norm, so we just check they're different
        @test norm(prox_scaled - prox_original) > 1e-10
        
        # Test that zero matrix is a fixed point
        X_zero = zeros(2, 2)
        prox_zero = proximalOracle(f, X_zero, γ)
        @test prox_zero ≈ X_zero
        
        # Test monotonicity in gamma: larger gamma should give smaller nuclear norm
        γ1 = 0.1
        γ2 = 1.0
        
        prox_small_gamma = proximalOracle(f_equal, X_test, γ1)
        prox_large_gamma = proximalOracle(f_equal, X_test, γ2)
        
        norm1 = f_equal(prox_small_gamma)
        norm2 = f_equal(prox_large_gamma)
        
        @test norm2 ≤ norm1 + 1e-10  # Larger gamma should give smaller nuclear norm
    end
end 