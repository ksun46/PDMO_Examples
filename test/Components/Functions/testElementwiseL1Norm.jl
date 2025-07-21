using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "ElementwiseL1Norm Tests" begin
    @testset "Constructor" begin
        # Test default constructor
        f_default = ElementwiseL1Norm()
        @test f_default isa ElementwiseL1Norm
        @test f_default isa AbstractFunction
        @test f_default.coefficient ≈ 1.0
        
        # Test with positive coefficient
        coeff = 2.5
        f_coeff = ElementwiseL1Norm(coeff)
        @test f_coeff.coefficient ≈ coeff
        
        # Test with zero coefficient
        f_zero = ElementwiseL1Norm(0.0)
        @test f_zero.coefficient ≈ 0.0
        
        # Test error for negative coefficient
        @test_throws ErrorException ElementwiseL1Norm(-1.0)
        @test_throws ErrorException ElementwiseL1Norm(-0.1)
    end

    @testset "Function Traits" begin
        @test isProximal(ElementwiseL1Norm) == true
        @test isSmooth(ElementwiseL1Norm) == false
        @test isConvex(ElementwiseL1Norm) == true
        @test isSet(ElementwiseL1Norm) == false
    end

    @testset "Function Evaluation" begin
        # Test with coefficient = 1
        f1 = ElementwiseL1Norm(1.0)
        
        # Test scalar input
        @test f1(3.0) ≈ 3.0
        @test f1(-2.0) ≈ 2.0
        @test f1(0.0) ≈ 0.0
        
        # Test vector input
        x_vec = [1.0, -2.0, 3.0, -4.0]
        expected_vec = norm(x_vec, 1)  # |1| + |-2| + |3| + |-4| = 10
        @test f1(x_vec) ≈ expected_vec
        
        # Test matrix input
        X_mat = [1.0 -2.0; 3.0 -4.0]
        expected_mat = norm(X_mat, 1)  # Sum of absolute values
        @test f1(X_mat) ≈ expected_mat
        
        # Test with different coefficient
        coeff = 2.5
        f_coeff = ElementwiseL1Norm(coeff)
        
        @test f_coeff(x_vec) ≈ coeff * norm(x_vec, 1)
        @test f_coeff(X_mat) ≈ coeff * norm(X_mat, 1)
        
        # Test with zero coefficient
        f_zero = ElementwiseL1Norm(0.0)
        @test f_zero(x_vec) ≈ 0.0
        @test f_zero(X_mat) ≈ 0.0
        
        # Test with sparse input
        X_sparse = sparse([1.0 0.0 -3.0; 0.0 2.0 0.0])
        @test f1(X_sparse) ≈ norm(X_sparse, 1)
    end

    @testset "Proximal Oracle" begin
        # Test soft thresholding for scalar inputs
        f = ElementwiseL1Norm(1.0)
        
        # Test scalar case: soft thresholding
        γ = 0.5
        
        # x > γ: prox(x) = x - γ
        @test proximalOracle(f, 2.0, γ) ≈ 2.0 - γ
        @test proximalOracle(f, 1.0, γ) ≈ 1.0 - γ
        
        # x < -γ: prox(x) = x + γ
        @test proximalOracle(f, -2.0, γ) ≈ -2.0 + γ
        @test proximalOracle(f, -1.0, γ) ≈ -1.0 + γ
        
        # |x| ≤ γ: prox(x) = 0
        @test proximalOracle(f, 0.3, γ) ≈ 0.0
        @test proximalOracle(f, -0.3, γ) ≈ 0.0
        @test proximalOracle(f, 0.0, γ) ≈ 0.0
        
        # Test vector case: elementwise soft thresholding
        x_vec = [2.0, -1.5, 0.3, -0.2, 1.0]
        γ_vec = 0.5
        
        prox_vec = proximalOracle(f, x_vec, γ_vec)
        expected_vec = [1.5, -1.0, 0.0, 0.0, 0.5]  # Soft threshold each element
        @test prox_vec ≈ expected_vec
        @test size(prox_vec) == size(x_vec)
        
        # Test in-place proximal for vectors
        prox_inplace = similar(x_vec)
        proximalOracle!(prox_inplace, f, x_vec, γ_vec)
        @test prox_inplace ≈ expected_vec
        
        # Test matrix case
        X_mat = [2.0 -1.5; 0.3 -0.2]
        prox_mat = proximalOracle(f, X_mat, γ_vec)
        expected_mat = [1.5 -1.0; 0.0 0.0]
        @test prox_mat ≈ expected_mat
        
        # Test in-place proximal for matrices
        prox_mat_inplace = similar(X_mat)
        proximalOracle!(prox_mat_inplace, f, X_mat, γ_vec)
        @test prox_mat_inplace ≈ expected_mat
        
        # Test with different coefficient
        coeff = 2.0
        f_coeff = ElementwiseL1Norm(coeff)
        γ_test = 0.5
        
        # Effective threshold is γ * coefficient
        x_test = 3.0
        expected_prox = x_test - γ_test * coeff  # 3 - 0.5*2 = 2
        @test proximalOracle(f_coeff, x_test, γ_test) ≈ expected_prox
        
        # Test error for scalar in-place
        @test_throws ErrorException proximalOracle!(0.0, f, 5.0, 1.0)
    end

    @testset "Soft Thresholding Properties" begin
        f = ElementwiseL1Norm(1.0)
        γ = 1.0
        
        # Test that soft thresholding reduces magnitude
        x_pos = 3.0
        prox_pos = proximalOracle(f, x_pos, γ)
        @test abs(prox_pos) ≤ abs(x_pos)
        
        x_neg = -3.0
        prox_neg = proximalOracle(f, x_neg, γ)
        @test abs(prox_neg) ≤ abs(x_neg)
        
        # Test that sign is preserved (except when thresholded to zero)
        @test sign(prox_pos) == sign(x_pos)
        @test sign(prox_neg) == sign(x_neg)
        
        # Test threshold boundary cases
        threshold = γ * f.coefficient
        
        # Exactly at threshold
        @test proximalOracle(f, threshold, γ) ≈ 0.0
        @test proximalOracle(f, -threshold, γ) ≈ 0.0
        
        # Just above threshold
        ε = 1e-8  # Use larger epsilon for numerical stability
        @test proximalOracle(f, threshold + ε, γ) ≈ ε atol=1e-10
        @test proximalOracle(f, -threshold - ε, γ) ≈ -ε atol=1e-10
    end

    @testset "Edge Cases" begin
        f = ElementwiseL1Norm(1.0)
        
        # Test with zero gamma
        x = [1.0, -2.0, 3.0]
        prox_zero_gamma = proximalOracle(f, x, 0.0)
        @test prox_zero_gamma ≈ x  # Should be identity
        
        # Test with very large gamma
        γ_large = 1e6
        prox_large_gamma = proximalOracle(f, x, γ_large)
        @test prox_large_gamma ≈ zeros(3)  # Should threshold everything to zero
        
        # Test with very small coefficient
        f_small = ElementwiseL1Norm(1e-10)
        x_test = [1.0, -1.0]
        γ_test = 1.0
        
        # With very small coefficient, threshold is very small
        prox_small = proximalOracle(f_small, x_test, γ_test)
        expected_small = x_test .- γ_test * f_small.coefficient * sign.(x_test)
        @test prox_small ≈ expected_small
        
        # Test with empty arrays
        x_empty = Float64[]
        @test f(x_empty) ≈ 0.0
        @test proximalOracle(f, x_empty, 1.0) ≈ Float64[]
        
        # Test with sparse matrices
        X_sparse = sparse([2.0 0.0 -1.5; 0.0 0.3 0.0])
        prox_sparse = proximalOracle(f, X_sparse, 0.5)
        
        # Check that sparsity pattern is preserved or enhanced
        @test nnz(prox_sparse) ≤ nnz(X_sparse)
        
        # Test with very large values
        x_large = [1e6, -1e6]
        prox_large = proximalOracle(f, x_large, 1.0)
        expected_large = [1e6 - 1.0, -1e6 + 1.0]
        @test prox_large ≈ expected_large
        
        # Test numerical stability
        x_tiny = [1e-15, -1e-15]
        γ_normal = 1e-10
        prox_tiny = proximalOracle(f, x_tiny, γ_normal)
        @test all(isfinite.(prox_tiny))
    end

    @testset "Mathematical Properties" begin
        f = ElementwiseL1Norm(2.0)
        
        # Test that proximal operator is non-expansive
        # ||prox_γf(x) - prox_γf(y)|| ≤ ||x - y||
        x1 = randn(5)
        x2 = randn(5)
        γ = 0.5
        
        prox1 = proximalOracle(f, x1, γ)
        prox2 = proximalOracle(f, x2, γ)
        
        @test norm(prox1 - prox2) ≤ norm(x1 - x2) + 1e-10  # Allow small numerical error
        
        # Test that proximal operator is firmly non-expansive
        # <prox_γf(x) - prox_γf(y), x - y> ≥ ||prox_γf(x) - prox_γf(y)||²
        inner_prod = dot(prox1 - prox2, x1 - x2)
        prox_diff_norm_sq = norm(prox1 - prox2)^2
        
        @test inner_prod ≥ prox_diff_norm_sq - 1e-10
        
        # Test subdifferential property for smooth points
        # For non-smooth functions, we can test at smooth points (where derivative exists)
        x_smooth = [2.0, -3.0, 1.5]  # All elements away from zero
        γ_test = 0.1
        
        prox_result = proximalOracle(f, x_smooth, γ_test)
        
        # At smooth points, prox should satisfy: x - prox = γ * ∂f(prox)
        # For L1 norm, ∂f(x) = coefficient * sign(x) when x ≠ 0
        subgrad = f.coefficient * sign.(prox_result)
        expected_diff = γ_test * subgrad
        actual_diff = x_smooth - prox_result
        
        @test actual_diff ≈ expected_diff atol=1e-10
    end
end 