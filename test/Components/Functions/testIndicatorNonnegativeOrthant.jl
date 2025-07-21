using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "IndicatorNonnegativeOrthant Tests" begin
    @testset "Constructor" begin
        # Test construction (no parameters needed)
        f = IndicatorNonnegativeOrthant()
        @test f isa IndicatorNonnegativeOrthant
        @test f isa AbstractFunction
        
        # Test multiple instances
        f1 = IndicatorNonnegativeOrthant()
        f2 = IndicatorNonnegativeOrthant()
        @test f1 isa IndicatorNonnegativeOrthant
        @test f2 isa IndicatorNonnegativeOrthant
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorNonnegativeOrthant) == true
        @test isConvex(IndicatorNonnegativeOrthant) == true
        @test isSet(IndicatorNonnegativeOrthant) == true
        @test isSmooth(IndicatorNonnegativeOrthant) == false
    end

    @testset "Function Evaluation - Scalar" begin
        f = IndicatorNonnegativeOrthant()
        
        # Inside the nonnegative orthant
        @test f(0.0) ≈ 0.0
        @test f(1.0) ≈ 0.0
        @test f(10.0) ≈ 0.0
        @test f(1e-10) ≈ 0.0
        
        # Test with FeasTolerance
        ε = FeasTolerance / 2
        @test f(-ε) ≈ 0.0  # Should be feasible due to tolerance
        
        # Outside the nonnegative orthant
        @test f(-1.0) == Inf
        @test f(-0.1) == Inf
        @test f(-1e-5) == Inf  # Clearly outside tolerance
    end

    @testset "Function Evaluation - Vector" begin
        f = IndicatorNonnegativeOrthant()
        
        # All components nonnegative
        x1 = [1.0, 2.0, 3.0]
        @test f(x1) ≈ 0.0
        
        x2 = [0.0, 0.0, 0.0]
        @test f(x2) ≈ 0.0
        
        x3 = [1e-10, 1e10, 0.0]
        @test f(x3) ≈ 0.0
        
        # Some components negative
        x4 = [1.0, -1.0, 2.0]
        @test f(x4) == Inf
        
        x5 = [-1.0, 2.0, 3.0]
        @test f(x5) == Inf
        
        x6 = [1.0, 2.0, -1e-5]  # Clearly outside tolerance
        @test f(x6) == Inf
        
        # Test with FeasTolerance
        ε = FeasTolerance / 2
        x7 = [1.0, -ε, 2.0]  # Should be feasible due to tolerance
        @test f(x7) ≈ 0.0
    end

    @testset "Function Evaluation - Matrix" begin
        f = IndicatorNonnegativeOrthant()
        
        # All elements nonnegative
        X1 = [1.0 2.0; 3.0 4.0]
        @test f(X1) ≈ 0.0
        
        X2 = [0.0 0.0; 0.0 0.0]
        @test f(X2) ≈ 0.0
        
        # Some elements negative
        X3 = [1.0 -1.0; 2.0 3.0]
        @test f(X3) == Inf
        
        X4 = [-1.0 2.0; 3.0 4.0]
        @test f(X4) == Inf
    end

    @testset "Proximal Oracle - Scalar" begin
        f = IndicatorNonnegativeOrthant()
        
        # Positive values - should remain unchanged
        @test proximalOracle(f, 1.0) ≈ 1.0
        @test proximalOracle(f, 10.0) ≈ 10.0
        @test proximalOracle(f, 1e-10) ≈ 1e-10
        
        # Zero - should remain zero
        @test proximalOracle(f, 0.0) ≈ 0.0
        
        # Negative values - should be projected to zero
        @test proximalOracle(f, -1.0) ≈ 0.0
        @test proximalOracle(f, -10.0) ≈ 0.0
        @test proximalOracle(f, -1e-10) ≈ 0.0
        
        # Test gamma parameter doesn't affect projection (indicator functions)
        @test proximalOracle(f, -5.0, 0.1) ≈ 0.0
        @test proximalOracle(f, -5.0, 10.0) ≈ 0.0
        @test proximalOracle(f, 3.0, 0.1) ≈ 3.0
        @test proximalOracle(f, 3.0, 10.0) ≈ 3.0
    end

    @testset "Proximal Oracle - Vector" begin
        f = IndicatorNonnegativeOrthant()
        
        # All positive - should remain unchanged
        x1 = [1.0, 2.0, 3.0]
        prox1 = proximalOracle(f, x1)
        @test prox1 ≈ x1
        @test size(prox1) == size(x1)
        
        # Test in-place version
        prox1_inplace = similar(x1)
        proximalOracle!(prox1_inplace, f, x1)
        @test prox1_inplace ≈ x1
        
        # Mixed positive and negative - negative components projected to zero
        x2 = [2.0, -1.0, 3.0, -5.0]
        expected2 = [2.0, 0.0, 3.0, 0.0]
        prox2 = proximalOracle(f, x2)
        @test prox2 ≈ expected2
        
        # Test in-place version
        prox2_inplace = similar(x2)
        proximalOracle!(prox2_inplace, f, x2)
        @test prox2_inplace ≈ expected2
        
        # All negative - should become zero vector
        x3 = [-1.0, -2.0, -3.0]
        expected3 = [0.0, 0.0, 0.0]
        prox3 = proximalOracle(f, x3)
        @test prox3 ≈ expected3
        
        # Zero vector - should remain zero
        x_zero = [0.0, 0.0, 0.0]
        prox_zero = proximalOracle(f, x_zero)
        @test prox_zero ≈ x_zero
        
        # Test error for scalar in-place
        @test_throws ErrorException proximalOracle!(0.0, f, -5.0)
    end

    @testset "Proximal Oracle - Matrix" begin
        f = IndicatorNonnegativeOrthant()
        
        # All positive elements - should remain unchanged
        X1 = [1.0 2.0; 3.0 4.0]
        prox_X1 = proximalOracle(f, X1)
        @test prox_X1 ≈ X1
        
        # Mixed positive and negative elements
        X2 = [1.0 -2.0; -3.0 4.0]
        expected_X2 = [1.0 0.0; 0.0 4.0]
        prox_X2 = proximalOracle(f, X2)
        @test prox_X2 ≈ expected_X2
        
        # Test in-place version
        prox_X2_inplace = similar(X2)
        proximalOracle!(prox_X2_inplace, f, X2)
        @test prox_X2_inplace ≈ expected_X2
        
        # All negative elements - should become zero matrix
        X3 = [-1.0 -2.0; -3.0 -4.0]
        expected_X3 = [0.0 0.0; 0.0 0.0]
        prox_X3 = proximalOracle(f, X3)
        @test prox_X3 ≈ expected_X3
    end

    @testset "Edge Cases" begin
        f = IndicatorNonnegativeOrthant()
        
        # Test with very small positive values
        x_small_pos = [1e-15, 1e-15]
        @test f(x_small_pos) ≈ 0.0
        prox_small_pos = proximalOracle(f, x_small_pos)
        @test prox_small_pos ≈ x_small_pos
        
        # Test with very small negative values
        x_small_neg = [-1e-5, -1e-5]  # Clearly outside FeasTolerance (1e-6)
        @test f(x_small_neg) == Inf
        prox_small_neg = proximalOracle(f, x_small_neg)
        @test prox_small_neg ≈ [0.0, 0.0]
        
        # Test with very large values
        x_large = [1e10, -1e10, 1e-10]
        expected_large = [1e10, 0.0, 1e-10]
        prox_large = proximalOracle(f, x_large)
        @test prox_large ≈ expected_large
        
        # Test with empty arrays
        x_empty = Float64[]
        @test f(x_empty) ≈ 0.0
        prox_empty = proximalOracle(f, x_empty)
        @test prox_empty ≈ Float64[]
        
        # Test with single element
        @test f([5.0]) ≈ 0.0
        @test f([-5.0]) == Inf
        @test proximalOracle(f, [5.0]) ≈ [5.0]
        @test proximalOracle(f, [-5.0]) ≈ [0.0]
        
        # Test with sparse arrays
        x_sparse = sparse([1, 3], [1, 1], [-2.0, 3.0], 3, 1)
        prox_sparse = proximalOracle(f, x_sparse)
        expected_sparse = sparse([3], [1], [3.0], 3, 1)
        @test prox_sparse ≈ expected_sparse
        
        # Test numerical stability
        x_mixed = [1e-16, -1e-16, 1e16, -1e16]
        prox_mixed = proximalOracle(f, x_mixed)
        expected_mixed = [1e-16, 0.0, 1e16, 0.0]
        @test prox_mixed ≈ expected_mixed
    end

    @testset "Mathematical Properties" begin
        f = IndicatorNonnegativeOrthant()
        
        # Test that projection is idempotent: proj(proj(x)) = proj(x)
        x_test = [-2.0, 3.0, -1.0, 5.0]
        proj1 = proximalOracle(f, x_test)
        proj2 = proximalOracle(f, proj1)
        @test proj1 ≈ proj2
        
        # Test that projection is non-expansive: ||proj(x) - proj(y)|| ≤ ||x - y||
        x1 = [1.0, -2.0, 3.0]
        x2 = [-1.0, 4.0, -5.0]
        
        proj_x1 = proximalOracle(f, x1)
        proj_x2 = proximalOracle(f, x2)
        
        @test norm(proj_x1 - proj_x2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that projection minimizes distance to the nonnegative orthant
        x_outside = [-2.0, -3.0, 1.0]
        proj_x = proximalOracle(f, x_outside)
        
        # Check that proj_x is in the nonnegative orthant
        @test all(proj_x .≥ -1e-10)
        @test f(proj_x) ≈ 0.0
        
        # For any point in the nonnegative orthant, distance should be ≥ distance to projection
        x_in_orthant = [1.0, 2.0, 3.0]
        @test f(x_in_orthant) ≈ 0.0  # Verify it's in the orthant
        
        @test norm(x_outside - proj_x) ≤ norm(x_outside - x_in_orthant) + 1e-10
        
        # Test component-wise projection property
        x_test_comp = [2.0, -3.0, 0.0, -1.0, 5.0]
        proj_comp = proximalOracle(f, x_test_comp)
        
        for i in 1:length(x_test_comp)
            @test proj_comp[i] ≈ max(x_test_comp[i], 0.0)
        end
        
        # Test that projection preserves nonnegative components
        x_mixed = [3.0, -2.0, 7.0, -1.0]
        proj_mixed = proximalOracle(f, x_mixed)
        
        # Positive components should be unchanged
        @test proj_mixed[1] ≈ x_mixed[1]  # 3.0
        @test proj_mixed[3] ≈ x_mixed[3]  # 7.0
        
        # Negative components should become zero
        @test proj_mixed[2] ≈ 0.0
        @test proj_mixed[4] ≈ 0.0
        
        # Test scaling property: proj(αx) = α * proj(x) for α ≥ 0
        α = 2.0
        x_scale = [1.0, -2.0, 3.0]
        proj_scale = proximalOracle(f, x_scale)
        proj_scaled = proximalOracle(f, α * x_scale)
        
        @test proj_scaled ≈ α * proj_scale
    end
end 