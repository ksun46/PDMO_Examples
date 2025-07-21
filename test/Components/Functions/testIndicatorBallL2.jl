using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "IndicatorBallL2 Tests" begin
    @testset "Constructor" begin
        # Test valid construction
        r = 2.5
        f = IndicatorBallL2(r)
        @test f isa IndicatorBallL2
        @test f isa AbstractFunction
        @test f.r == r
        
        # Test with different radii
        f1 = IndicatorBallL2(1.0)
        @test f1.r == 1.0
        
        f2 = IndicatorBallL2(0.1)
        @test f2.r == 0.1
        
        f3 = IndicatorBallL2(100.0)
        @test f3.r == 100.0
        
        # Test error cases
        @test_throws ErrorException IndicatorBallL2(0.0)   # Zero radius
        @test_throws ErrorException IndicatorBallL2(-1.0)  # Negative radius
        @test_throws ErrorException IndicatorBallL2(-0.1)  # Small negative radius
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorBallL2) == true
        @test isConvex(IndicatorBallL2) == true
        @test isSet(IndicatorBallL2) == true
        @test isSmooth(IndicatorBallL2) == false
    end

    @testset "Function Evaluation - Scalar" begin
        f = IndicatorBallL2(2.0)
        
        # Inside the ball
        @test f(1.0) ≈ 0.0
        @test f(-1.5) ≈ 0.0
        @test f(0.0) ≈ 0.0
        
        # On the boundary (within tolerance)
        @test f(2.0) ≈ 0.0
        @test f(-2.0) ≈ 0.0
        
        # Test with FeasTolerance
        ε = FeasTolerance / 2
        @test f(2.0 + ε) ≈ 0.0  # Should be feasible due to tolerance
        @test f(-2.0 - ε) ≈ 0.0
        
        # Outside the ball
        @test f(3.0) == Inf
        @test f(-3.0) == Inf
        @test f(2.1) == Inf
    end

    @testset "Function Evaluation - Vector" begin
        f = IndicatorBallL2(3.0)
        
        # Inside the ball
        x1 = [1.0, 2.0]  # ||x|| = √5 ≈ 2.236 < 3
        @test f(x1) ≈ 0.0
        
        x2 = [0.0, 0.0, 0.0]  # ||x|| = 0 < 3
        @test f(x2) ≈ 0.0
        
        x3 = [1.0, 1.0, 1.0]  # ||x|| = √3 ≈ 1.732 < 3
        @test f(x3) ≈ 0.0
        
        # On the boundary
        x4 = [3.0, 0.0]  # ||x|| = 3
        @test f(x4) ≈ 0.0
        
        x5 = [0.0, 0.0, 3.0]  # ||x|| = 3
        @test f(x5) ≈ 0.0
        
        # Outside the ball
        x6 = [2.0, 2.0, 2.0]  # ||x|| = 2√3 ≈ 3.464 > 3
        @test f(x6) == Inf
        
        x7 = [4.0, 0.0]  # ||x|| = 4 > 3
        @test f(x7) == Inf
    end

    @testset "Function Evaluation - Matrix" begin
        f = IndicatorBallL2(5.0)
        
        # Inside the ball
        X1 = [1.0 2.0; 3.0 1.0]  # ||X||_F = √(1+4+9+1) = √15 ≈ 3.873 < 5
        @test f(X1) ≈ 0.0
        
        # On the boundary
        X2 = [3.0 4.0; 0.0 0.0]  # ||X||_F = √(9+16) = 5
        @test f(X2) ≈ 0.0
        
        # Outside the ball
        X3 = [3.0 4.0; 3.0 4.0]  # ||X||_F = √(9+16+9+16) = √50 ≈ 7.071 > 5
        @test f(X3) == Inf
    end

    @testset "Proximal Oracle - Scalar" begin
        f = IndicatorBallL2(2.0)
        
        # Inside the ball - should remain unchanged
        @test proximalOracle(f, 1.0) ≈ 1.0
        @test proximalOracle(f, -1.5) ≈ -1.5
        @test proximalOracle(f, 0.0) ≈ 0.0
        
        # On the boundary - should remain unchanged
        @test proximalOracle(f, 2.0) ≈ 2.0
        @test proximalOracle(f, -2.0) ≈ -2.0
        
        # Outside the ball - should be projected to boundary
        @test proximalOracle(f, 4.0) ≈ 2.0   # Project to r * sign(x)
        @test proximalOracle(f, -6.0) ≈ -2.0
        @test proximalOracle(f, 10.0) ≈ 2.0
        
        # Test gamma parameter doesn't affect projection (indicator functions)
        @test proximalOracle(f, 4.0, 0.1) ≈ 2.0
        @test proximalOracle(f, 4.0, 10.0) ≈ 2.0
    end

    @testset "Proximal Oracle - Vector" begin
        f = IndicatorBallL2(3.0)
        
        # Inside the ball - should remain unchanged
        x1 = [1.0, 2.0]  # ||x|| = √5 ≈ 2.236 < 3
        prox1 = proximalOracle(f, x1)
        @test prox1 ≈ x1
        @test size(prox1) == size(x1)
        
        # Test in-place version
        prox1_inplace = similar(x1)
        proximalOracle!(prox1_inplace, f, x1)
        @test prox1_inplace ≈ x1
        
        # On the boundary - should remain unchanged
        x2 = [3.0, 0.0]  # ||x|| = 3
        prox2 = proximalOracle(f, x2)
        @test prox2 ≈ x2
        
        # Outside the ball - should be projected
        x3 = [6.0, 8.0]  # ||x|| = 10 > 3
        prox3 = proximalOracle(f, x3)
        
        # Should be projected to (r/||x||) * x = (3/10) * [6,8] = [1.8, 2.4]
        expected3 = (3.0 / 10.0) * x3
        @test prox3 ≈ expected3
        @test norm(prox3) ≈ 3.0  # Should be on the boundary
        
        # Test in-place version
        prox3_inplace = similar(x3)
        proximalOracle!(prox3_inplace, f, x3)
        @test prox3_inplace ≈ expected3
        
        # Test with zero vector
        x_zero = [0.0, 0.0, 0.0]
        prox_zero = proximalOracle(f, x_zero)
        @test prox_zero ≈ x_zero
        
        # Test error for scalar in-place (should throw error)
        @test_throws ErrorException proximalOracle!(0.0, f, 5.0)
    end

    @testset "Proximal Oracle - Matrix" begin
        f = IndicatorBallL2(4.0)
        
        # Inside the ball - should remain unchanged
        X1 = [1.0 1.0; 1.0 1.0]  # ||X||_F = 2 < 4
        prox_X1 = proximalOracle(f, X1)
        @test prox_X1 ≈ X1
        
        # Outside the ball - should be projected
        X2 = [3.0 4.0; 0.0 0.0]  # ||X||_F = 5 > 4
        prox_X2 = proximalOracle(f, X2)
        
        # Should be projected to (r/||X||_F) * X = (4/5) * X
        expected_X2 = (4.0 / 5.0) * X2
        @test prox_X2 ≈ expected_X2
        @test norm(prox_X2) ≈ 4.0
        
        # Test in-place version
        prox_X2_inplace = similar(X2)
        proximalOracle!(prox_X2_inplace, f, X2)
        @test prox_X2_inplace ≈ expected_X2
    end

    @testset "Edge Cases" begin
        # Test with very small radius
        f_small = IndicatorBallL2(1e-6)
        
        x_small = [1e-5, 1e-5]  # ||x|| = √2 * 1e-5 ≈ 1.414e-5 >> 1e-6
        @test f_small(x_small) == Inf
        
        prox_small = proximalOracle(f_small, x_small)
        @test norm(prox_small) ≈ 1e-6 atol=1e-8
        
        # Test with very large radius
        f_large = IndicatorBallL2(1e6)
        
        x_large = [1e5, 1e5]  # ||x|| = √2 * 1e5 ≈ 1.414e5 < 1e6
        @test f_large(x_large) ≈ 0.0
        
        prox_large = proximalOracle(f_large, x_large)
        @test prox_large ≈ x_large
        
        # Test with unit radius and unit vector
        f_unit = IndicatorBallL2(1.0)
        x_unit = [1.0, 0.0]  # ||x|| = 1
        @test f_unit(x_unit) ≈ 0.0
        
        prox_unit = proximalOracle(f_unit, x_unit)
        @test prox_unit ≈ x_unit
        
        # Test with very large input
        x_huge = [1e10, 1e10]
        prox_huge = proximalOracle(f_unit, x_huge)
        expected_huge = x_huge / norm(x_huge)  # Should be unit vector
        @test prox_huge ≈ expected_huge
        @test norm(prox_huge) ≈ 1.0
        
        # Test numerical stability
        x_tiny = [1e-15, 1e-15]
        prox_tiny = proximalOracle(f_unit, x_tiny)
        @test all(isfinite.(prox_tiny))
        
        # Test with sparse vectors
        x_sparse = sparsevec([1, 3], [4.0, 3.0], 5)  # ||x|| = 5
        f_sparse = IndicatorBallL2(2.0)
        prox_sparse = proximalOracle(f_sparse, x_sparse)
        @test norm(prox_sparse) ≈ 2.0
    end

    @testset "Mathematical Properties" begin
        f = IndicatorBallL2(2.5)
        
        # Test that projection is idempotent: proj(proj(x)) = proj(x)
        x_test = [4.0, 3.0]  # Outside ball
        proj1 = proximalOracle(f, x_test)
        proj2 = proximalOracle(f, proj1)
        @test proj1 ≈ proj2
        
        # Test that projection is non-expansive: ||proj(x) - proj(y)|| ≤ ||x - y||
        x1 = [1.0, 2.0]
        x2 = [3.0, 1.0]
        
        proj_x1 = proximalOracle(f, x1)
        proj_x2 = proximalOracle(f, x2)
        
        @test norm(proj_x1 - proj_x2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that projection minimizes distance to the ball
        x_outside = [5.0, 0.0]  # Outside ball
        proj_x = proximalOracle(f, x_outside)
        
        # Check that proj_x is in the ball
        @test norm(proj_x) ≤ f.r + 1e-10
        
        # For any point in the ball, distance should be ≥ distance to projection
        x_in_ball = [1.0, 1.0]  # ||x|| = √2 < 2.5
        @test f(x_in_ball) ≈ 0.0  # Verify it's in the ball
        
        @test norm(x_outside - proj_x) ≤ norm(x_outside - x_in_ball) + 1e-10
        
        # Test that projection preserves direction for points outside the ball
        x_dir_test = [6.0, 8.0]  # ||x|| = 10
        proj_dir = proximalOracle(f, x_dir_test)
        
        # Direction should be preserved: proj_x = (r/||x||) * x
        expected_dir = (f.r / norm(x_dir_test)) * x_dir_test
        @test proj_dir ≈ expected_dir
        
        # Test scaling property
        scale = 2.0
        x_scale = [3.0, 4.0]  # ||x|| = 5
        proj_scale = proximalOracle(f, x_scale)
        
        # If we scale the input, the projection should scale proportionally (when outside)
        x_scaled = scale * x_scale
        proj_scaled = proximalOracle(f, x_scaled)
        
        # Both should have the same direction but norm = r
        @test norm(proj_scale) ≈ f.r
        @test norm(proj_scaled) ≈ f.r
        
        # Directions should be the same
        dir1 = proj_scale / norm(proj_scale)
        dir2 = proj_scaled / norm(proj_scaled)
        @test dir1 ≈ dir2
        
        # Test that projection onto ball with different radii scales correctly
        f_double = IndicatorBallL2(2 * f.r)
        proj_double = proximalOracle(f_double, x_scale)
        
        # If radius doubles and point is outside both balls, projection should scale
        if norm(x_scale) > 2 * f.r
            @test norm(proj_double) ≈ 2 * f.r
        end
    end
end 