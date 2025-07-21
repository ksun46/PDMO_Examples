using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "IndicatorHyperplane Tests" begin
    @testset "Constructor" begin
        # Test valid construction
        slope = [1.0, 2.0]
        intercept = 3.0
        f = IndicatorHyperplane(slope, intercept)
        @test f isa IndicatorHyperplane
        @test f isa AbstractFunction
        @test f.slope == slope
        @test f.intercept == intercept
        
        # Check that scaledSlope is computed correctly
        norm_sq = dot(slope, slope)  # 1 + 4 = 5
        expected_scaled = slope ./ norm_sq  # [1/5, 2/5]
        @test f.scaledSlope ≈ expected_scaled
        
        # Test with different slopes and intercepts
        slope2 = [3.0, 4.0, 0.0]
        intercept2 = -2.0
        f2 = IndicatorHyperplane(slope2, intercept2)
        @test f2.slope == slope2
        @test f2.intercept == intercept2
        
        norm_sq2 = dot(slope2, slope2)  # 9 + 16 + 0 = 25
        expected_scaled2 = slope2 ./ norm_sq2  # [3/25, 4/25, 0]
        @test f2.scaledSlope ≈ expected_scaled2
        
        # Test with unit vector
        slope_unit = [1.0, 0.0]
        f_unit = IndicatorHyperplane(slope_unit, 1.0)
        @test f_unit.scaledSlope ≈ [1.0, 0.0]  # Should be unchanged
        
        # Test error cases
        @test_throws AssertionError IndicatorHyperplane(Float64[], 1.0)  # Empty slope
        @test_throws AssertionError IndicatorHyperplane([0.0, 0.0], 1.0)  # Zero slope
        @test_throws AssertionError IndicatorHyperplane([1e-15, 0.0], 1.0)  # Very small slope (below ZeroTolerance)
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorHyperplane) == true
        @test isConvex(IndicatorHyperplane) == true
        @test isSet(IndicatorHyperplane) == true
        @test isSmooth(IndicatorHyperplane) == false
    end

    @testset "Function Evaluation" begin
        # Test hyperplane: x + 2y = 5
        slope = [1.0, 2.0]
        intercept = 5.0
        f = IndicatorHyperplane(slope, intercept)
        
        # Points on the hyperplane
        x_on1 = [1.0, 2.0]  # 1 + 2*2 = 5 ✓
        @test f(x_on1) ≈ 0.0
        
        x_on2 = [5.0, 0.0]  # 5 + 2*0 = 5 ✓
        @test f(x_on2) ≈ 0.0
        
        x_on3 = [3.0, 1.0]  # 3 + 2*1 = 5 ✓
        @test f(x_on3) ≈ 0.0
        
        x_on4 = [-1.0, 3.0]  # -1 + 2*3 = 5 ✓
        @test f(x_on4) ≈ 0.0
        
        # Points not on the hyperplane
        x_off1 = [0.0, 0.0]  # 0 + 2*0 = 0 ≠ 5
        @test f(x_off1) == Inf
        
        x_off2 = [1.0, 1.0]  # 1 + 2*1 = 3 ≠ 5
        @test f(x_off2) == Inf
        
        x_off3 = [2.0, 3.0]  # 2 + 2*3 = 8 ≠ 5
        @test f(x_off3) == Inf
        
        # Test with FeasTolerance
        ε = FeasTolerance / 2
        x_near = [1.0, 2.0 + ε/2.0]  # Close to hyperplane
        residual = dot(slope, x_near) - intercept
        if abs(residual) <= FeasTolerance
            @test f(x_near) ≈ 0.0
        else
            @test f(x_near) == Inf
        end
        
        # Test with different hyperplane: 3x - y + 2z = 1
        slope2 = [3.0, -1.0, 2.0]
        intercept2 = 1.0
        f2 = IndicatorHyperplane(slope2, intercept2)
        
        x_on_2d = [1.0, 2.0, 0.0]  # 3*1 - 2 + 2*0 = 1 ✓
        @test f2(x_on_2d) ≈ 0.0
        
        x_off_2d = [0.0, 0.0, 0.0]  # 3*0 - 0 + 2*0 = 0 ≠ 1
        @test f2(x_off_2d) == Inf
        
        # Test error for dimension mismatch
        @test_throws AssertionError f([1.0])  # Wrong dimension
        @test_throws AssertionError f([1.0, 2.0, 3.0])  # Wrong dimension
    end

    @testset "Proximal Oracle" begin
        # Test projection onto hyperplane: x + 2y = 5
        slope = [1.0, 2.0]
        intercept = 5.0
        f = IndicatorHyperplane(slope, intercept)
        
        # Test projection from arbitrary point
        x_start = [0.0, 0.0]  # Not on hyperplane
        prox = proximalOracle(f, x_start)
        
        # Projection formula: y = x - (<a,x> - b)a/‖a‖²
        # residual = <[1,2], [0,0]> - 5 = -5
        # y = [0,0] - (-5) * [1,2]/5 = [0,0] + [1,2] = [1,2]
        expected = [1.0, 2.0]
        @test prox ≈ expected
        @test size(prox) == size(x_start)
        
        # Verify that result is on hyperplane
        @test f(prox) ≈ 0.0
        
        # Test in-place version
        prox_inplace = similar(x_start)
        proximalOracle!(prox_inplace, f, x_start)
        @test prox_inplace ≈ expected
        
        # Test projection from another point
        x_start2 = [3.0, 4.0]  # 3 + 2*4 = 11, residual = 6
        prox2 = proximalOracle(f, x_start2)
        
        # y = [3,4] - 6 * [1,2]/5 = [3,4] - [6/5, 12/5] = [3-6/5, 4-12/5] = [9/5, 8/5]
        expected2 = [9.0/5.0, 8.0/5.0]
        @test prox2 ≈ expected2
        @test f(prox2) ≈ 0.0
        
        # Test that points already on hyperplane remain unchanged
        x_on_plane = [1.0, 2.0]  # Already on hyperplane
        prox_unchanged = proximalOracle(f, x_on_plane)
        @test prox_unchanged ≈ x_on_plane atol=1e-10
        
        # Test with different hyperplane: 2x - y = 3
        slope3 = [2.0, -1.0]
        intercept3 = 3.0
        f3 = IndicatorHyperplane(slope3, intercept3)
        
        x_test3 = [0.0, 1.0]  # 2*0 - 1 = -1, residual = -4
        prox3 = proximalOracle(f3, x_test3)
        
        # y = [0,1] - (-4) * [2,-1]/5 = [0,1] + [8/5, -4/5] = [8/5, 1/5]
        expected3 = [8.0/5.0, 1.0/5.0]
        @test prox3 ≈ expected3
        @test f3(prox3) ≈ 0.0
    end

    @testset "Proximal Oracle - Higher Dimensions" begin
        # Test 3D hyperplane: x + y + z = 6
        slope = [1.0, 1.0, 1.0]
        intercept = 6.0
        f = IndicatorHyperplane(slope, intercept)
        
        x_start = [1.0, 2.0, 4.0]  # sum = 7, residual = 1
        prox = proximalOracle(f, x_start)
        
        # y = [1,2,4] - 1 * [1,1,1]/3 = [1,2,4] - [1/3,1/3,1/3] = [2/3, 5/3, 11/3]
        expected = [2.0/3.0, 5.0/3.0, 11.0/3.0]
        @test prox ≈ expected
        @test f(prox) ≈ 0.0
        
        # Test 4D hyperplane: 2w + x - y + 3z = 10
        slope4d = [2.0, 1.0, -1.0, 3.0]
        intercept4d = 10.0
        f4d = IndicatorHyperplane(slope4d, intercept4d)
        
        x_start4d = [1.0, 2.0, 3.0, 1.0]  # 2*1 + 2 - 3 + 3*1 = 4, residual = -6
        prox4d = proximalOracle(f4d, x_start4d)
        
        # norm_sq = 4 + 1 + 1 + 9 = 15
        # y = [1,2,3,1] - (-6) * [2,1,-1,3]/15 = [1,2,3,1] + [12/15, 6/15, -6/15, 18/15]
        expected4d = [1.0 + 12.0/15.0, 2.0 + 6.0/15.0, 3.0 - 6.0/15.0, 1.0 + 18.0/15.0]
        @test prox4d ≈ expected4d
        @test f4d(prox4d) ≈ 0.0 atol=1e-10
    end

    @testset "Edge Cases" begin
        # Test with unit normal vector
        slope_unit = [1.0, 0.0]
        intercept_unit = 2.0
        f_unit = IndicatorHyperplane(slope_unit, intercept_unit)
        
        x_unit = [0.0, 5.0]  # Should project to [2.0, 5.0]
        prox_unit = proximalOracle(f_unit, x_unit)
        @test prox_unit ≈ [2.0, 5.0]
        @test f_unit(prox_unit) ≈ 0.0
        
        # Test with very small slope components
        slope_small = [1e-6, 1.0]
        intercept_small = 1.0
        f_small = IndicatorHyperplane(slope_small, intercept_small)
        
        x_small = [0.0, 0.0]
        prox_small = proximalOracle(f_small, x_small)
        @test f_small(prox_small) ≈ 0.0
        @test all(isfinite.(prox_small))
        
        # Test with large slope components
        slope_large = [1e6, 1e6]
        intercept_large = 2e6
        f_large = IndicatorHyperplane(slope_large, intercept_large)
        
        x_large = [0.0, 0.0]
        prox_large = proximalOracle(f_large, x_large)
        @test f_large(prox_large) ≈ 0.0 atol=1e-6
        @test all(isfinite.(prox_large))
        
        # Test with zero intercept
        slope_zero_int = [1.0, 1.0]
        f_zero_int = IndicatorHyperplane(slope_zero_int, 0.0)
        
        x_zero_int = [1.0, 2.0]  # sum = 3, should project to origin direction
        prox_zero_int = proximalOracle(f_zero_int, x_zero_int)
        @test f_zero_int(prox_zero_int) ≈ 0.0
        
        # Test with negative intercept
        slope_neg = [1.0, 1.0]
        intercept_neg = -5.0
        f_neg = IndicatorHyperplane(slope_neg, intercept_neg)
        
        x_neg = [0.0, 0.0]  # sum = 0, residual = 5
        prox_neg = proximalOracle(f_neg, x_neg)
        @test f_neg(prox_neg) ≈ 0.0
        
        # Test numerical stability
        slope_cond = [1.0, 1e-12]
        intercept_cond = 1.0
        f_cond = IndicatorHyperplane(slope_cond, intercept_cond)
        
        x_cond = [0.0, 1e12]
        prox_cond = proximalOracle(f_cond, x_cond)
        @test all(isfinite.(prox_cond))
        
        # Test with single dimension
        slope_1d = [3.0]
        intercept_1d = 6.0
        f_1d = IndicatorHyperplane(slope_1d, intercept_1d)
        
        x_1d = [1.0]  # 3*1 = 3 ≠ 6
        prox_1d = proximalOracle(f_1d, x_1d)
        @test prox_1d ≈ [2.0]  # Should be exactly 6/3 = 2
        @test f_1d(prox_1d) ≈ 0.0
    end

    @testset "Mathematical Properties" begin
        slope = [3.0, 4.0]
        intercept = 10.0
        f = IndicatorHyperplane(slope, intercept)
        
        # Test that projection is idempotent: proj(proj(x)) = proj(x)
        x_test = [1.0, 1.0]
        proj1 = proximalOracle(f, x_test)
        proj2 = proximalOracle(f, proj1)
        @test proj1 ≈ proj2 atol=1e-10
        
        # Test that projection is non-expansive: ||proj(x) - proj(y)|| ≤ ||x - y||
        x1 = [1.0, 2.0]
        x2 = [3.0, 1.0]
        
        proj_x1 = proximalOracle(f, x1)
        proj_x2 = proximalOracle(f, x2)
        
        @test norm(proj_x1 - proj_x2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that projection minimizes distance to the hyperplane
        x_outside = [0.0, 0.0]
        proj_x = proximalOracle(f, x_outside)
        
        # Check that proj_x is on the hyperplane
        @test f(proj_x) ≈ 0.0
        
        # For any point on the hyperplane, distance should be ≥ distance to projection
        x_on_plane = [2.0, 1.0]  # 3*2 + 4*1 = 10 ✓
        @test f(x_on_plane) ≈ 0.0  # Verify it's on the hyperplane
        
        @test norm(x_outside - proj_x) ≤ norm(x_outside - x_on_plane) + 1e-10
        
        # Test that gamma parameter doesn't affect projection (indicator functions)
        @test proximalOracle(f, x_test, 0.1) ≈ proximalOracle(f, x_test, 10.0) atol=1e-10
        
        # Test orthogonality property: (x - proj(x)) is parallel to normal vector
        x_orth_test = [5.0, 2.0]
        proj_orth = proximalOracle(f, x_orth_test)
        difference = x_orth_test - proj_orth
        
        # The difference should be parallel to the slope vector
        # This means difference = α * slope for some scalar α
        if norm(difference) > 1e-10  # Avoid division by zero
            normalized_diff = difference / norm(difference)
            normalized_slope = slope / norm(slope)
            # Check if they are parallel (dot product should be ±1)
            @test abs(abs(dot(normalized_diff, normalized_slope)) - 1.0) < 1e-10
        end
        
        # Test projection formula directly
        x_formula_test = [1.0, 3.0]
        residual = dot(slope, x_formula_test) - intercept  # 3*1 + 4*3 - 10 = 5
        norm_sq = dot(slope, slope)  # 9 + 16 = 25
        expected_proj = x_formula_test - (residual / norm_sq) * slope
        # expected = [1,3] - (5/25) * [3,4] = [1,3] - [3/5, 4/5] = [2/5, 11/5]
        
        actual_proj = proximalOracle(f, x_formula_test)
        @test actual_proj ≈ expected_proj
        
        # Test that projection preserves distances along the hyperplane
        # Points that differ only in the direction parallel to the hyperplane
        # should have the same distance after projection
        
        # Find two points on the hyperplane
        x_on1 = [2.0, 1.0]  # 3*2 + 4*1 = 10 ✓
        x_on2 = [6.0, -2.0]  # 3*6 + 4*(-2) = 10 ✓
        
        # Project them (should remain unchanged)
        proj_on1 = proximalOracle(f, x_on1)
        proj_on2 = proximalOracle(f, x_on2)
        
        @test proj_on1 ≈ x_on1 atol=1e-10
        @test proj_on2 ≈ x_on2 atol=1e-10
        
        # Test scaling property of the hyperplane
        # If we scale the slope and intercept by the same factor, 
        # the hyperplane should remain the same
        scale_factor = 2.0
        slope_scaled = scale_factor * slope
        intercept_scaled = scale_factor * intercept
        f_scaled = IndicatorHyperplane(slope_scaled, intercept_scaled)
        
        x_scale_test = [1.0, 1.0]
        proj_original = proximalOracle(f, x_scale_test)
        proj_scaled = proximalOracle(f_scaled, x_scale_test)
        
        @test proj_original ≈ proj_scaled atol=1e-10
        
        # Test that the projection lies on the line connecting x and the closest point
        x_line_test = [0.0, 5.0]
        proj_line = proximalOracle(f, x_line_test)
        
        # The projection should minimize ||x - y||² subject to <slope, y> = intercept
        # This means proj_line should be the closest point on the hyperplane to x_line_test
        
        # Verify by checking that (x - proj) is orthogonal to the hyperplane
        diff_line = x_line_test - proj_line
        # For any vector v on the hyperplane (i.e., <slope, v> = 0), we should have <diff, v> = 0
        # Since slope is normal to the hyperplane, diff should be parallel to slope
        
        if norm(diff_line) > 1e-10
            # Check that diff_line is parallel to slope
            cross_product_norm = norm(diff_line[1] * slope[2] - diff_line[2] * slope[1])
            @test cross_product_norm < 1e-10
        end
    end
end 