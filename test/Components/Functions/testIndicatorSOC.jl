using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "IndicatorSOC Tests" begin
    @testset "Constructor" begin
        # Test valid construction with radius at end
        f1 = IndicatorSOC(3, 3)
        @test f1 isa IndicatorSOC
        @test f1 isa AbstractFunction
        @test f1.dim == 3
        @test f1.radiusIndex == 3
        
        # Test valid construction with radius at beginning
        f2 = IndicatorSOC(4, 1)
        @test f2.dim == 4
        @test f2.radiusIndex == 1
        
        # Test error cases
        @test_throws AssertionError IndicatorSOC(1, 1)  # Dimension too small
        @test_throws AssertionError IndicatorSOC(3, 2)  # Invalid radius index
        @test_throws AssertionError IndicatorSOC(3, 0)  # Invalid radius index
        @test_throws AssertionError IndicatorSOC(3, 4)  # Invalid radius index
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorSOC) == true
        @test isConvex(IndicatorSOC) == true
        @test isSet(IndicatorSOC) == true
        @test isSmooth(IndicatorSOC) == false
    end

    @testset "Function Evaluation - Radius at End" begin
        # Test SOC: ||x[1:n-1]|| ≤ x[n]
        f = IndicatorSOC(3, 3)
        
        # Inside the cone
        x1 = [1.0, 1.0, 2.0]  # ||[1,1]|| = √2 ≈ 1.414 ≤ 2
        @test f(x1) ≈ 0.0
        
        x2 = [0.0, 0.0, 0.0]  # ||[0,0]|| = 0 ≤ 0
        @test f(x2) ≈ 0.0
        
        x3 = [3.0, 4.0, 5.0]  # ||[3,4]|| = 5 ≤ 5
        @test f(x3) ≈ 0.0
        
        # On the boundary (within tolerance)
        x4 = [1.0, 0.0, 1.0]  # ||[1,0]|| = 1 ≤ 1
        @test f(x4) ≈ 0.0
        
        # Outside the cone
        x5 = [2.0, 2.0, 1.0]  # ||[2,2]|| = 2√2 ≈ 2.828 > 1
        @test f(x5) == Inf
        
        x6 = [1.0, 1.0, -1.0]  # ||[1,1]|| = √2 > -1
        @test f(x6) == Inf
        
        # Test with FeasTolerance
        ε = FeasTolerance / 2
        x7 = [1.0, 0.0, 1.0 - ε]  # ||[1,0]|| = 1 ≤ 1 - ε + FeasTolerance
        @test f(x7) ≈ 0.0
    end

    @testset "Function Evaluation - Radius at Beginning" begin
        # Test SOC: ||x[2:n]|| ≤ x[1]
        f = IndicatorSOC(3, 1)
        
        # Inside the cone
        x1 = [2.0, 1.0, 1.0]  # ||[1,1]|| = √2 ≈ 1.414 ≤ 2
        @test f(x1) ≈ 0.0
        
        x2 = [0.0, 0.0, 0.0]  # ||[0,0]|| = 0 ≤ 0
        @test f(x2) ≈ 0.0
        
        x3 = [5.0, 3.0, 4.0]  # ||[3,4]|| = 5 ≤ 5
        @test f(x3) ≈ 0.0
        
        # Outside the cone
        x4 = [1.0, 2.0, 2.0]  # ||[2,2]|| = 2√2 ≈ 2.828 > 1
        @test f(x4) == Inf
        
        x5 = [-1.0, 1.0, 1.0]  # ||[1,1]|| = √2 > -1
        @test f(x5) == Inf
    end

    @testset "Proximal Oracle - Radius at End" begin
        f = IndicatorSOC(3, 3)
        
        # Point inside the cone - should remain unchanged
        x1 = [1.0, 1.0, 2.0]  # ||[1,1]|| = √2 ≤ 2
        prox1 = proximalOracle(f, x1)
        @test prox1 ≈ x1
        @test size(prox1) == size(x1)
        
        # Test in-place version
        prox1_inplace = similar(x1)
        proximalOracle!(prox1_inplace, f, x1)
        @test prox1_inplace ≈ x1
        
        # Point on the boundary - should remain unchanged
        x2 = [3.0, 4.0, 5.0]  # ||[3,4]|| = 5 = 5
        prox2 = proximalOracle(f, x2)
        @test prox2 ≈ x2
        
        # Point outside the cone - should be projected
        x3 = [2.0, 2.0, 1.0]  # ||[2,2]|| = 2√2 ≈ 2.828 > 1
        prox3 = proximalOracle(f, x3)
        
        # Check that projection is in the cone
        vec_norm = norm(prox3[1:end-1])
        @test vec_norm ≤ prox3[end] + 1e-10
        
        # Check that projection is on the boundary (for points outside)
        @test abs(vec_norm - prox3[end]) < 1e-10
        
        # Point with negative radius component
        x4 = [1.0, 1.0, -2.0]
        prox4 = proximalOracle(f, x4)
        
        # Should project to origin when ||x[1:n-1]|| ≤ -x[n]
        vec_norm4 = norm(x4[1:end-1])  # √2 ≈ 1.414
        radius4 = x4[end]  # -2
        if vec_norm4 <= -radius4  # 1.414 ≤ 2, so should project to zero
            @test prox4 ≈ zeros(3)
        end
        
        # Test the projection formula for general case
        x5 = [6.0, 8.0, 2.0]  # ||[6,8]|| = 10 > 2
        prox5 = proximalOracle(f, x5)
        
        vec_norm5 = norm(x5[1:end-1])  # 10
        radius5 = x5[end]  # 2
        scaler = (vec_norm5 + radius5) / (2 * vec_norm5)  # (10 + 2) / 20 = 0.6
        
        expected_vec = scaler * x5[1:end-1]  # 0.6 * [6,8] = [3.6, 4.8]
        expected_radius = scaler * vec_norm5  # 0.6 * 10 = 6
        
        @test prox5[1:end-1] ≈ expected_vec atol=1e-6
        @test prox5[end] ≈ expected_radius atol=1e-6
    end

    @testset "Proximal Oracle - Radius at Beginning" begin
        f = IndicatorSOC(3, 1)
        
        # Point inside the cone - should remain unchanged
        x1 = [2.0, 1.0, 1.0]  # ||[1,1]|| = √2 ≤ 2
        prox1 = proximalOracle(f, x1)
        @test prox1 ≈ x1
        
        # Point outside the cone - should be projected
        x2 = [1.0, 6.0, 8.0]  # ||[6,8]|| = 10 > 1
        prox2 = proximalOracle(f, x2)
        
        # Check that projection is in the cone
        vec_norm = norm(prox2[2:end])
        @test vec_norm ≤ prox2[1] + 1e-10
        
        # Check projection formula
        vec_norm2 = norm(x2[2:end])  # 10
        radius2 = x2[1]  # 1
        scaler = (vec_norm2 + radius2) / (2 * vec_norm2)  # (10 + 1) / 20 = 0.55
        
        expected_radius = scaler * vec_norm2  # 0.55 * 10 = 5.5
        expected_vec = scaler * x2[2:end]  # 0.55 * [6,8] = [3.3, 4.4]
        
        @test prox2[1] ≈ expected_radius atol=1e-6
        @test prox2[2:end] ≈ expected_vec atol=1e-6
        
        # Test in-place version
        prox2_inplace = similar(x2)
        proximalOracle!(prox2_inplace, f, x2)
        @test prox2_inplace ≈ prox2
    end

    @testset "Edge Cases" begin
        # Test with minimum dimension
        f_min = IndicatorSOC(2, 2)
        
        x_min = [1.0, 2.0]  # ||[1]|| = 1 ≤ 2
        @test f_min(x_min) ≈ 0.0
        
        prox_min = proximalOracle(f_min, x_min)
        @test prox_min ≈ x_min
        
        # Test with zero vector
        x_zero = [0.0, 0.0, 0.0]
        f = IndicatorSOC(3, 3)
        @test f(x_zero) ≈ 0.0
        
        prox_zero = proximalOracle(f, x_zero)
        @test prox_zero ≈ x_zero
        
        # Test with very large values
        x_large = [1e6, 1e6, 2e6]
        @test f(x_large) ≈ 0.0  # ||[1e6, 1e6]|| = √2 * 1e6 ≈ 1.414e6 < 2e6
        
        prox_large = proximalOracle(f, x_large)
        @test prox_large ≈ x_large
        
        # Test with very small values
        x_small = [1e-10, 1e-10, 2e-10]
        @test f(x_small) ≈ 0.0
        
        prox_small = proximalOracle(f, x_small)
        @test prox_small ≈ x_small
        
        # Test numerical stability near boundary
        x_boundary = [1.0, 0.0, 1.0 + 1e-12]  # Very close to boundary
        @test f(x_boundary) ≈ 0.0  # Should be feasible due to tolerance
        
        # Test with negative radius and large vector norm
        x_neg = [1e6, 1e6, -1.0]
        prox_neg = proximalOracle(f, x_neg)
        # The actual projection formula doesn't necessarily give zero
        # Just check that the result is finite and satisfies cone constraints
        @test all(isfinite.(prox_neg))
        vec_norm_result = norm(prox_neg[1:end-1])
        @test vec_norm_result ≤ prox_neg[end] + 1e-6  # Should be in the cone
    end

    @testset "Mathematical Properties" begin
        f = IndicatorSOC(4, 4)
        
        # Test that projection is idempotent: proj(proj(x)) = proj(x)
        x_test = [3.0, 4.0, 5.0, 2.0]  # Outside cone
        proj1 = proximalOracle(f, x_test)
        proj2 = proximalOracle(f, proj1)
        @test proj1 ≈ proj2
        
        # Test that projection is non-expansive: ||proj(x) - proj(y)|| ≤ ||x - y||
        x1 = [1.0, 2.0, 3.0, 1.0]
        x2 = [2.0, 1.0, 1.0, 3.0]
        
        proj_x1 = proximalOracle(f, x1)
        proj_x2 = proximalOracle(f, x2)
        
        @test norm(proj_x1 - proj_x2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that projection minimizes distance to the cone
        x_outside = [6.0, 8.0, 0.0, 1.0]  # ||[6,8,0]|| = 10 > 1
        proj_x = proximalOracle(f, x_outside)
        
        # Check that proj_x is in the cone
        vec_norm = norm(proj_x[1:end-1])
        @test vec_norm ≤ proj_x[end] + 1e-10
        
        # For any point in the cone, distance should be ≥ distance to projection
        x_in_cone = [0.5, 0.5, 0.0, 1.0]  # ||[0.5,0.5,0]|| = √0.5 < 1
        @test f(x_in_cone) ≈ 0.0  # Verify it's in the cone
        
        @test norm(x_outside - proj_x) ≤ norm(x_outside - x_in_cone) + 1e-10
        
        # Test symmetry properties for radius at beginning vs end
        f_begin = IndicatorSOC(3, 1)
        f_end = IndicatorSOC(3, 3)
        
        # Transform coordinates: [r, v] ↔ [v, r]
        x_begin_format = [2.0, 1.0, 1.0]  # radius first
        x_end_format = [1.0, 1.0, 2.0]    # radius last
        
        @test f_begin(x_begin_format) == f_end(x_end_format)
        
        # Test gamma parameter doesn't affect projection (indicator functions)
        x_test_gamma = [2.0, 2.0, 1.0]  # Make sure this matches f dimension (4)
        f_gamma = IndicatorSOC(3, 3)  # Use dimension 3 to match x_test_gamma
        @test proximalOracle(f_gamma, x_test_gamma, 0.1) ≈ proximalOracle(f_gamma, x_test_gamma, 10.0)
    end
end 