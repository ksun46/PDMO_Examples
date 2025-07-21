using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "IndicatorSumOfNVariables Tests" begin
    @testset "Constructor" begin
        # Test scalar case
        f_scalar = IndicatorSumOfNVariables(3, 5.0)
        @test f_scalar isa IndicatorSumOfNVariables
        @test f_scalar isa AbstractFunction
        @test f_scalar.numberVariables == 3
        @test f_scalar.rhs == 5.0
        
        # Test vector case
        rhs_vec = [1.0, 2.0, 3.0]
        f_vector = IndicatorSumOfNVariables(2, rhs_vec)
        @test f_vector.numberVariables == 2
        @test f_vector.rhs == rhs_vec
        
        # Test matrix case
        rhs_mat = [1.0 2.0; 3.0 4.0]
        f_matrix = IndicatorSumOfNVariables(4, rhs_mat)
        @test f_matrix.numberVariables == 4
        @test f_matrix.rhs == rhs_mat
        
        # Test with different numbers of variables
        f_many = IndicatorSumOfNVariables(10, 0.0)
        @test f_many.numberVariables == 10
        
        f_single = IndicatorSumOfNVariables(1, 1.0)
        @test f_single.numberVariables == 1
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorSumOfNVariables) == true
        @test isConvex(IndicatorSumOfNVariables) == true
        @test isSet(IndicatorSumOfNVariables) == true
        @test isSmooth(IndicatorSumOfNVariables) == false
    end

    @testset "Function Evaluation - Scalar Case" begin
        # Test constraint: x₁ + x₂ + x₃ = 5
        f = IndicatorSumOfNVariables(3, 5.0)
        
        # Satisfies constraint
        x_valid = [1.0, 2.0, 2.0]  # sum = 5
        @test f(x_valid) ≈ 0.0
        
        x_valid2 = [0.0, 0.0, 5.0]  # sum = 5
        @test f(x_valid2) ≈ 0.0
        
        x_valid3 = [-1.0, 3.0, 3.0]  # sum = 5
        @test f(x_valid3) ≈ 0.0
        
        # Does not satisfy constraint
        x_invalid = [1.0, 2.0, 3.0]  # sum = 6 ≠ 5
        @test f(x_invalid) == Inf
        
        x_invalid2 = [0.0, 0.0, 0.0]  # sum = 0 ≠ 5
        @test f(x_invalid2) == Inf
        
        # Test with FeasTolerance
        ε = FeasTolerance / 2
        x_near = [1.0, 2.0, 2.0 + ε]  # sum = 5 + ε, should be feasible
        @test f(x_near) ≈ 0.0
        
        # Test clearly outside tolerance
        x_outside = [1.0, 2.0, 2.1]  # sum = 5.1, clearly outside tolerance
        @test f(x_outside) == Inf
        
        # Test error for wrong dimension
        x_wrong_dim = [1.0, 2.0]  # Length 2, expected 3
        @test_throws ErrorException f(x_wrong_dim)
    end

    @testset "Function Evaluation - Vector Case" begin
        # Test constraint: x₁ + x₂ = [3, 4] where each xᵢ is a 2-element vector
        rhs = [3.0, 4.0]
        f = IndicatorSumOfNVariables(2, rhs)
        
        # Input should be 4-element vector: [x₁₁, x₁₂, x₂₁, x₂₂]
        # Constraint: [x₁₁ + x₂₁, x₁₂ + x₂₂] = [3, 4]
        
        # Satisfies constraint
        x_valid = [1.0, 2.0, 2.0, 2.0]  # x₁=[1,2], x₂=[2,2], sum=[3,4]
        @test f(x_valid) ≈ 0.0
        
        x_valid2 = [0.0, 0.0, 3.0, 4.0]  # x₁=[0,0], x₂=[3,4], sum=[3,4]
        @test f(x_valid2) ≈ 0.0
        
        # Does not satisfy constraint
        x_invalid = [1.0, 1.0, 1.0, 1.0]  # x₁=[1,1], x₂=[1,1], sum=[2,2] ≠ [3,4]
        @test f(x_invalid) == Inf
        
        # Test error for wrong first dimension
        x_wrong_dim = [1.0, 2.0, 3.0]  # Length 3, expected 4
        @test_throws ErrorException f(x_wrong_dim)
    end

    @testset "Function Evaluation - Matrix Case" begin
        # Test constraint: X₁ + X₂ + X₃ = [[1, 2], [3, 4]] where each Xᵢ is 2×2
        rhs = [1.0 2.0; 3.0 4.0]
        f = IndicatorSumOfNVariables(3, rhs)
        
        # Input should be 6×2 matrix (3 blocks of 2×2 stacked vertically)
        x_valid = [0.0 1.0; 1.0 1.0; 0.0 0.0; 2.0 3.0; 0.0 0.0; 0.0 0.0]
        # Block 1: [0,1; 1,1], Block 2: [0,0; 2,3], Block 3: [0,0; 0,0]
        # Sum: [0,1; 3,4] ≠ [1,2; 3,4] - this should be invalid
        @test f(x_valid) == Inf
        
        # Correct example
        x_valid_correct = [1.0 2.0; 0.0 0.0; 0.0 0.0; 1.0 2.0; 0.0 0.0; 2.0 2.0]
        # Block 1: [1,2; 0,0], Block 2: [0,0; 1,2], Block 3: [0,0; 2,2]
        # Sum: [1,2; 3,4] = rhs
        @test f(x_valid_correct) ≈ 0.0
    end

    @testset "Proximal Oracle - Scalar Case" begin
        # Test projection onto constraint: x₁ + x₂ + x₃ = 5
        f = IndicatorSumOfNVariables(3, 5.0)
        
        # Test projection from arbitrary point
        x_start = [1.0, 2.0, 4.0]  # sum = 7, excess = 2
        prox = proximalOracle(f, x_start)
        
        # Expected: subtract uniform shift (7-5)/3 = 2/3 from each element
        # Result: [1-2/3, 2-2/3, 4-2/3] = [1/3, 4/3, 10/3]
        expected = [1.0/3.0, 4.0/3.0, 10.0/3.0]
        @test prox ≈ expected
        @test sum(prox) ≈ 5.0  # Should satisfy constraint
        @test size(prox) == size(x_start)
        
        # Test with point already satisfying constraint
        x_valid = [1.0, 2.0, 2.0]  # sum = 5
        prox_valid = proximalOracle(f, x_valid)
        @test prox_valid ≈ x_valid  # Should remain unchanged
        
        # Test with negative values
        x_negative = [-2.0, 3.0, 6.0]  # sum = 7
        prox_negative = proximalOracle(f, x_negative)
        expected_negative = [-2.0 - 2.0/3.0, 3.0 - 2.0/3.0, 6.0 - 2.0/3.0]
        @test prox_negative ≈ expected_negative
        @test sum(prox_negative) ≈ 5.0
        
        # Test with zero target
        f_zero = IndicatorSumOfNVariables(2, 0.0)
        x_zero_test = [3.0, -1.0]  # sum = 2
        prox_zero = proximalOracle(f_zero, x_zero_test)
        # Expected: subtract 2/2 = 1 from each: [2, -2]
        @test prox_zero ≈ [2.0, -2.0]
        @test sum(prox_zero) ≈ 0.0
    end

    @testset "Proximal Oracle - Vector Case" begin
        # Test projection onto constraint: x₁ + x₂ = [3, 4]
        rhs = [3.0, 4.0]
        f = IndicatorSumOfNVariables(2, rhs)
        
        # Input: [x₁₁, x₁₂, x₂₁, x₂₂] representing x₁=[x₁₁,x₁₂], x₂=[x₂₁,x₂₂]
        x_start = [1.0, 2.0, 3.0, 4.0]  # x₁=[1,2], x₂=[3,4], sum=[4,6]
        prox = proximalOracle(f, x_start)
        
        # Residual: [4,6] - [3,4] = [1,2]
        # Shift per block: [1,2]/2 = [0.5,1]
        # Expected: x₁ - shift = [1,2] - [0.5,1] = [0.5,1]
        #          x₂ - shift = [3,4] - [0.5,1] = [2.5,3]
        # Result: [0.5, 1, 2.5, 3]
        expected = [0.5, 1.0, 2.5, 3.0]
        @test prox ≈ expected
        
        # Verify constraint is satisfied
        block1 = prox[1:2]
        block2 = prox[3:4]
        @test block1 + block2 ≈ rhs
        
        # Test in-place version
        prox_inplace = similar(x_start)
        proximalOracle!(prox_inplace, f, x_start)
        @test prox_inplace ≈ expected
        
        # Test error for wrong output type in-place
        @test_throws MethodError proximalOracle!(0.0, f, x_start)
    end

    @testset "Proximal Oracle - Matrix Case" begin
        # Test projection onto constraint: X₁ + X₂ = [[1, 2], [3, 4]]
        rhs = [1.0 2.0; 3.0 4.0]
        f = IndicatorSumOfNVariables(2, rhs)
        
        # Input: 4×2 matrix representing two 2×2 blocks stacked vertically
        x_start = [2.0 3.0; 1.0 2.0; 0.0 1.0; 4.0 3.0]
        # Block 1: [2,3; 1,2], Block 2: [0,1; 4,3]
        # Sum: [2,4; 5,5], Target: [1,2; 3,4]
        # Residual: [1,2; 2,1], Shift: [0.5,1; 1,0.5]
        
        prox = proximalOracle(f, x_start)
        
        # Expected: Block 1 - shift = [2,3; 1,2] - [0.5,1; 1,0.5] = [1.5,2; 0,1.5]
        #          Block 2 - shift = [0,1; 4,3] - [0.5,1; 1,0.5] = [-0.5,0; 3,2.5]
        expected = [1.5 2.0; 0.0 1.5; -0.5 0.0; 3.0 2.5]
        @test prox ≈ expected
        
        # Verify constraint is satisfied
        block1 = prox[1:2, :]
        block2 = prox[3:4, :]
        @test block1 + block2 ≈ rhs
        
        # Test in-place version
        prox_inplace = similar(x_start)
        proximalOracle!(prox_inplace, f, x_start)
        @test prox_inplace ≈ expected
    end

    @testset "Edge Cases" begin
        # Test with single variable (n=1)
        f_single = IndicatorSumOfNVariables(1, 3.0)
        x_single = [5.0]
        prox_single = proximalOracle(f_single, x_single)
        # With n=1, projection should give exactly the target value
        @test prox_single ≈ [3.0]
        
        # Test with large number of variables
        n_large = 100
        target_large = 50.0
        f_large = IndicatorSumOfNVariables(n_large, target_large)
        
        x_large = ones(n_large)  # sum = 100
        prox_large = proximalOracle(f_large, x_large)
        # Expected: subtract (100-50)/100 = 0.5 from each
        @test all(prox_large .≈ 0.5)
        @test sum(prox_large) ≈ target_large
        
        # Test with zero target
        f_zero_target = IndicatorSumOfNVariables(3, 0.0)
        x_zero_test = [1.0, -2.0, 4.0]  # sum = 3
        prox_zero_target = proximalOracle(f_zero_target, x_zero_test)
        # Expected: subtract 3/3 = 1 from each: [0, -3, 3]
        @test prox_zero_target ≈ [0.0, -3.0, 3.0]
        @test sum(prox_zero_target) ≈ 0.0
        
        # Test with very small vectors
        rhs_tiny = [1e-10]
        f_tiny = IndicatorSumOfNVariables(2, rhs_tiny)
        x_tiny = [1e-5, 1e-5]  # Much larger than target
        prox_tiny = proximalOracle(f_tiny, x_tiny)
        @test sum(prox_tiny) ≈ 1e-10 atol=1e-12
        
        # Test numerical stability
        rhs_large = [1e10, -1e10]
        f_stable = IndicatorSumOfNVariables(2, rhs_large)
        x_stable = [1e11, -1e11, 0.0, 0.0]
        prox_stable = proximalOracle(f_stable, x_stable)
        @test all(isfinite.(prox_stable))
        
        # Verify constraint satisfaction
        block1_stable = prox_stable[1:2]
        block2_stable = prox_stable[3:4]
        @test block1_stable + block2_stable ≈ rhs_large atol=1e-6
    end

    @testset "Mathematical Properties" begin
        # Test that projection is idempotent: proj(proj(x)) = proj(x)
        f = IndicatorSumOfNVariables(3, 10.0)
        x_test = [2.0, 3.0, 7.0]  # sum = 12
        
        proj1 = proximalOracle(f, x_test)
        proj2 = proximalOracle(f, proj1)
        @test proj1 ≈ proj2 atol=1e-10
        
        # Test that projection is non-expansive: ||proj(x) - proj(y)|| ≤ ||x - y||
        x1 = [1.0, 2.0, 3.0]
        x2 = [4.0, 1.0, 2.0]
        
        proj_x1 = proximalOracle(f, x1)
        proj_x2 = proximalOracle(f, x2)
        
        @test norm(proj_x1 - proj_x2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that projection minimizes distance to the constraint set
        x_outside = [1.0, 2.0, 10.0]  # sum = 13
        proj_x = proximalOracle(f, x_outside)
        
        # Check that proj_x satisfies constraint
        @test abs(sum(proj_x) - 10.0) < 1e-10
        
        # For any point satisfying constraint, distance should be ≥ distance to projection
        x_in_set = [2.0, 3.0, 5.0]  # sum = 10
        @test abs(sum(x_in_set) - 10.0) < 1e-10  # Verify it's in the set
        
        @test norm(x_outside - proj_x) ≤ norm(x_outside - x_in_set) + 1e-10
        
        # Test that gamma parameter doesn't affect projection (indicator functions)
        @test proximalOracle(f, x_test, 0.1) ≈ proximalOracle(f, x_test, 10.0) atol=1e-10
        
        # Test linearity property: projection preserves affine combinations in constraint set
        # If x₁ and x₂ both satisfy the constraint, then proj(αx₁ + (1-α)x₂) = αx₁ + (1-α)x₂
        x_valid1 = [2.0, 3.0, 5.0]  # sum = 10
        x_valid2 = [1.0, 4.0, 5.0]  # sum = 10
        α = 0.3
        
        combo = α * x_valid1 + (1-α) * x_valid2
        proj_combo = proximalOracle(f, combo)
        expected_combo = α * x_valid1 + (1-α) * x_valid2  # Should be unchanged
        
        @test proj_combo ≈ expected_combo atol=1e-10
        
        # Test vector case properties
        rhs_vec = [5.0, -2.0]
        f_vec = IndicatorSumOfNVariables(2, rhs_vec)
        
        x_vec_test = [1.0, 2.0, 3.0, 4.0]  # Blocks: [1,2], [3,4], sum=[4,6]
        proj_vec = proximalOracle(f_vec, x_vec_test)
        
        # Verify constraint satisfaction
        block1_vec = proj_vec[1:2]
        block2_vec = proj_vec[3:4]
        @test block1_vec + block2_vec ≈ rhs_vec atol=1e-10
        
        # Test that projection is idempotent for vector case
        proj_vec2 = proximalOracle(f_vec, proj_vec)
        @test proj_vec ≈ proj_vec2 atol=1e-10
        
        # Test uniform shift property for scalar case
        # The projection should subtract the same value from all components
        x_uniform_test = [1.0, 2.0, 3.0, 4.0]  # sum = 10
        f_uniform = IndicatorSumOfNVariables(4, 6.0)  # target = 6
        proj_uniform = proximalOracle(f_uniform, x_uniform_test)
        
        # Shift should be (10-6)/4 = 1
        expected_uniform = x_uniform_test .- 1.0
        @test proj_uniform ≈ expected_uniform
        @test sum(proj_uniform) ≈ 6.0
    end
end 