using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "Zero Function Tests" begin
    @testset "Constructor" begin
        f = Zero()
        @test f isa Zero
        @test f isa AbstractFunction
    end

    @testset "Function Traits" begin
        @test isProximal(Zero) == true
        @test isSmooth(Zero) == true
        @test isConvex(Zero) == true
        @test isSet(Zero) == false
    end

    @testset "Function Evaluation" begin
        f = Zero()
        
        # Test scalar input
        @test f(5.0) ≈ 0.0
        @test f(-3.0) ≈ 0.0
        @test f(0.0) ≈ 0.0
        
        # Test vector input
        x = randn(5)
        @test f(x) ≈ 0.0
        
        # Test matrix input
        X = randn(3, 4)
        @test f(X) ≈ 0.0
        
        # Test sparse input
        X_sparse = sprandn(10, 10, 0.3)
        @test f(X_sparse) ≈ 0.0
    end

    @testset "Gradient Oracle" begin
        f = Zero()
        
        # Test scalar input (non-mutating)
        @test gradientOracle(f, 5.0) ≈ 0.0
        @test gradientOracle(f, -2.0) ≈ 0.0
        
        # Test vector input (non-mutating)
        x = randn(5)
        grad = gradientOracle(f, x)
        @test grad ≈ zeros(5)
        @test size(grad) == size(x)
        
        # Test matrix input (non-mutating)
        X = randn(3, 4)
        grad_X = gradientOracle(f, X)
        @test grad_X ≈ zeros(3, 4)
        @test size(grad_X) == size(X)
        
        # Test in-place gradient for vectors
        x = randn(5)
        grad = similar(x)
        gradientOracle!(grad, f, x)
        @test grad ≈ zeros(5)
        
        # Test in-place gradient for matrices
        X = randn(3, 4)
        grad_X = similar(X)
        gradientOracle!(grad_X, f, X)
        @test grad_X ≈ zeros(3, 4)
        
        # Test error for scalar in-place
        @test_throws ErrorException gradientOracle!(0.0, f, 5.0)
    end

    @testset "Proximal Oracle" begin
        f = Zero()
        
        # Test scalar input (non-mutating)
        @test proximalOracle(f, 5.0) ≈ 5.0
        @test proximalOracle(f, -2.0) ≈ -2.0
        @test proximalOracle(f, 0.0) ≈ 0.0
        
        # Test with different gamma values for scalar
        @test proximalOracle(f, 3.0, 2.0) ≈ 3.0
        @test proximalOracle(f, -1.0, 0.5) ≈ -1.0
        
        # Test vector input (non-mutating)
        x = randn(5)
        prox = proximalOracle(f, x)
        @test prox ≈ x
        @test size(prox) == size(x)
        
        # Test matrix input (non-mutating)
        X = randn(3, 4)
        prox_X = proximalOracle(f, X)
        @test prox_X ≈ X
        @test size(prox_X) == size(X)
        
        # Test in-place proximal for vectors
        x = randn(5)
        x_copy = copy(x)
        prox = similar(x)
        proximalOracle!(prox, f, x)
        @test prox ≈ x_copy
        @test x ≈ x_copy  # Original should be unchanged
        
        # Test in-place proximal for matrices
        X = randn(3, 4)
        X_copy = copy(X)
        prox_X = similar(X)
        proximalOracle!(prox_X, f, X)
        @test prox_X ≈ X_copy
        @test X ≈ X_copy  # Original should be unchanged
        
        # Test error for scalar in-place
        @test_throws ErrorException proximalOracle!(0.0, f, 5.0)
    end

    @testset "Edge Cases" begin
        f = Zero()
        
        # Test with very large values
        large_val = 1e10
        @test f(large_val) ≈ 0.0
        @test gradientOracle(f, large_val) ≈ 0.0
        @test proximalOracle(f, large_val) ≈ large_val
        
        # Test with very small values
        small_val = 1e-15
        @test f(small_val) ≈ 0.0
        @test gradientOracle(f, small_val) ≈ 0.0
        @test proximalOracle(f, small_val) ≈ small_val
        
        # Test with empty arrays
        empty_vec = Float64[]
        @test f(empty_vec) ≈ 0.0
        @test gradientOracle(f, empty_vec) ≈ Float64[]
        @test proximalOracle(f, empty_vec) ≈ Float64[]
        
        # Test with sparse matrices
        X_sparse = sprandn(5, 5, 0.3)
        @test f(X_sparse) ≈ 0.0
        grad_sparse = gradientOracle(f, X_sparse)
        @test grad_sparse ≈ spzeros(5, 5)
        prox_sparse = proximalOracle(f, X_sparse)
        @test prox_sparse ≈ X_sparse
    end
end 