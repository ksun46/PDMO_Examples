using Test
using PDMO
using LinearAlgebra
using SparseArrays
using Random
include("../../test_helper.jl")

@testset "IndicatorPSD Tests" begin
    @testset "Constructor" begin
        # Valid construction
        @test_nowarn IndicatorPSD(2)
        
        # Invalid dimension
        @test_throws AssertionError IndicatorPSD(0)
    end

    @testset "Function Evaluation" begin
        f = IndicatorPSD(2)
        
        # Test symmetric PSD matrix
        X1 = [2.0 1.0; 1.0 2.0]
        @test f(X1) ≈ 0.0
        
        # Test symmetric PSD sparse matrix
        X2 = sparse([2.0 1.0; 1.0 2.0])
        @test f(X2) ≈ 0.0
        
        # Test non-symmetric matrix
        X3 = [2.0 1.0; -1.0 2.0]
        @test_throws ErrorException f(X3)
        
        # Test non-PSD matrix
        X4 = [1.0 2.0; 2.0 1.0]
        @test f(X4) == Inf
        
        # Test wrong dimensions
        X5 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        @test_throws AssertionError f(X5)
        
        # Test nearly PSD matrix (within tolerance)
        X6 = [1.0 0.0; 0.0 -1e-10]
        @test f(X6) ≈ 0.0
    end

    @testset "Proximal Operator" begin
        f = IndicatorPSD(3)
        
        # Test projection of PSD matrix
        X1 = [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 1.0]
        Y1 = similar(X1)
        proximalOracle!(Y1, f, X1, 1.0)
        @test Y1 ≈ X1
        
        # Test projection of non-PSD matrix
        X2 = [1.0 0.0 0.0; 0.0 -2.0 0.0; 0.0 0.0 1.0]
        Y2 = similar(X2)
        proximalOracle!(Y2, f, X2, 1.0)
        @test Y2[1,1] ≈ 1.0
        @test Y2[2,2] ≈ 0.0  # Negative eigenvalue should become 0
        @test Y2[3,3] ≈ 1.0
        @test issymmetric(Y2)
        @test minimum(eigvals(Y2)) >= -1e-10
        
        # Test sparse matrix projection
        X3 = sparse([1.0 0.0; 0.0 -1.0])
        f_small = IndicatorPSD(2)
        Y3 = similar(X3)
        proximalOracle!(Y3, f_small, X3, 1.0)
        @test Y3[1,1] ≈ 1.0
        @test Y3[2,2] ≈ 0.0
        @test issparse(Y3)
        
        # Test random matrix
        Random.seed!(123)
        X4 = randn(3,3)
        X4 = X4 + X4'  # Make symmetric
        Y4 = similar(X4)
        proximalOracle!(Y4, f, X4, 1.0)
        @test issymmetric(Y4)
        @test minimum(eigvals(Y4)) >= -1e-10
    end

    @testset "Non-mutating Proximal Operator" begin
        f = IndicatorPSD(2)
        X = [1.0 0.0; 0.0 -1.0]
        Y = proximalOracle(f, X, 1.0)
        @test Y[1,1] ≈ 1.0
        @test Y[2,2] ≈ 0.0
        @test typeof(Y) == typeof(X)
        
        # Test sparse input
        X_sparse = sparse(X)
        Y_sparse = proximalOracle(f, X_sparse, 1.0)
        @test issparse(Y_sparse)
        @test Y_sparse[1,1] ≈ 1.0
        @test Y_sparse[2,2] ≈ 0.0
    end

    @testset "Edge Cases" begin
        f = IndicatorPSD(2)
        
        # Test zero matrix
        X1 = zeros(2,2)
        @test f(X1) ≈ 0.0
        Y1 = proximalOracle(f, X1, 1.0)
        @test Y1 ≈ X1
        
        # Test nearly symmetric matrix
        X2 = [1.0 1e-11; 0.0 1.0]
        @test_throws ErrorException f(X2)
        
        # Test matrix with very small eigenvalues
        X3 = [1e-11 0.0; 0.0 1e-11]
        @test f(X3) ≈ 0.0
        
        # Test sparse zero matrix
        X4 = spzeros(2,2)
        @test f(X4) ≈ 0.0
        Y4 = proximalOracle(f, X4, 1.0)
        @test issparse(Y4)
        @test Y4 ≈ X4
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorPSD) == true
        @test isSmooth(IndicatorPSD) == false
        @test isConvex(IndicatorPSD) == true
        @test isSet(IndicatorPSD) == true
    end
end 