using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "AbstractMultiblockFunction Tests" begin
    
    @testset "Type Hierarchy Tests" begin
        # Test that AbstractMultiblockFunction is properly defined
        @test AbstractMultiblockFunction isa Type
        @test AbstractMultiblockFunction <: AbstractFunction
        
        # Test that it's abstract
        @test_throws MethodError AbstractMultiblockFunction()
    end
    
    @testset "Default Trait Tests" begin
        # Test default trait implementations
        @test isSmooth(AbstractMultiblockFunction) == true
        @test isConvex(AbstractMultiblockFunction) == false  # Should be false by default
        
        # Test instance-based trait calls work (using QuadraticMultiblockFunction as example)
        if @isdefined(QuadraticMultiblockFunction)
            Q = sparse([2.0 1.0; 1.0 3.0])
            q = [1.0, 2.0]
            r = 3.0
            blockDims = [1, 1]
            
            f = QuadraticMultiblockFunction(Q, q, r, blockDims)
            @test isSmooth(f) == true
            @test f isa AbstractMultiblockFunction
            @test f isa AbstractFunction
        end
    end
    
    @testset "Interface Method Definitions" begin
        # Test that interface methods exist by checking they are defined
        @test isdefined(PDMO, :partialGradientOracle!)
        @test isdefined(PDMO, :partialGradientOracle)
        @test isdefined(PDMO, :gradientOracle!)
        @test isdefined(PDMO, :gradientOracle)
        @test isdefined(PDMO, :getNumberOfBlocks)
        @test isdefined(PDMO, :validateBlockDimensions)
    end
    
    @testset "Abstract Method Error Tests" begin
        # Define a minimal concrete type for testing error conditions
        struct TestMultiblockFunction <: AbstractMultiblockFunction
            blockDims::Vector{Int}
        end
        
        # Override getNumberOfBlocks to make it minimally functional
        PDMO.getNumberOfBlocks(f::TestMultiblockFunction) = length(f.blockDims)
        
        test_func = TestMultiblockFunction([2, 3])
        x_blocks = NumericVariable[[1.0, 2.0], [3.0, 4.0, 5.0]]
        x_concat = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test that abstract methods throw appropriate errors
        @test_throws ErrorException test_func(x_blocks)
        
        grad_blocks = NumericVariable[similar(x_blocks[1], Float64), similar(x_blocks[2], Float64)]
        @test_throws ErrorException gradientOracle!(grad_blocks, test_func, x_blocks)
        
        grad_concat = similar(x_concat, Float64)
        @test_throws ErrorException gradientOracle!(grad_concat, test_func, x_concat)
        
        partial_grad = similar(x_blocks[1], Float64)
        @test_throws ErrorException partialGradientOracle!(partial_grad, test_func, x_blocks, 1)
    end
    
    @testset "Default Implementation Tests" begin
        # Test default implementations that should work
        struct TestValidationFunction <: AbstractMultiblockFunction
            blockDims::Vector{Int}
        end
        
        PDMO.getNumberOfBlocks(f::TestValidationFunction) = length(f.blockDims)
        
        test_func = TestValidationFunction([2, 3])
        x_blocks = NumericVariable[[1.0, 2.0], [3.0, 4.0, 5.0]]
        
        # Test default validateBlockDimensions (should not throw)
        @test validateBlockDimensions(test_func, x_blocks) === nothing
        
        # Test getNumberOfBlocks works
        @test getNumberOfBlocks(test_func) == 2
    end
    
    @testset "Gradient Oracle Interface Consistency" begin
        # Test that all gradient oracle methods have consistent signatures
        # This ensures the interface is properly defined
        
        # Test Vector{NumericVariable} signatures
        @test hasmethod(gradientOracle!, (Vector{NumericVariable}, AbstractMultiblockFunction, Vector{NumericVariable}))
        @test hasmethod(gradientOracle, (AbstractMultiblockFunction, Vector{NumericVariable}))
        
        # Test NumericVariable signatures  
        @test hasmethod(gradientOracle!, (NumericVariable, AbstractMultiblockFunction, NumericVariable))
        @test hasmethod(gradientOracle, (AbstractMultiblockFunction, NumericVariable))
        
        # Test partial gradient signatures
        @test hasmethod(partialGradientOracle!, (NumericVariable, AbstractMultiblockFunction, Vector{NumericVariable}, Int))
        @test hasmethod(partialGradientOracle, (AbstractMultiblockFunction, Vector{NumericVariable}, Int))
    end
    
    @testset "Exported Symbols Tests" begin
        # Test that key symbols are properly exported
        @test @isdefined(AbstractMultiblockFunction)
        @test @isdefined(partialGradientOracle!)
        @test @isdefined(partialGradientOracle)
        @test @isdefined(gradientOracle!)  # Should be exported via AbstractFunction
        @test @isdefined(gradientOracle)   # Should be exported via AbstractFunction
        @test @isdefined(getNumberOfBlocks)
        @test @isdefined(validateBlockDimensions)
        @test @isdefined(isSmooth)
        @test @isdefined(isConvex)
    end
    
    @testset "Type Stability Tests" begin
        # Test that the interface promotes type stability
        if @isdefined(QuadraticMultiblockFunction)
            Q = sparse([2.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 1.5])
            q = [1.0, 2.0, 0.5]
            r = 3.0
            blockDims = [2, 1]
            
            f = QuadraticMultiblockFunction(Q, q, r, blockDims)
            x_blocks = NumericVariable[[1.0, 2.0], [3.0]]
            x_concat = [1.0, 2.0, 3.0]
            
            # Test return types are as expected
            @test typeof(f(x_blocks)) == Float64
            @test typeof(getNumberOfBlocks(f)) == Int
            @test typeof(isSmooth(f)) == Bool
            @test typeof(isConvex(f)) == Bool
            
            # Test gradient oracle return types
            grad_blocks = gradientOracle(f, x_blocks)
            @test grad_blocks isa Vector{NumericVariable}
            @test length(grad_blocks) == 2
            
            grad_concat = gradientOracle(f, x_concat)
            @test grad_concat isa Vector{Float64}
            @test length(grad_concat) == 3
            
            partial_grad = partialGradientOracle(f, x_blocks, 1)
            @test partial_grad isa Vector{Float64}
            @test length(partial_grad) == 2
        end
    end
end