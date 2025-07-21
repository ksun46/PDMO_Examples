using Test
using PDMO
using LinearAlgebra
include("../test_helper.jl")

# Import the validation function directly from Bipartization
import PDMO: checkBlockVariableValidity

@testset "BlockVariable Creation" begin
    println("    ├─ Testing BlockVariable constructors...")
    # Test basic construction with default constructor
    bv = BlockVariable(1)
    
    @test bv.id == 1
    @test bv.f isa Zero
    @test bv.g isa Zero
    @test bv.val == 0.0
    
    # Test with string ID
    bv2 = BlockVariable("test_block")
    @test bv2.id == "test_block"
    @test bv2.f isa Zero
    @test bv2.g isa Zero
    println("    └─ ✅ BlockVariable creation tests passed")
end

@testset "BlockVariable Operations" begin
    println("    ├─ Testing BlockVariable operations...")
    bv1 = BlockVariable(1)
    bv2 = BlockVariable(2)
    
    # Test value assignment for scalar
    test_val = 5.0
    bv1.val = test_val
    @test bv1.val ≈ test_val
    
    # Test copying scalar values
    bv2.val = bv1.val
    @test bv2.val ≈ bv1.val
    println("    └─ ✅ BlockVariable operations tests passed")
end

@testset "BlockVariable with Array Values" begin
    println("    ├─ Testing BlockVariable with array values...")
    bv = BlockVariable(1)
    
    # Test with array values
    n = 5
    test_vals = randn(n)
    bv.val = test_vals
    @test bv.val ≈ test_vals
    @test length(bv.val) == n
    println("    └─ ✅ BlockVariable array values tests passed")
end

@testset "BlockVariable Properties" begin
    println("    ├─ Testing BlockVariable properties and norms...")
    bv = BlockVariable(1)
    
    # Test with array data
    n = 20
    test_data = randn(n)
    bv.val = test_data
    
    # Test norm calculations
    @test norm(bv.val) ≈ norm(test_data)
    @test norm(bv.val, 1) ≈ norm(test_data, 1)
    @test norm(bv.val, Inf) ≈ norm(test_data, Inf)
    println("    └─ ✅ BlockVariable properties tests passed")
end

@testset "BlockVariable Validation" begin
    println("    ├─ Testing BlockVariable validation function...")
    # Test the validation function - now guaranteed to be defined
    bv = BlockVariable(1)
    
    # Test that the function is now defined
    @test @isdefined(checkBlockVariableValidity) == true
    println("    │  ├─ checkBlockVariableValidity function is available")
    
    # Test with Zero functions (should pass validation)
    try
        result = checkBlockVariableValidity(bv)
        @test result isa Bool
        @test result == true  # Should pass with Zero functions since they are valid
        println("    │  ├─ ✅ checkBlockVariableValidity with Zero functions: $result (expected true)")
    catch e
        # This should not happen with valid Zero functions
        @test false  # Force test failure if exception occurs
        println("    │  ├─ ❌ checkBlockVariableValidity unexpectedly failed with Zero functions: $e")
    end
    
    # Test with valid functions (if available)
    if @isdefined(QuadraticFunction) && @isdefined(IndicatorNonnegativeOrthant)
        bv_valid = BlockVariable(2)
        # Create a simple quadratic function: f(x) = 0.5 * x'*I*x  
        bv_valid.f = QuadraticFunction(sparse([1.0;;]), [0.0], 0.0)
        bv_valid.g = IndicatorNonnegativeOrthant()
        bv_valid.val = [1.0]
        
        result_valid = checkBlockVariableValidity(bv_valid)
        @test result_valid isa Bool
        @info "✅ checkBlockVariableValidity with valid functions" result=result_valid
    else
        println("    │  ├─ ⚠️ QuadraticFunction or IndicatorNonnegativeOrthant not available for positive validation test")
    end
    println("    └─ ✅ BlockVariable validation tests completed")
end 