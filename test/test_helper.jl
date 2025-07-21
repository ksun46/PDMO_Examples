# Guard against multiple inclusions
if !@isdefined(TEST_HELPER_LOADED)
    const TEST_HELPER_LOADED = true

    using Test
    using LinearAlgebra
    using SparseArrays
    using Random
    using Printf
    using PDMO

    # Set seed for reproducible tests
    Random.seed!(42)

    # Test tolerances
    const TEST_TOL = 1e-6
    const STRICT_TOL = 1e-10

    """
    # Test Output Style Guide

    For consistent test reporting across the PDMO.jl test suite:

    ## Output Hierarchy:
    1. **Top-level banners**: `println` with emoji and `=` borders
       - Example: `println("🧪 Starting PDMO.jl Test Suite")`
    
    2. **Major section headers**: `@info` with emoji and structured data
       - Example: `@info "📦 COMPONENTS TESTS" description="Testing core components"`
    
    3. **Sub-section progress**: `@info` for test announcements
       - Example: `@info "Testing BlockVariable functionality..."`
    
    4. **Test details**: `println` with tree structure (`├─`, `│`, `└─`)
       - Example: `println("    ├─ Testing BlockVariable constructors...")`
       - Example: `println("    │  ├─ ✅ checkBlockVariableValidity: result")`
       - Example: `println("    └─ ✅ BlockVariable creation tests passed")`
    
    5. **Important results**: `@info` with structured data
       - Example: `@info "✅ Test completed successfully" result=value count=n`
    
    6. **Warnings/Expected failures**: `println` with ⚠️ emoji
       - Example: `println("    │  ├─ ⚠️ Expected failure: $error")`

    ## Emoji Usage:
    - ✅ Success/completion
    - ⚠️ Warning/expected failure  
    - ❌ Error/disabled
    - 🧪 Testing/experiments
    - 📦 Components
    - 🏗️ Formulations
    - ⏸️ Skipped/paused
    """

    # Test counting utilities
    mutable struct TestCounter
        component_tests::Int
        formulation_tests::Int
        algorithm_tests::Int
        utility_tests::Int
        total_tests::Int
        
        TestCounter() = new(0, 0, 0, 0, 0)
    end

    # Global test counter
    const test_counter = TestCounter()

    function count_tests_in_testset(ts::Test.DefaultTestSet)
        """Recursively count tests in a testset"""
        count = 0
        for result in ts.results
            if isa(result, Test.DefaultTestSet)
                count += count_tests_in_testset(result)
            elseif isa(result, Test.Pass) || isa(result, Test.Fail) || isa(result, Test.Error)
                count += 1
            end
        end
        return count
    end

    function print_final_summary()
        """Print the final test summary"""
        total = test_counter.component_tests + test_counter.formulation_tests + 
                test_counter.algorithm_tests + test_counter.utility_tests
        
        println("\n" * "="^80)
        @info "🎯 FINAL TEST SUMMARY" components=test_counter.component_tests formulations=test_counter.formulation_tests algorithms=test_counter.algorithm_tests utilities=test_counter.utility_tests total_active=(test_counter.component_tests + test_counter.formulation_tests)
        println("="^80)
    end

    # Common test data generators
    function generate_test_sparse_matrix(n::Int, m::Int, density::Float64 = 0.1)
        """Generate a random sparse matrix for testing"""
        nnz_target = max(1, round(Int, n * m * density))
        I_vals = rand(1:n, nnz_target)
        J_vals = rand(1:m, nnz_target)
        V_vals = randn(nnz_target)
        return sparse(I_vals, J_vals, V_vals, n, m)
    end

    function generate_test_qp_data(n::Int)
        """Generate test data for quadratic programming problems"""
        Q = sparse(randn(n, n))
        Q = Q' * Q + 0.01 * I  # Make positive definite
        c = randn(n)
        return Q, c
    end

    function generate_test_lp_data(n::Int, m::Int)
        """Generate test data for linear programming problems"""
        A = generate_test_sparse_matrix(m, n, 0.3)
        b = randn(m)
        c = randn(n)
        return A, b, c
    end

    function is_approximately_equal(x, y, tol = TEST_TOL)
        """Check if two values are approximately equal within tolerance"""
        return abs(x - y) <= tol
    end

    function is_approximately_zero(x, tol = TEST_TOL)
        """Check if a value is approximately zero"""
        return abs(x) <= tol
    end

    # Export test utilities
    export TEST_TOL, STRICT_TOL
    export generate_test_sparse_matrix, generate_test_qp_data, generate_test_lp_data
    export is_approximately_equal, is_approximately_zero
    export test_counter, count_tests_in_testset, print_final_summary

end 