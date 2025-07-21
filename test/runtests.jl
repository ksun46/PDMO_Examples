using Test
using PDMO
include("test_helper.jl")

println("="^80)
println("🧪 Starting PDMO.jl Test Suite")
println("="^80)

# Include test files
@testset verbose=true "PDMO.jl Tests" begin
    
    # COMPONENTS TESTS
    @info "📦 COMPONENTS TESTS" description="Testing core components: BlockVariable, BlockConstraint, Functions, Mappings"
    components_testset = @testset verbose=true "Components Tests" begin
        
        @info "Testing BlockVariable functionality..."
        @testset "BlockVariable Tests" begin
            include("Components/testBlockVariable.jl")
        end
        
        @info "Testing BlockConstraint functionality..."
        @testset "BlockConstraint Tests" begin
            include("Components/testBlockConstraint.jl")
        end
        
        @info "Testing Abstract Function interface..."
        @testset "AbstractFunction Tests" begin
            include("Components/Functions/testAbstractFunction.jl")
        end
        
        @info "Testing Abstract Mapping interface..."
        @testset "AbstractMapping Tests" begin
            include("Components/Mappings/testAbstractMapping.jl")
        end
    end
    
    # Count and report components tests
    test_counter.component_tests = count_tests_in_testset(components_testset)
    @info "✅ COMPONENTS TESTS COMPLETED" tests_executed=test_counter.component_tests
    
    # FORMULATIONS TESTS
    @info "🏗️ FORMULATIONS TESTS" description="Testing problem formulations: MultiblockProblem, Graph, Bipartization Algorithms"
    formulations_testset = @testset verbose=true "Formulations Tests" begin
        
        @info "Testing MultiblockProblem formulation..."
        @testset "MultiblockProblem Tests" begin
            include("Formulations/testMultiblockProblem.jl")
        end
        
        @info "Testing MultiblockGraph representation..."
        @testset "MultiblockGraph Tests" begin
            include("Formulations/testMultiblockGraph.jl")
        end
        
        @info "Testing Bipartization Algorithms..."
        @testset "BipartizationAlgorithms Tests" begin
            include("Formulations/testBipartizationAlgorithms.jl")
        end
        
        @info "Testing ADMM Bipartite Graph..."
        @testset "AdmmBipartiteGraph Tests" begin
            include("Formulations/testAdmmBipartiteGraph.jl")
        end
    end
    
    # Count and report formulations tests
    test_counter.formulation_tests = count_tests_in_testset(formulations_testset)
    @info "✅ FORMULATIONS TESTS COMPLETED" tests_executed=test_counter.formulation_tests
    
    # SKIPPED SECTIONS REPORTING
    @info "⏸️ SKIPPED TEST SECTIONS" description="The following test sections are currently disabled"
    println("  Algorithms Tests (ADMM, AdaPDM) - 0 tests")
    println("  Utilities Tests (I/O utilities) - 0 tests")
    println("  └─ To enable these tests, uncomment the sections in runtests.jl")
    
    test_counter.algorithm_tests = 0  # Currently skipped
    test_counter.utility_tests = 0    # Currently skipped
    
    # Commented out sections for future use:
    # @testset verbose=true "Algorithms Tests" begin
    #     @info "Testing ADMM Algorithm..."
    #     @testset "ADMM Tests" begin
    #         include("Algorithms/ADMM/test_bipartite_admm.jl")
    #     end
    #     
    #     @info "Testing AdaPDM Algorithm..."
    #     @testset "AdaPDM Tests" begin
    #         include("Algorithms/AdaPDM/test_adapdm.jl")
    #     end
    # end
    
    # @testset verbose=true "Utilities Tests" begin
    #     @info "Testing I/O utilities..."
    #     @testset "I/O Tests" begin
    #         include("Util/test_io.jl")
    #     end
    # end
end

# Print final comprehensive summary
# print_final_summary()

println("\n" * "="^80)
println("✅ PDMO.jl Test Suite Complete")
println("="^80) 