using Test
using PDMO
using LinearAlgebra
using SparseArrays

@testset "Constraint to Quadratic Transformation Tests" begin
    
    @testset "Identity Mappings Case (Ultra-Efficient Path)" begin
        println("    ├─ Testing identity mappings with different coefficients...")
        
        # Create a multiblock problem with identity mappings
        # Constraint: 2*x1 + 3*x2 - 0.5*x3 = [1; 4]
        mbp = MultiblockProblem()
        
        # Add block variables with different dimensions to test robustness
        addBlockVariable!(mbp, BlockVariable("x1", Zero(), Zero(), [1.0, 2.0]))     # 2D
        addBlockVariable!(mbp, BlockVariable("x2", Zero(), Zero(), [3.0, 4.0]))     # 2D  
        addBlockVariable!(mbp, BlockVariable("x3", Zero(), Zero(), [5.0, 6.0]))     # 2D
        
        # Create constraint with identity mappings having different coefficients
        constraint = BlockConstraint("identity_test")
        addBlockMappingToConstraint!(constraint, "x1", LinearMappingIdentity(2.0))
        addBlockMappingToConstraint!(constraint, "x2", LinearMappingIdentity(3.0))
        addBlockMappingToConstraint!(constraint, "x3", LinearMappingIdentity(-0.5))
        constraint.rhs = [1.0, 4.0]
        
        addBlockConstraint!(mbp, constraint)
        
        # Transform to quadratic penalty
        penalty_function = transformConstraintsToQuadraticPenalty(mbp)
        
        @test penalty_function isa QuadraticMultiblockFunction
        @test getNumberOfBlocks(penalty_function) == 3
        
        # Test at feasible point: 2*[0.5, 2] + 3*[0, 0] + (-0.5)*[0, 0] = [1, 4]
        x_feasible = NumericVariable[[0.5, 2.0], [0.0, 0.0], [0.0, 0.0]]
        penalty_feasible = penalty_function(x_feasible)
        @test penalty_feasible ≈ 0.0 atol=1e-12
        println("    │  ├─ ✅ Penalty at feasible point: $penalty_feasible")
        
        # Test at infeasible point: 2*[1, 1] + 3*[0, 0] + (-0.5)*[0, 0] = [2, 2], violation = [1, -2]
        x_infeasible = NumericVariable[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        penalty_infeasible = penalty_function(x_infeasible)
        expected_penalty = 1.0^2 + (-2.0)^2  # ||[1, -2]||² = 5
        @test penalty_infeasible ≈ expected_penalty atol=1e-12
        println("    │  ├─ ✅ Penalty at infeasible point: $penalty_infeasible (expected: $expected_penalty)")
        
        # Test gradient computation
        grad = gradientOracle(penalty_function, x_infeasible)
        
        # Expected gradients:
        # ∇_x1 = 2 * A1' * (A1*x1 + A2*x2 + A3*x3 - b) = 2 * 2*I * [1, -2] = [4, -8]
        # ∇_x2 = 2 * A2' * (A1*x1 + A2*x2 + A3*x3 - b) = 2 * 3*I * [1, -2] = [6, -12]
        # ∇_x3 = 2 * A3' * (A1*x1 + A2*x2 + A3*x3 - b) = 2 * (-0.5)*I * [1, -2] = [-1, 2]
        @test grad[1] ≈ [4.0, -8.0] atol=1e-12
        @test grad[2] ≈ [6.0, -12.0] atol=1e-12  
        @test grad[3] ≈ [-1.0, 2.0] atol=1e-12
        println("    │  ├─ ✅ Gradient computation correct")
        
        println("    │  └─ ✅ Identity mappings case passed")
    end
    
    @testset "General Matrix Mappings Case" begin
        println("    ├─ Testing general matrix mappings...")
        
        # Create a multiblock problem with matrix mappings
        mbp = MultiblockProblem()
        
        # Add block variables
        addBlockVariable!(mbp, BlockVariable("y1", Zero(), Zero(), [1.0, 2.0, 3.0]))  # 3D
        addBlockVariable!(mbp, BlockVariable("y2", Zero(), Zero(), [4.0, 5.0]))        # 2D
        
        # Create constraint with matrix mappings: A1*y1 + A2*y2 = b
        constraint = BlockConstraint("matrix_test")
        
        # A1: 2x3 matrix, A2: 2x2 matrix -> constraint dimension = 2
        A1 = sparse([1.0 2.0 1.0; 0.0 1.0 2.0])  # 2x3
        A2 = sparse([3.0 0.0; 1.0 2.0])          # 2x2
        
        addBlockMappingToConstraint!(constraint, "y1", LinearMappingMatrix(A1))
        addBlockMappingToConstraint!(constraint, "y2", LinearMappingMatrix(A2))
        constraint.rhs = [10.0, 15.0]
        
        addBlockConstraint!(mbp, constraint)
        
        # Transform to quadratic penalty
        penalty_function = transformConstraintsToQuadraticPenalty(mbp)
        
        @test penalty_function isa QuadraticMultiblockFunction
        @test getNumberOfBlocks(penalty_function) == 2
        
        # Test at a point where we can compute expected values
        # y1 = [1, 2, 3], y2 = [4, 5]
        # A1*y1 = [1*1 + 2*2 + 1*3; 0*1 + 1*2 + 2*3] = [8; 8]
        # A2*y2 = [3*4 + 0*5; 1*4 + 2*5] = [12; 14]
        # A1*y1 + A2*y2 = [8; 8] + [12; 14] = [20; 22]
        # violation = [20; 22] - [10; 15] = [10; 7]
        # penalty = 10² + 7² = 149
        
        y_test = NumericVariable[[1.0, 2.0, 3.0], [4.0, 5.0]]
        penalty_val = penalty_function(y_test)
        expected_penalty = 10.0^2 + 7.0^2  # 149
        @test penalty_val ≈ expected_penalty atol=1e-12
        println("    │  ├─ ✅ Penalty computation: $penalty_val (expected: $expected_penalty)")
        
        # Test gradient computation
        grad = gradientOracle(penalty_function, y_test)
        
        # Expected gradients:
        # violation = [10, 7]
        # ∇_y1 = 2 * A1' * violation = 2 * [1 0; 2 1; 1 2] * [10; 7] = 2 * [10; 27; 24] = [20; 54; 48]
        # ∇_y2 = 2 * A2' * violation = 2 * [3 1; 0 2] * [10; 7] = 2 * [37; 14] = [74; 28]
        @test grad[1] ≈ [20.0, 54.0, 48.0] atol=1e-12
        @test grad[2] ≈ [74.0, 28.0] atol=1e-12
        println("    │  ├─ ✅ Gradient computation correct")
        
        println("    │  └─ ✅ Matrix mappings case passed")
    end
    
    @testset "Mixed Mappings Case" begin
        println("    ├─ Testing mixed identity and matrix mappings...")
        
        # Create a problem with both identity and matrix mappings
        mbp = MultiblockProblem()
        
        addBlockVariable!(mbp, BlockVariable("z1", Zero(), Zero(), [1.0, 2.0]))
        addBlockVariable!(mbp, BlockVariable("z2", Zero(), Zero(), [3.0, 4.0]))
        
        # Mixed constraint: 2*I*z1 + A*z2 = b
        constraint = BlockConstraint("mixed_test")
        addBlockMappingToConstraint!(constraint, "z1", LinearMappingIdentity(2.0))
        
        A_matrix = sparse([1.0 0.5; 0.0 2.0])  # 2x2
        addBlockMappingToConstraint!(constraint, "z2", LinearMappingMatrix(A_matrix))
        constraint.rhs = [5.0, 10.0]
        
        addBlockConstraint!(mbp, constraint)
        
        # This should use the general path (not ultra-efficient) because not all mappings are identity
        penalty_function = transformConstraintsToQuadraticPenalty(mbp)
        
        @test penalty_function isa QuadraticMultiblockFunction
        
        # Test computation
        # z1 = [1, 2], z2 = [3, 4]  
        # 2*I*z1 = [2, 4]
        # A*z2 = [1*3 + 0.5*4; 0*3 + 2*4] = [5; 8]
        # total = [2, 4] + [5, 8] = [7, 12]
        # violation = [7, 12] - [5, 10] = [2, 2]
        # penalty = 2² + 2² = 8
        
        z_test = NumericVariable[[1.0, 2.0], [3.0, 4.0]]
        penalty_val = penalty_function(z_test)
        expected_penalty = 2.0^2 + 2.0^2  # 8
        @test penalty_val ≈ expected_penalty atol=1e-12
        println("    │  ├─ ✅ Mixed mappings penalty: $penalty_val (expected: $expected_penalty)")
        
        println("    │  └─ ✅ Mixed mappings case passed")
    end
    
    @testset "Multiple Constraints Case" begin
        println("    ├─ Testing multiple constraints...")
        
        # Create a problem with multiple constraints
        mbp = MultiblockProblem()
        
        addBlockVariable!(mbp, BlockVariable("w1", Zero(), Zero(), [1.0]))  # 1D
        addBlockVariable!(mbp, BlockVariable("w2", Zero(), Zero(), [2.0]))  # 1D
        
        # First constraint: 2*w1 + 3*w2 = 5 (identity mappings)
        constraint1 = BlockConstraint("constraint1")
        addBlockMappingToConstraint!(constraint1, "w1", LinearMappingIdentity(2.0))
        addBlockMappingToConstraint!(constraint1, "w2", LinearMappingIdentity(3.0))
        constraint1.rhs = [5.0]
        addBlockConstraint!(mbp, constraint1)
        
        # Second constraint: w1 - w2 = 1 (identity mappings)
        constraint2 = BlockConstraint("constraint2")
        addBlockMappingToConstraint!(constraint2, "w1", LinearMappingIdentity(1.0))
        addBlockMappingToConstraint!(constraint2, "w2", LinearMappingIdentity(-1.0))
        constraint2.rhs = [1.0]
        addBlockConstraint!(mbp, constraint2)
        
        penalty_function = transformConstraintsToQuadraticPenalty(mbp)
        
        # Test at infeasible point w1=1, w2=1
        # Constraint 1 violation: 2*1 + 3*1 - 5 = 0
        # Constraint 2 violation: 1*1 + (-1)*1 - 1 = -1  
        # Total penalty: 0² + (-1)² = 1
        
        w_test = NumericVariable[[1.0], [1.0]]
        penalty_val = penalty_function(w_test)
        expected_penalty = 0.0^2 + (-1.0)^2  # 1
        @test penalty_val ≈ expected_penalty atol=1e-12
        println("    │  ├─ ✅ Multiple constraints penalty: $penalty_val (expected: $expected_penalty)")
        
        println("    │  └─ ✅ Multiple constraints case passed")
    end
    
    @testset "Edge Cases" begin
        println("    ├─ Testing edge cases...")
        
        # Test with no constraints
        mbp_empty = MultiblockProblem()
        addBlockVariable!(mbp_empty, BlockVariable("x", Zero(), Zero(), [1.0]))
        
        # Should return zero function with warning
        penalty_empty = transformConstraintsToQuadraticPenalty(mbp_empty)
        @test penalty_empty isa QuadraticMultiblockFunction
        
        x_test = NumericVariable[[1.0]]
        @test penalty_empty(x_test) ≈ 0.0
        println("    │  ├─ ✅ Empty constraints case handled")
        
        # Test createFeasibilityProblem with penalty
        mbp_feas_test = MultiblockProblem()
        addBlockVariable!(mbp_feas_test, BlockVariable("x", QuadraticFunction(1), Zero(), [1.0]))
        
        constraint_feas = BlockConstraint("feas_constraint")
        addBlockMappingToConstraint!(constraint_feas, "x", LinearMappingIdentity(1.0))
        constraint_feas.rhs = [2.0]
        addBlockConstraint!(mbp_feas_test, constraint_feas)
        
        mbp_feas = createFeasibilityProblem(mbp_feas_test; penalizeConstraints=true)
        
        @test length(mbp_feas.constraints) == 0  # Constraints should be removed
        @test mbp_feas.couplingFunction !== nothing  # Should have coupling function
        @test mbp_feas.blocks[1].f isa Zero  # Objective should be zero
        
        println("    │  ├─ ✅ createFeasibilityProblem with penalty works")
        
        println("    │  └─ ✅ Edge cases passed")
    end
    
    println("    └─ ✅ All constraint transformation tests passed!")
end
