using Base.Threads 
"""
    AbstractADMMSubproblemSolver

Abstract base type for ADMM subproblem solvers.

ADMM subproblem solvers are responsible for solving the primal variable update steps
in the ADMM algorithm. Each solver implements a specific strategy for handling the
optimization subproblems that arise in the ADMM decomposition.

**Mathematical Background**

The ADMM algorithm alternates between updating primal variables (x, z) and dual variables (λ).
The primal subproblems take the general form:

```math
\\min_x f(x) + g(x) + ⟨λ, Ax + Bz - c⟩ + \\frac{ρ}{2}\\|Ax + Bz - c\\|_2^2
```

which can be rewritten as:

```math
\\min_x f(x) + g(x) + ⟨λ, Ax⟩ + \\frac{ρ}{2}(x^T A^T A x + 2⟨A^T(Bz - c), x⟩ + \\text{const})
```

**Required Interface Methods**

All concrete subproblem solvers must implement:

1. `initialize!(solver, admmGraph, info)`: Initialize solver with problem structure
2. `solve!(solver, nodeID, accelerator, admmGraph, info, isLeft, enableParallel)`: Solve subproblem for a specific node
3. `updateDualResidualsInBuffer!(solver, info, admmGraph, accelerator)`: Compute dual residuals
4. `update!(solver, info, admmGraph, rhoUpdated)`: Update solver state when parameters change

**Available Solver Types**

- `OriginalADMMSubproblemSolver`: Exact solution using specialized solvers (JuMP, proximal, linear)
- `DoublyLinearizedSolver`: Linearized approach with separate stepsizes for left/right nodes
- `AdaptiveLinearizedSolver`: Adaptive linearized method with dynamic stepsize selection

**Solver Selection Guidelines**

- **OriginalADMMSubproblemSolver**: Use when exact solutions are needed and subproblems have known structure
- **DoublyLinearizedSolver**: Use for large-scale problems where exact solutions are expensive
- **AdaptiveLinearizedSolver**: Use when optimal convergence rates are critical and problem geometry varies

**Performance Considerations**

- **Exact solvers**: Higher per-iteration cost but fewer iterations
- **Linearized solvers**: Lower per-iteration cost but more iterations
- **Adaptive solvers**: Moderate per-iteration cost with optimal convergence rates

**Example Usage**

```julia
# Initialize solver
solver = DoublyLinearizedSolver(dualStepsize=1.0)
initialize!(solver, admmGraph, info)

# Solve subproblems for all nodes
for nodeID in keys(admmGraph.nodes)
    isLeft = nodeID in admmGraph.left
    solve!(solver, nodeID, accelerator, admmGraph, info, isLeft)
end

# Update dual residuals
updateDualResidualsInBuffer!(solver, info, admmGraph, accelerator)
```

**Implementation Notes**

- Solvers should handle both left and right nodes in bipartite graphs
- Buffer management is crucial for performance
- Parallel computation should be supported when beneficial
- State updates are needed when penalty parameters change
"""
abstract type AbstractADMMSubproblemSolver end 

"""
    EdgeData

Data structure for storing preprocessed edge information used by ADMM subproblem solvers.

This structure precomputes and stores frequently used linear algebra operations
for each edge in the ADMM bipartite graph. By preprocessing these operations,
solvers can avoid repeated computations during iterative solving.

**Mathematical Background**

For an edge with constraint `A₁x₁ + A₂x₂ = b`, the following operations are precomputed:
- `A₁ᵀb`, `A₂ᵀb`: Adjoint mappings applied to the right-hand side
- `A₁ᵀA₁`, `A₂ᵀA₂`: Self-adjoint mappings (Hessian-like terms)
- `A₁ᵀA₂`: Cross-adjoint mapping between nodes

**Fields**

- `mapping1AdjRhs::NumericVariable`: A₁ᵀb (adjoint of first mapping applied to RHS)
- `mapping2AdjRhs::NumericVariable`: A₂ᵀb (adjoint of second mapping applied to RHS)
- `mapping1AdjSelf::AbstractMapping`: A₁ᵀA₁ (self-adjoint of first mapping)
- `mapping2AdjSelf::AbstractMapping`: A₂ᵀA₂ (self-adjoint of second mapping)
- `mapping1AdjMapping2::AbstractMapping`: A₁ᵀA₂ (cross-adjoint between mappings)

**Usage Context**

This structure is used by:
- `OriginalADMMSubproblemSolver`: For exact subproblem formulation
- Other solvers that need precomputed adjoints for efficiency

**Performance Benefits**

- Avoids repeated adjoint computations during iterations
- Enables efficient quadratic form evaluations
- Reduces memory allocations in hot paths

**Example**

```julia
# Create edge data during solver initialization
edgeData = EdgeData(admmEdge)

# Access precomputed values
rhs_term = edgeData.mapping1AdjRhs  # A₁ᵀb
hessian_term = edgeData.mapping1AdjSelf  # A₁ᵀA₁
```

**Memory Considerations**

- Memory usage scales with problem size and number of edges
- Precomputed mappings may require significant storage for large problems
- Trade-off between memory usage and computational efficiency
"""
mutable struct EdgeData 
    mapping1AdjRhs::NumericVariable        # A'b 
    mapping2AdjRhs::NumericVariable        # B'b 
    mapping1AdjSelf::AbstractMapping     # A'A 
    mapping2AdjSelf::AbstractMapping     # B'B 
    mapping1AdjMapping2::AbstractMapping # A'B 

    """
        EdgeData(edge::ADMMEdge)

    Construct edge data by precomputing adjoint operations for an ADMM edge.

    **Arguments**
    - `edge::ADMMEdge`: The ADMM edge containing mappings and right-hand side

    **Computation**
    - Computes `A₁ᵀb` and `A₂ᵀb` where b is the constraint RHS
    - Constructs `A₁ᵀA₁` and `A₂ᵀA₂` self-adjoint mappings
    - Constructs `A₁ᵀA₂` cross-adjoint mapping

    **Performance**
    - All computations are done once during initialization
    - Subsequent access is O(1) for precomputed values
    - Self-adjoint mappings enable efficient quadratic form evaluations
    """
    function EdgeData(edge::ADMMEdge)
        new(adjoint(edge.mappings[edge.nodeID1], edge.rhs), 
            adjoint(edge.mappings[edge.nodeID2], edge.rhs),  
            adjoint(edge.mappings[edge.nodeID1], edge.mappings[edge.nodeID1]), 
            adjoint(edge.mappings[edge.nodeID2], edge.mappings[edge.nodeID2]), 
            adjoint(edge.mappings[edge.nodeID1], edge.mappings[edge.nodeID2]))
    end 
end 

"""
    initialize!(solver::AbstractADMMSubproblemSolver, 
               admmGraph::ADMMBipartiteGraph, 
               info::ADMMIterationInfo)

Initialize the ADMM subproblem solver with problem data and algorithm state.

This method is called once before the main ADMM iterations begin. It allows solvers
to perform preprocessing, validate problem structure, allocate workspace, and set up
any solver-specific data structures.

**Arguments**
- `solver::AbstractADMMSubproblemSolver`: The solver instance to initialize
- `admmGraph::ADMMBipartiteGraph`: The ADMM bipartite graph structure containing nodes and edges
- `info::ADMMIterationInfo`: Initial iteration information and algorithm state
- `logLevel::Int64`: Logging level
**Required Implementation**
Every concrete ADMM subproblem solver MUST implement this method. The implementation
should handle:

1. **Problem Structure Analysis**: Examine the bipartite graph structure and node/edge properties
2. **Edge Data Preprocessing**: Pre-compute expensive adjoint operations using `EdgeData`
3. **Workspace Allocation**: Pre-allocate buffers for linear coefficients and temporary variables
4. **Specialized Solver Selection**: Choose appropriate specialized solvers for each node type
5. **Validation**: Check that the solver can handle the given problem structure


**Error Handling**
- Return `false` or throw descriptive errors for unsupported problem structures
- Validate bipartite graph consistency and node/edge compatibility
- Check for required dependencies (e.g., JuMP, specialized solvers)

**Performance Considerations**
- Pre-compute all `EdgeData` structures to avoid repeated adjoint computations
- Allocate all workspace buffers to minimize allocations during iterations
- Cache problem structure information for efficient access during solving

See also: `solve!`, `update!`, `updateDualResidualsInBuffer!`, `EdgeData`
"""
function initialize!(solver::AbstractADMMSubproblemSolver, 
                    admmGraph::ADMMBipartiteGraph, 
                    info::ADMMIterationInfo, 
                    logLevel::Int64)
    error("AbstractADMMSubproblemSolver: initialize! is not implemented for $(typeof(solver))")
end

"""
    solve!(solver::AbstractADMMSubproblemSolver,
           nodeID::String, 
           accelerator::AbstractADMMAccelerator,
           admmGraph::ADMMBipartiteGraph, 
           info::ADMMIterationInfo, 
           isLeft::Bool,
           enableParallel::Bool = false)

Solve the ADMM subproblem for a specific node in the bipartite graph.

This is the core method that every concrete ADMM subproblem solver MUST implement.
It solves the primal variable update step for the specified node while keeping
variables from other nodes fixed at their current values.

**Mathematical Problem**
The method solves the ADMM subproblem:
```math
\\min_{x_i} f_i(x_i) + g_i(x_i) + ⟨λ, A_i x_i⟩ + \\frac{ρ}{2}\\|A_i x_i + \\sum_{j≠i} A_j x_j - c\\|_2^2
```

which can be rewritten as:
```math
\\min_{x_i} f_i(x_i) + g_i(x_i) + ⟨A_i^T(λ + ρ(\\sum_{j≠i} A_j x_j - c)), x_i⟩ + \\frac{ρ}{2}x_i^T A_i^T A_i x_i
```

**Arguments**
- `solver::AbstractADMMSubproblemSolver`: The solver instance
- `nodeID::String`: Identifier of the node to solve for
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (affects dual variable handling)
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information and solutions
- `isLeft::Bool`: Whether the node is on the left side of the bipartite graph
- `enableParallel::Bool`: Whether to enable parallel computation (default: false)

**Required Implementation Behavior**
1. **Extract Current State**: Get current primal and dual variables from `info`
2. **Compute Linear Coefficients**: Calculate augmented Lagrangian linear terms
3. **Handle Acceleration**: Special processing for accelerated dual variables
4. **Solve Subproblem**: Use solver-specific method to find optimal node value
5. **Update Solution**: Store new value in `info.primalSol[nodeID]` and previous in `info.primalSolPrev[nodeID]`

**Example Implementation**
```julia
function solve!(solver::MyConcreteSolver, nodeID::String, accelerator::AbstractADMMAccelerator,
               admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo, isLeft::Bool, enableParallel::Bool = false)
    
    # Get penalty parameter
    rho = info.rhoHistory[end][1]
    
    # Compute augmented Lagrangian linear coefficient
    linearCoeff = computeAugmentedLagrangianLinearCoeff(solver, nodeID, accelerator, admmGraph, info, isLeft)
    
    # Solve using specialized solver
    info.primalSolPrev[nodeID] = copy(info.primalSol[nodeID])
    info.primalSol[nodeID] = solveSpecializedSubproblem(solver, nodeID, linearCoeff, rho, admmGraph)
end
```

**Acceleration Handling**
- **Standard ADMM**: Use current dual variables λ
- **Anderson Acceleration**: Use converter output ζ for right nodes
- **Other Accelerators**: Follow accelerator-specific protocols

**Side Effects**
- MUST update `info.primalSol[nodeID]` with the new solution
- MUST store previous solution in `info.primalSolPrev[nodeID]`
- MAY use solver buffers for intermediate computations

See also: `initialize!`, `update!`, `updateDualResidualsInBuffer!`, `AbstractADMMAccelerator`
"""
function solve!(solver::AbstractADMMSubproblemSolver,
               nodeID::String, 
               accelerator::AbstractADMMAccelerator,
               admmGraph::ADMMBipartiteGraph, 
               info::ADMMIterationInfo, 
               isLeft::Bool,
               enableParallel::Bool = false)
    error("AbstractADMMSubproblemSolver: solve! is not implemented for $(typeof(solver))")
end

"""
    updateDualResidualsInBuffer!(solver::AbstractADMMSubproblemSolver,
                                info::ADMMIterationInfo,
                                admmGraph::ADMMBipartiteGraph, 
                                accelerator::AbstractADMMAccelerator)

Compute dual residuals and store them in the iteration info buffers.

This method is called after primal variable updates to compute the dual residuals
for convergence checking. The dual residual measures how much the dual variables
have changed between iterations.

**Mathematical Background**
The dual residual for ADMM is typically:
```math
\\text{dual residual} = ρ \\sum_i A_i^T (A_i x_i^{k+1} - A_i x_i^k)
```

**Arguments**
- `solver::AbstractADMMSubproblemSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information containing solutions
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (may affect residual computation)

**Required Implementation Behavior**
1. **Compute Node-wise Residuals**: Calculate dual residual contribution from each node
2. **Aggregate Residuals**: Combine contributions across all nodes and edges
3. **Store in Buffers**: Update `info.dualResidualBuffer` with computed values
4. **Handle Acceleration**: Account for any acceleration-specific modifications

**Example Implementation**
```julia
function updateDualResidualsInBuffer!(solver::MyConcreteSolver, info::ADMMIterationInfo,
                                     admmGraph::ADMMBipartiteGraph, accelerator::AbstractADMMAccelerator)
    
    rho = info.rhoHistory[end][1]
    
    # Initialize residual buffer
    fill!(info.dualResidualBuffer, 0.0)
    
    # Compute residual for each node
    for (nodeID, node) in admmGraph.nodes
        primalDiff = info.primalSol[nodeID] - info.primalSolPrev[nodeID]
        
        # Accumulate dual residual contributions
        for edgeID in node.edgeIDs
            edge = admmGraph.edges[edgeID]
            mapping = edge.mappings[nodeID]
            adjointContrib = rho * adjoint(mapping, mapping(primalDiff))
            info.dualResidualBuffer[nodeID] += adjointContrib
        end
    end
end
```

**Performance Considerations**
- Reuse pre-allocated buffers to minimize memory allocations
- Leverage precomputed `EdgeData` structures for efficient adjoint operations
- Consider parallel computation for large problems when `enableParallel` is true

**Convergence Integration**
The computed dual residuals are used by the main ADMM algorithm for:
- Convergence checking against tolerance criteria
- Adaptive penalty parameter updates
- Progress monitoring and logging

See also: `solve!`, `update!`, `ADMMIterationInfo`, `EdgeData`
"""
function updateDualResidualsInBuffer!(solver::AbstractADMMSubproblemSolver,
                                     info::ADMMIterationInfo,
                                     admmGraph::ADMMBipartiteGraph, 
                                     accelerator::AbstractADMMAccelerator)
    error("AbstractADMMSubproblemSolver: updateDualResidualsInBuffer! is not implemented for $(typeof(solver))")
end

"""
    update!(solver::AbstractADMMSubproblemSolver, 
           info::ADMMIterationInfo, 
           admmGraph::ADMMBipartiteGraph, 
           rhoUpdated::Bool)

Update the solver state based on algorithm progress and parameter changes.

This method is called periodically during ADMM iterations to allow solvers to
adapt their behavior based on convergence progress, penalty parameter updates,
or other algorithmic changes.

**Arguments**
- `solver::AbstractADMMSubproblemSolver`: The solver instance to update
- `info::ADMMIterationInfo`: Current iteration information including convergence history
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure (for reference)
- `rhoUpdated::Bool`: Whether the penalty parameter ρ has been updated since last call

**Optional Implementation**
This method is optional for concrete solvers. Many solvers implement this as a no-op,
which is appropriate for static solvers that don't adapt their behavior.

**Common Update Scenarios**
1. **Penalty Parameter Updates**: When `rhoUpdated = true`, solvers may need to:
   - Update precomputed matrices or factorizations
   - Adjust internal tolerances
   - Recompute cached quantities that depend on ρ

2. **Adaptive Behavior**: Based on algorithm progress:
   - Adjust step sizes for linearized solvers
   - Modify solver tolerances based on convergence rate
   - Switch between different solving strategies

**Example Implementation**
```julia
function update!(solver::AdaptiveADMMSolver, info::ADMMIterationInfo, 
                admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    
    # Handle penalty parameter updates
    if rhoUpdated
        newRho = info.rhoHistory[end][1]
        # Update cached factorizations that depend on ρ
        updateCachedFactorizations!(solver, newRho)
    end
    
    # Adaptive step size adjustment
    if length(info.lagrangianObj) > 10
        recentProgress = info.lagrangianObj[end-9] - info.lagrangianObj[end]
        if recentProgress > 0.01
            solver.stepSize = min(solver.stepSize * 1.1, solver.maxStepSize)
        else
            solver.stepSize = max(solver.stepSize * 0.9, solver.minStepSize)
        end
    end
    
    # Adjust tolerances based on residuals
    if !isempty(info.presL2) && info.presL2[end] < 1e-4
        solver.subproblemTolerance = min(solver.subproblemTolerance, info.presL2[end] / 100)
    end
end
```

**Update Timing**
- Called after each complete ADMM iteration (all nodes updated)
- Called whenever penalty parameters are updated
- Should be computationally lightweight to avoid overhead

**State Modification Guidelines**
- Modify solver internal state as needed
- Do NOT modify `admmGraph` (problem structure should remain constant)
- May read from `info` but avoid modifying historical data
- Update cached quantities that depend on algorithm parameters

See also: `initialize!`, `solve!`, `updateDualResidualsInBuffer!`, `ADMMIterationInfo`
"""
function update!(solver::AbstractADMMSubproblemSolver, 
                info::ADMMIterationInfo, 
                admmGraph::ADMMBipartiteGraph, 
                rhoUpdated::Bool)
    error("AbstractADMMSubproblemSolver: update! is not implemented for $(typeof(solver))")
end

include("OriginalADMMSubproblemSolver.jl")
include("DoublyLinearizedSolver.jl")
include("AdaptiveLinearizedSolver.jl")

"""
    getADMMSubproblemSolverName(solver::AbstractADMMSubproblemSolver)

Get a string identifier for the subproblem solver type.

**Arguments**
- `solver::AbstractADMMSubproblemSolver`: The solver instance

**Returns**
- `String`: A string identifier for the solver type:
  - `"ORIGINAL_ADMM_SUBPROBLEM_SOLVER"` for `OriginalADMMSubproblemSolver`
  - `"DOUBLY_LINEARIZED_SOLVER"` for `DoublyLinearizedSolver`
  - `"ADAPTIVE_LINEARIZED_SOLVER"` for `AdaptiveLinearizedSolver`
  - `"UNKNOWN_SUBPROBLEM_SOLVER"` for unrecognized types

**Example**
```julia
solver = DoublyLinearizedSolver()
name = getADMMSubproblemSolverName(solver)  # Returns "DOUBLY_LINEARIZED_SOLVER"
```

**Usage**
This function is commonly used for:
- Logging and debugging output
- Performance tracking and comparison
- Algorithm selection in experiments
- Configuration management
- Result reporting and analysis
"""
function getADMMSubproblemSolverName(solver::AbstractADMMSubproblemSolver)
    if typeof(solver) == OriginalADMMSubproblemSolver
        return "ORIGINAL_ADMM_SUBPROBLEM_SOLVER"
    elseif typeof(solver) == DoublyLinearizedSolver
        return "DOUBLY_LINEARIZED_SOLVER"
    elseif typeof(solver) == AdaptiveLinearizedSolver
        return "ADAPTIVE_LINEARIZED_SOLVER"
    else 
        return "UNKNOWN_SUBPROBLEM_SOLVER"
    end 
end 
