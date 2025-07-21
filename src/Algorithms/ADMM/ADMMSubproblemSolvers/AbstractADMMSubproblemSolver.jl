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
