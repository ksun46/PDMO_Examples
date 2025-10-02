# BCD

This page documents the Block Coordinate Descent (BCD) algorithm components in `PDMO.jl`.


## Parameters and Iteration Information

```@docs
BCDParam
BCDIterationInfo
BCDTerminationStatus
```

## Block Update Order Rules

Block selection strategies for determining which blocks to update and in what order.

```@docs
AbstractBlockUpdateOrder
CyclicRule
updateBlockOrder!
```

## Subproblem Solvers

Different strategies for solving the block-wise optimization subproblems.

### Base Types and Interface

```@docs
AbstractBCDSubproblemSolver
initialize!
solve!
updateDualResidual!
getBCDSubproblemSolverName
```

### Concrete Solver Implementations

```@docs
BCDProximalSubproblemSolver
BCDProximalLinearSubproblemSolver
```

## Termination Criteria

Convergence checking and termination status management.

```@docs
BCDTerminationCriteria
```

## Utility Functions

Logging and iteration management utilities.

```@docs
BCDLog
```
