# ADMM

This page documents the Alternating Direction Method of Multipliers (ADMM) algorithm components in PDMO.jl.

## Overview

PDMO.jl provides a comprehensive ADMM implementation with the following key components:

## Parameters and Iteration Information
- `ADMMParam` - Main parameter configuration for ADMM algorithms
- `ADMMIterationInfo` - Stores iteration-specific information and statistics

## Subproblem Solvers

Various implementations for solving ADMM subproblems:

- `AbstractADMMSubproblemSolver` - Base abstract type for all ADMM subproblem solvers
- `OriginalADMMSubproblemSolver` - Standard ADMM subproblem solver
- `SpecializedOriginalADMMSubproblemSolver` - Optimized versions for specific problem types
- `LinearSolver` - For linear subproblems
- `JuMPSolver` - Interface to JuMP optimization models
- `ProximalMappingSolver` - Uses proximal mappings
- `DoublyLinearizedSolver` - Linearized approximation approach
- `AdaptiveLinearizedSolver` - Adaptive linearization strategy

## Adapters

Penalty parameter adaptation strategies:

- `AbstractADMMAdapter` - Base type for adaptation strategies
- `RBAdapter` - Residual balancing adaptation
- `SRAAdapter` - Spectral radius adaptation

## Accelerators

Acceleration techniques for improving convergence:

- `AbstractADMMAccelerator` - Base type for acceleration methods
- `AndersonAccelerator` - Anderson acceleration scheme
- `AutoHalpernAccelerator` - Automatic Halpern acceleration

> **Note**: Detailed API documentation with function signatures and examples will be added in a future release.
