# Formulations

This page documents the problem formulation components in PDMO.jl for defining optimization problems.

## Overview

PDMO.jl provides a flexible framework for formulating multiblock optimization problems with block variables, constraints, and linear operators.

## Block Components

Core building blocks for multiblock problems:

- `BlockVariable` - Represents a block of optimization variables
- `BlockConstraint` - Represents a constraint involving multiple variable blocks

## Problem Representation

Main structures for defining optimization problems:

- `MultiblockProblem` - Core problem representation with multiple variable blocks
- `MultiblockGraph` - Graph representation of problem structure showing variable-constraint relationships

## JuMP Integration

Integration with JuMP.jl for modeling:

- `MultiblockProblemJuMP` - Interface between JuMP models and multiblock problems
- Automatic extraction of block structure from JuMP models

## Problem Transformation

Utilities for problem preprocessing and transformation:

- `MultiblockProblemScaling` - Automatic scaling of variables and constraints
- Problem transformation utilities for improved numerical stability

## Graph Algorithms

Specialized algorithms for multiblock problem graphs:

- `BipartizationAlgorithms` - Algorithms for converting general graphs to bipartite form
- `ADMMBipartiteGraph` - Bipartite graph representation optimized for ADMM algorithms

## Graph Analysis

Tools for analyzing problem structure:

- Graph connectivity analysis
- Block decomposition strategies
- Computational complexity estimation

> **Note**: Detailed API documentation with function signatures and examples will be added in a future release.
