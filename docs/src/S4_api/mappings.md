# Mappings

This page documents the linear mapping components in PDMO.jl for representing linear operators between variable blocks.

## Overview

PDMO.jl provides efficient implementations of linear mappings that connect different variable blocks in multiblock optimization problems.

## Abstract Base Type

- `AbstractMapping` - Base abstract type for all linear mappings

## Linear Mapping Types

Core linear mapping implementations:

- `LinearMappingMatrix` - Dense matrix representation of linear operators
- `LinearMappingIdentity` - Identity mapping for efficiency
- `LinearMappingExtraction` - Extraction/selection mappings for subsets of variables

## Mapping Operations

Standard operations supported by all mappings:

- Forward application: `A * x` (matrix-vector multiplication)
- Adjoint application: `A' * y` (adjoint/transpose operation)
- Composition: combining multiple mappings
- Scaling: scalar multiplication of mappings

## Efficient Implementations

Specialized implementations for common cases:

- Sparse matrix support for large-scale problems
- Identity mappings with O(1) storage
- Block-diagonal structures
- Extraction operators for variable selection

## Integration with Block Structure

How mappings work within the multiblock framework:

- Connecting variable blocks to constraint blocks
- Automatic adjoint computation
- Memory-efficient storage and computation
- Support for both dense and sparse representations

> **Note**: Detailed API documentation with function signatures and examples will be added in a future release.

