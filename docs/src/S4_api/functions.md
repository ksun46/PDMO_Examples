# Functions

This page documents the function components available in PDMO.jl for modeling optimization problems.

## Overview

PDMO.jl provides a comprehensive library of functions commonly used in convex optimization, including both smooth and non-smooth functions.

## Abstract Base Types

- `AbstractFunction` - Base abstract type for all functions
- `AbstractFunctionUtil` - Utilities for function operations

## Smooth Functions

Differentiable functions with gradient information:

- `AffineFunction` - Linear and affine functions
- `QuadraticFunction` - Quadratic functions with matrix representation
- `ComponentwiseExponentialFunction` - Element-wise exponential functions
- `UserDefinedSmoothFunction` - User-defined smooth functions with custom gradients

## Non-smooth Functions

Non-differentiable functions with proximal operators:

- `ElementwiseL1Norm` - Element-wise L1 norm (absolute value)
- `MatrixNuclearNorm` - Nuclear norm for matrices
- `WeightedMatrixL1Norm` - Weighted L1 norm for matrices
- `UserDefinedProximalFunction` - User-defined functions with custom proximal operators

## Indicator Functions

Functions that enforce constraints by being zero on feasible sets and infinite elsewhere:

- `IndicatorBox` - Box constraints (upper and lower bounds)
- `IndicatorBallL2` - L2 ball constraints
- `IndicatorHyperplane` - Hyperplane constraints
- `IndicatorLinearSubspace` - Linear subspace constraints
- `IndicatorNonnegativeOrthant` - Non-negativity constraints
- `IndicatorPSD` - Positive semidefinite matrix constraints
- `IndicatorSOC` - Second-order cone constraints
- `IndicatorRotatedSOC` - Rotated second-order cone constraints
- `IndicatorSumOfNVariables` - Sum equality constraints
- `UserDefinedIndicatorFunction` - User-defined indicator functions

## Utility Functions

Special purpose and wrapper functions:

- `Zero` - Zero function (always returns 0)
- `FrobeniusNormSquare` - Squared Frobenius norm for matrices
- `WrapperScalarInputFunction` - Wrapper for scalar input functions
- `WrapperScalingTranslationFunction` - Scaling and translation wrapper

> **Note**: Detailed API documentation with function signatures and examples will be added in a future release. 