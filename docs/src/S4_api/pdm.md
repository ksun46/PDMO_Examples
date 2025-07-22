# AdaPDM

This page documents the Adaptive Primal-Dual Method (AdaPDM) algorithm components in `PDMO.jl`.

## Overview

AdaPDM is a family of adaptive primal-dual algorithms that automatically adjust step sizes for optimal convergence. PDMO.jl provides several variants and supporting utilities.

## Parameters and Iteration Information

Algorithm parameter configuration:

- `AbstractAdaPDMParam` - Base abstract type for all AdaPDM parameters
- `AdaPDMParam` - Standard AdaPDM algorithm parameters
- `AdaPDMPlusParam` - AdaPDM+ variant with enhanced features
- `MalitskyPockParam` - Parameters for Malitsky-Pock algorithm
- `CondatVuParam` - Parameters for Condat-Vu algorithm
- `AdaPDMIterationInfo` - Iteration-specific information and statistics
- `AdaPDMTerminationStatus` - Status indicators for algorithm termination

## Core Algorithm Functions

Main algorithmic components:

- `computeNormEstimate` - Estimates operator norms for step size computation
- `getAdaPDMName` - Returns the name of the current AdaPDM variant

## Update Functions

Algorithm-specific update procedures:

- `updateDualSolution!` - Updates the dual variable in each iteration
- `updatePrimalSolution!` - Updates the primal variable in each iteration
- `setupInitialPrimalDualStepSize!` - Initializes step sizes for primal and dual variables

## Iteration Utilities

Supporting functions for iteration management:

- `computeAdaPDMDualResiduals!` - Computes dual residuals for convergence checking
- `computeAdaPDMPrimalResiduals!` - Computes primal residuals for convergence checking
- `computePDMResidualsAndObjective!` - Computes both residuals and objective value
- `computePartialObjective!` - Computes partial objective function values
- `computeLipschitzAndCocoercivityEstimate` - Estimates Lipschitz and cocoercivity constants
- `prepareProximalCenterForConjugateProximalOracle!` - Prepares centers for conjugate proximal operations
- `prepareProximalCenterForPrimalUpdate!` - Prepares centers for primal updates

## Termination Management

Convergence and termination checking:

- `AdaPDMTerminationCriteria` - Defines termination criteria
- `checkTerminationCriteria` - Main termination checking function
- `checkOptimalTermination` - Checks for optimal solution conditions
- `checkIterationLimit` - Checks iteration count limits
- `checkTimeLimit` - Checks time-based termination
- `checkNumericalError` - Detects numerical instabilities
- `checkUnboundedness` - Detects unbounded problems
- `getTerminationStatus` - Returns current termination status

> **Note**: Detailed API documentation with function signatures and examples will be added in a future release.
