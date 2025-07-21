# AdaPDM

This page documents the Adaptive Primal-Dual Method (AdaPDM) algorithm components in `PDMO.jl`.


## Parameters and Iteration Information
```@docs
AbstractAdaPDMParam
AdaPDMParam
AdaPDMPlusParam
MalitskyPockParam
CondatVuParam
AdaPDMIterationInfo
AdaPDMTerminationStatus
```

## Utility Functions
```@docs
computeNormEstimate
getAdaPDMName
```

## Update Functions

Algorithm-specific update functions for dual and primal solutions.

```@docs
updateDualSolution!
updatePrimalSolution!
setupInitialPrimalDualStepSize!
```

## Iteration Utilities

Common functions for iteration management and computations.

```@docs
computeAdaPDMDualResiduals!
computeAdaPDMPrimalResiduals!
computePDMResidualsAndObjective!
computePartialObjective!
computeLipschitzAndCocoercivityEstimate
prepareProximalCenterForConjugateProximalOracle!
prepareProximalCenterForPrimalUpdate!
```

## Termination Criteria

Termination checking and status management.

```@docs
AdaPDMTerminationCriteria
checkTerminationCriteria
checkOptimalTermination
checkIterationLimit
checkTimeLimit
checkNumericalError
checkUnboundedness
getTerminationStatus
```
