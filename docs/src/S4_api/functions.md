# Functions

This page documents the function types used in the PDMO optimization framework.

## Abstract Function Interface

```@docs
AbstractFunction
NumericVariable
isSmooth
isProximal
isConvex
isSet
```

## First Order Oracles

```@docs
proximalOracle
proximalOracle!
gradientOracle
gradientOracle!
proximalOracleOfConjugate
proximalOracleOfConjugate!
estimateLipschitzConstant
```

## Functions implemented in `PDMO.jl`
### Basic Functions
```@docs
Zero
AffineFunction
QuadraticFunction
```

### Norm Functions

```@docs
ElementwiseL1Norm
FrobeniusNormSquare
MatrixNuclearNorm
WeightedMatrixL1Norm
```

### Indicator Functions

```@docs
IndicatorBallL2
IndicatorBox
IndicatorHyperplane
IndicatorLinearSubspace
IndicatorNonnegativeOrthant
IndicatorPSD
IndicatorRotatedSOC
IndicatorSOC
IndicatorSumOfNVariables
```

### User-Defined Functions

```@docs
ComponentwiseExponentialFunction
UserDefinedProximalFunction
UserDefinedSmoothFunction
```

### Wrapper Functions

```@docs
WrapperScalarInputFunction
WrapperScalingTranslationFunction
``` 