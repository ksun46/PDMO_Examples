# Mappings API

This document describes the mappings available in the optimization framework.

## Abstract Mapping Interface

```@docs
AbstractMapping
```

## Interface Functions

All mapping types implement function call operators:
- `(L::ConcreteMapping)(x::NumericVariable)` - Out-of-place application
- `(L::ConcreteMapping)(x::NumericVariable, ret::NumericVariable, add::Bool = false)` - In-place application

All mapping types implement adjoint operators that extend Julia's `adjoint` and `adjoint!` functions:
- `adjoint!(L::ConcreteMapping, y::NumericVariable, ret::NumericVariable, add::Bool = false)` - In-place adjoint
- `adjoint(L::ConcreteMapping, y::NumericVariable)` - Out-of-place adjoint

```@docs
operatorNorm2
```
## Basic Mapping Types

### NullMapping

```@docs
NullMapping
```

### LinearMappingIdentity

```@docs
LinearMappingIdentity
```

### LinearMappingExtraction

```@docs
LinearMappingExtraction
```

### LinearMappingMatrix

```@docs
LinearMappingMatrix
```

