**Formulations**

This page documents the problem formulation utilities in PDMO.jl.

**Core Data Types**

*Block Components*

```@docs
BlockID
BlockVariable
BlockConstraint
MultiblockProblem
```

*Block Component Functions*

```@docs
addBlockVariable!
addBlockConstraint!
checkBlockVariableValidity
checkBlockConstraintValidity
checkMultiblockProblemValidity
checkMultiblockProblemFeasibility
checkCompositeProblemValidity!
```

**Graph Formulations**

*Graph Data Types*

```@docs
MultiblockGraph
NodeType
EdgeType
Node
Edge
```

*Graph Construction Functions*

```@docs
createNodeID
createEdgeID
```

*Graph Analysis Functions*

```@docs
numberNodes
numberEdges
numberEdgesByTypes
getNodelNeighbors
isMultiblockGraphBipartite
isMultiblockGraphConnected
```

**Bipartization Algorithms**

*Bipartization Types*

```@docs
BipartizationAlgorithm
BfsBipartization
MilpBipartization
DfsBipartization
SpanningTreeBipartization
```

*Bipartization Functions*

```@docs
getBipartizationAlgorithmName
```

**ADMM Bipartite Graph**

*ADMM Data Types*

```@docs
ADMMBipartiteGraph
ADMMNode
ADMMEdge
```

*ADMM Construction Functions*

```@docs
createADMMNodeID
createADMMEdgeID
```

**JuMP Interface**

*JuMP Interface Functions*

```@docs
solveMultiblockProblemByJuMP
isSupportedObjectiveFunction
isSupportedProximalFunction
unwrapFunction
addBlockVariableToJuMPModel!
```
