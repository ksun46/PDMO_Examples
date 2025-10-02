# PDMO.jl - **Primal-Dual Methods for Optimization**

## Overview
`PDMO.jl` is a powerful Julia framework for primal-dual multiblock optimization, built for **rapid prototyping** and **high-performance computing**.

- **Solve Complex Problems**: Model and solve problems with multiple variable blocks and linear coupling constraints. 
- **Highly Customizable**: An open-source toolkit that is easy to adapt for your applications and specific algorithms.
- **Accelerate Research**: Benchmark your methods against classic and state-of-the-art solvers.

## Problem Formulation
`PDMO.jl` presents a unified framework for formulating and solving a ```MultiblockProblem``` of the form: 

```math 
\begin{aligned}
\min_{\mathbf{x}} \quad & \sum_{j=1}^n \left( f_j(x_j) + g_j(x_j) \right)\\ 
\mathrm{s.t.} \quad  & \mathbf{A} \mathbf{x} = \mathbf{b},
\end{aligned}
```
where we have the following problem variables and data:

| **$n$ Block Variables** | **$m$ Block Constraints** | **Block Matrix of $m\times n$ Linear Operators** |
|:---:|:---:|:---:|
| $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ | $\mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$ | $\mathbf{A} = \begin{bmatrix} \mathbf{A}_{1,1} & \mathbf{A}_{1,2} & \cdots & \mathbf{A}_{1,n} \\ \mathbf{A}_{2,1} & \mathbf{A}_{2,2} & \cdots & \mathbf{A}_{2,n} \\ \vdots & \vdots &  & \vdots \\ \mathbf{A}_{m,1} & \mathbf{A}_{m,2} & \cdots & \mathbf{A}_{m,n} \end{bmatrix}$ |

More specifically, 
- For each $j\in \{1,\cdots,n\}$, a `BlockVariable` $x_j$ represents a numeric array (i.e., scalar, vector, matrix, etc.), and is associated with two objective functions: 
    - each $f_j$ is differentiable, and $f_j(\cdot)$ and $\nabla f_j(\cdot)$ are available; 
    - each $g_j$ is proximable, and $g_j(\cdot)$ and $\text{prox}_{\gamma g_j}(\cdot)$ are available.
- For each $i \in \{1,\cdots,m\}$, a `BlockConstraint` is defined by $\mathbf{A}_{i,1},\cdots, \mathbf{A}_{i,n}$, and $b_i$: 
    - the linear operator $\mathbf{A}_{i,j}$ is **non-zero** if and only if constraint $i$ involves blocks $x_j$;
    - the adjoint operator of $\mathbf{A}_{i,j}$ is available;
    - the right-hand side $b_i$ can be a numeric array of any shape. 

## Available Algorithms

`PDMO.jl` provides various algorithms to solve problems of the above form.

- **Alternating Direction Method of Multipliers (ADMM)**
  - Graph-based bipartization methods automatically generate ADMM-ready reformulations of `MultiblockProblem`.
  - Various ADMM variants are available: 
    - Original ADMM 
    - Doubly linearized ADMM 
    - Adaptive linearized ADMM 
  - Various algorithmic component can be selected: 
    - Penalty adapters, e.g., Residual Balancing, Spectral Radius Approximation
    - Accelerators, e.g., Halpern (with or without restart), Filtered Anderson

- **Adaptive Primal-Dual Method (AdaPDM)**
  - A suite of efficient and adaptive methods for problems with simpler coupling.
  ```math 
    \begin{aligned}
    \min_{\mathbf{x}} \quad & \sum_{j=1}^{n-1} \left( f_j(x_j) + g_j(x_j) \right) + g_n(\mathbf{A}_{1,1}x_1 + \cdots + \mathbf{A}_{1,n-1}x_{n-1})
    \end{aligned}
  ```
  - Various methods can be selected : 
    - Original Condat-Vũ Method (Condat 2013, Vũ 2013)
    - Adaptive Primal-Dual Method & Plus (Latafat et al. 2024)
    - Malitsky-Pock Methd (Malitsky and Pock, 2018)

## Key Features 
- 🧱 **Unified Modeling**: A versatile interface for structured problems.
- 🔄 **Automatic Decomposition**: Intelligently analyzes and reformulates problems for supported algorithms.
- 🧩 **Extensible by Design**: Easily add custom functions, constraints, and algorithms.
- 📊 **Modular Solvers**: A rich library of classic and modern algorithms.
- ⚡  **Non-Convex Ready**: Equipped with features to tackle non-convexity.


## Installation
Before official release, we recommend the following practice to download and use ```PDMO.jl```. 
Download the package and navigate to the project folder:
```bash 
cd PDMO
```

### HSL Setup (Optional)
For enhanced performance, you can optionally use linear solvers from [HSL](https://www.hsl.rl.ac.uk):

1. Obtain HSL library from [https://www.hsl.rl.ac.uk/](https://www.hsl.rl.ac.uk/)
2. Set up your HSL_jll directory structure
3. Edit `warmup.jl` and update the HSL path

### Project Setup
Run the setup script:
```bash 
julia warmup.jl
```

This will set up all required dependencies and configure HSL if available.

After successful setup:
```julia 
using PDMO
``` 

## Quick Start
### Dual Square Root LASSO
We use the Dual Square Root LASSO as a beginning example: 
```math
\begin{aligned}
    \min_{x, z}\quad & \langle b, x\rangle \\
    \mathrm{s.t.} \quad & Ax - z = 0 \\
    & \|x\|_2 \leq 1, \|z\|_{\infty} \leq \lambda,
\end{aligned}
```
where $(A, b, \lambda)$ are given problem data of proper dimensions. 

To begin with, load ```PDMO.jl``` and other necessary packages.
```julia
using PDMO
using LinearAlgebra
using SparseArrays
using Random 
```
Next generate or load your own problem data. We use synthetic data here. 
```julia
numberRows = 10 
numberColumns = 20 
A = sparse(randn(numberRows, numberColumns))
b = randn(numberColumns)
lambda = 1.0
```
Then we can generate a ```MultiblockProblem``` for the Dual Square Root LASSO problem.
```julia
mbp = MultiblockProblem()

# add x block
block_x = BlockVariable() 
block_x.f = AffineFunction(b, 0.0)    # f_1(x) = <b, x>
block_x.g = IndicatorBallL2(1.0)      # g_1(x) = indicator of L2 ball 
block_x.val = zeros(numberColumns)    # initial value
xID = addBlockVariable!(mbp, block_x) # add x block to mbp; an ID is assigned

# add z block 
block_z = BlockVariable()                              
block_z.g = IndicatorBox(-lambda * ones(numberRows), # f_2(z) = Zero() by default
    ones(numberRows) * lambda)                       # g_2(x) = indicator of box
block_z.val = zeros(numberRows)                      # initial value
zID = addBlockVariable!(mbp, block_z)                # add z block to mbp; an ID is assigned

# add constraint: Ax-z=0
constr = BlockConstraint() 
addBlockMappingToConstraint!(constr, xID, LinearMappingMatrix(A))      # specify the mapping of x
addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0)) # specify the mapping of z 
constr.rhs = zeros(numberRows)                                         # specify RHS
addBlockConstraint!(mbp, constr)                                       # add constraint to mbp
```
Next we can run different variants of ADMM: 
```julia 
# run ADMM 
param = ADMMParam() 
param.solver = OriginalADMMSubproblemSolver()
param.adapter = RBAdapter(testRatio=10.0, adapterRatio=2.0)
param.accelerator = AndersonAccelerator()
result = runBipartiteADMM(mbp, param)
```
```julia
# run Doubly Linearized ADMM
param = ADMMParam() 
param.solver = DoublyLinearizedSolver() 
result = runBipartiteADMM(mbp, param)
```
```julia
# run Adaptive Linearized ADMM
param = ADMMParam() 
param.solver = AdaptiveLinearizedSolver()
result = runBipartiteADMM(mbp, param)
```
or different adaptive primal-dual methods: 
```julia
# run AdaPDM 
paramAdaPDM = AdaPDMParam(mbp)
result = runAdaPDM(mbp, paramAdaPDM)
```
```julia
# run AdaPDMPlus 
paramAdaPDMPlus = AdaPDMPlusParam(mbp)
result = runAdaPDM(mbp, paramAdaPDMPlus)
```
```julia
# run Malitsky-Pock 
paramMalitskyPock = MalitskyPockParam(mbp)
result = runAdaPDM(mbp, paramMalitskyPock)
```
```julia
# run Condat-Vu 
paramCondatVu = CondatVuParam(mbp)
result = runAdaPDM(mbp, paramCondatVu)
```

Upon termination of the selected algorithm, one can look for primal solution and iteration information through `result.solution` and `result.iterationInfo`, respectively. 


### User Defined Smooth and Proximable Functions
In addition to a set of built-in functions whose gradient or proximal oracles have been implemented, `PDMO.jl` supports user-defined smooth and proximable functions. Consider the function 
```math
    F(x) = x_1 + |x_2| + x_3^4, ~x = [x_1, x_2, x_3]^\top \in \R^3,
```
which can be expressed as the sum of a smooth $f$ and a proximable $g$: 
```math 
    f(x) = x_1 + x_3^4, \quad g(x) = |x_2|.
```
In `PDMO.jl`, this block can be constructed as follows:
```julia
block = BlockVariable()
block.f = UserDefinedSmoothFunction(
    x -> x[1] + x[3]^4,                  # f
    x -> [1.0, 0.0, 4*x[3]^3])           # ∇f
block.g = UserDefinedProximalFunction(
    x -> abs(x[2]),                      # g
    (x, gamma) -> [                      # prox_{gamma g} 
        x[1], 
        sign(x[2]) * max(abs(x[2]) - gamma, 0.0),
        x[3]])
block.val = zeros(3)                     # initial value
```

## Documentation

For comprehensive documentation, examples, and API references, visit our [full documentation](docs/src/index.md).


## Roadmap (Work in Progress)
- 🔍 Classification and detection for pathological problems
- 🚀 Advanced acceleration techniques for first-order methods 
- 🤖 AI coding assistant for user-defined functions
- 🛣️ Parallel, distributed, and GPU support

## Contributing

`PDMO.jl` is open source and welcomes contributions! Please contact [**info@mindopt.tech**](mailto:info@mindopt.tech) for more details.

## License

`PDMO.jl` is licensed under the MIT License. 