# Alternating Direction Method of Multipliers (ADMM)

## Introduction and Solution Process Overview

The Alternating Direction Method of Multipliers (ADMM) is a powerful optimization algorithm that decomposes complex problems into simpler subproblems. The original ADMM solves convex optimization problems in the standard two-block form:
```math
\begin{aligned}
\min_{x,z} \quad & f(x) + h(y) \\
\text{s.t.} \quad & Ax + By = c
\end{aligned}
```

The algorithm alternates between three updates:
```math 
\begin{align}
   x^{k+1} = & \arg\min_x  f(x) + \langle u^k, Ax + By^k - c\rangle + \frac{\rho}{2}\|Ax + By^k - c\|^2,  \notag \\
   y^{k+1} = & \arg\min_y h(y) + \langle u^k, Ax^{k+1} + By - c\rangle + \frac{\rho}{2}\|Ax^{k+1} + By - c\|^2, \notag \\
   u^{k+1} = & u^k + \rho(Ax^{k+1} + By^{k+1} - c), \notag 
\end{align}
```
where $\rho > 0$ is the penalty parameter and $u$ is the dual variable. When $f$ and $A$ (resp. $h$ and $B$) admit certain 
block-angular structures, the update of $x^{k+1}$ (resp. $y^{k+1}$) can be further parallelized. 

In practice, many optimization problems naturally arise in **multiblock form** with more than two variable blocks:
```math
\begin{aligned}
\min_{x_1,\ldots,x_n} \quad & \sum_{j=1}^n f_j(x_j) + g_j(x_j) \\
\text{s.t.} \quad & \sum_{j \in \{1,\cdots,n\}} A_{i,j} x_j = b_i,  ~ j \in \{1,\cdots, m\},
\end{aligned}
```
where $A_{i,j}$ is non-zero if and only constraint $i$ involes block $x_j$. 
```PDMO.jl``` implements a comprehensive ADMM framework designed to solve multiblock optimization problems through an automated three-stage process:
1. **User Input**: A user-defined ```MultiblockProblem``` together with a set of algorithmic options, i.e., ADMM subproblem solver, adapter, and accelerator. 
2. **Bipartization**: Automatic reformulation using graph-based algorithms to convert the input problem into bipartite form, on which standard ADMM and variants
can be readily applied. 
3. **ADMM Execution**: The selected ADMM variant solves the reformulated problem with usder-specified algorithmic options. 

We provide more details regarding these three stages in the following sections. 

## `MultiblockProblem`: A Unified Structure for Block-structued Problems

A `MultiblockProblem` is a container for multiblock optimization problems, maintaining collections of block variables and block constraints. The structure serves as the primary input to ```PDMO.jl```.

```julia
mutable struct MultiblockProblem
    blocks::Vector{BlockVariable}
    constraints::Vector{BlockConstraint}
end
```

- **`blocks`**: A vector containing all block variables in the problem
- **`constraints`**: A vector containing all block constraints connecting the variables

### `BlockVariable`

A `BlockVariable` represents an individual optimization variable block with its associated objective functions:

```julia
mutable struct BlockVariable 
    id::BlockID                    # Unique identifier (Int64 or String)
    f::AbstractFunction            # Smooth function component
    g::AbstractFunction            # Nonsmooth/proximal function component  
    val::NumericVariable           # Current variable value
end
```

**Components:**
- **`id`**: Unique identifier for the block (can be integer or string)
- **`f`**: Smooth function component $f_i$ (e.g., quadratic, affine, exponential functions)
- **`g`**: Nonsmooth function component $g_i$ handled via proximal operators (e.g., indicator functions, norms)
- **`val`**: Current value of the variable (scalar or vector)


### `BlockConstraint`

A `BlockConstraint` represents equality constraints connecting multiple block variables:

```julia
mutable struct BlockConstraint 
    id::BlockID                                    # Unique identifier
    involvedBlocks::Vector{BlockID}                # Block IDs participating in constraint
    mappings::Dict{BlockID, AbstractMapping}       # Linear mappings for each block
    rhs::NumericVariable                           # Right-hand side value
end
```

**Components:**
- **`id`**: Unique identifier for the constraint
- **`involvedBlocks`**: Vector of block IDs that participate in this constraint
- **`mappings`**: Dictionary mapping each block ID to its linear transformation
- **`rhs`**: Right-hand side of the equality constraint

Each constraint enforces the relationship:
```math
\sum_{i \in \text{involvedBlocks}} (\text{mappings}[i])(x_i) = \text{rhs}. 
```

## Graph-based Bipartization
Many optimization problems naturally arise in multiblock form where constraints involve more than two variable blocks. 
Since a direct application of ADMM to multiblick problem may fail to converge, ```PDMO.jl``` automatically converts these 
problems into bipartite form, i.e., a formulation where there are only two blocks, while each block consists of one or more sub-blocks. We briefly outline the procedures used by ```PDMO.jl```
to achieve this goal. 

To begin with, ```PDMO.jl``` will map a ```MultiblockProblem``` instance onto a graph with the following procedures:

1. For each ```BlockVariable```, introduce a new node in the graph. 
2. For each ```BlockConstraint``` involving exactly 2 blocks, add an edge between the two corresponding nodes. 
3. For each ```BlockConstraint``` involving more than 2 blocks, introduce a new node for this constraint, and connect this node with every node representing an involved block.  

In this way, a graph representation of a ```MultiblockProblem``` instance is constructed. As an illustrative example, consider a multiblock problem:
```math
\begin{aligned}
    \min_{x_1, x_2, x_3} \quad & f_1(x_1) + f_2(x_2) + f_3(x_3)\\
    \mathrm{s.t.}\quad &  A_1x_1 + A_2x_2 +A_3x_3 = a \\ 
    & B_1x_1 + B_2x_2 = b\\
    & C_2x_2 + C_3x_3 = c,
\end{aligned}
```
whose corresponding graph is show as the following: 
```@raw html
<div style="text-align: center; margin: 20px 0;">
  <svg width="400" height="300" viewBox="0 0 400 300">
    <!-- Variable and constraint nodes -->
    <circle cx="100" cy="80" r="25" fill="#ADD8E6" stroke="#000" stroke-width="2"/>
    <text x="100" y="85" text-anchor="middle" font-size="16">x₁</text>
    
    <circle cx="100" cy="220" r="25" fill="#ADD8E6" stroke="#000" stroke-width="2"/>
    <text x="100" y="225" text-anchor="middle" font-size="16">x₃</text>
    
    <circle cx="300" cy="80" r="25" fill="#ADD8E6" stroke="#000" stroke-width="2"/>
    <text x="300" y="85" text-anchor="middle" font-size="16">x₂</text>
    
    <circle cx="300" cy="220" r="25" fill="#FFA500" stroke="#000" stroke-width="2"/>
    <text x="300" y="225" text-anchor="middle" font-size="16">C</text>
    <text x="300" y="275" text-anchor="middle" font-size="12">A₁x₁ + A₂x₂ + A₃x₃ = a</text>
    
    <!-- Edges -->
    <line x1="125" y1="80" x2="275" y2="80" stroke="#000" stroke-width="2"/>
    <line x1="125" y1="220" x2="275" y2="80" stroke="#000" stroke-width="2"/>
    <line x1="275" y1="220" x2="125" y2="80" stroke="#000" stroke-width="2"/>
    <line x1="300" y1="195" x2="300" y2="105" stroke="#000" stroke-width="2"/>
    <line x1="275" y1="220" x2="125" y2="220" stroke="#000" stroke-width="2"/>
  </svg>
</div>
```
Edges $(x_1, x_2)$ and $(x_2, x_3)$ represent constraints $B_1x_1 + B_2x_2 = b$ and $C_2x_2 + C_3x_3 = c$, respectively. 
Node $C$ corresponds to a new ```BlockVariable``` that has a proximal-friendly component ```g```: the indicator function of the set 
```math 
Y = \{(y_1, y_2, y_3):~ y_1 + y_2 + y_3 = 0\}.
```
The edge between $C$ and $x_i$ represent the artifical constraint $A_ix_i - y_i = 0$ for $i \in \{1,2,3\}$. 

```PDMO.jl``` includes algorithms to convert a graph into bipartite form using a key operation called *edge subdivision*, which replaces an edge by a path of length 2. More specificly, for an edge $e$ with endpoints denoted by $x_i$ and $x_j$, introduce a new node, delete the original $e$, and connect the new node with with $n_i$ and $n_j$ respectively. Continuing with the previous example, we can remove the edge between $C$ and $x_2$, introduc a new node in green, and this new node with $C$ and $x_2$ respectively. The resulting graph becomes bipartite. 

```@raw html
<div style="text-align: center; margin: 20px 0;">
  <svg width="500" height="350" viewBox="0 0 500 350">
    <!-- Left Partition -->
    <text x="80" y="320" text-anchor="middle" font-size="16" font-weight="bold">Left Partition</text>
    <circle cx="80" cy="60" r="25" fill="#ADD8E6" stroke="#000" stroke-width="2"/>
    <text x="80" y="65" text-anchor="middle" font-size="16">x₁</text>
    <circle cx="80" cy="150" r="25" fill="#ADD8E6" stroke="#000" stroke-width="2"/>
    <text x="80" y="155" text-anchor="middle" font-size="16">x₃</text>
    <circle cx="80" cy="240" r="25" fill="#90EE90" stroke="#000" stroke-width="2"/>
    <text x="80" y="245" text-anchor="middle" font-size="12">C-x₂</text>
    
    <!-- Right Partition -->
    <text x="420" y="320" text-anchor="middle" font-size="16" font-weight="bold">Right Partition</text>
    <circle cx="420" cy="60" r="25" fill="#ADD8E6" stroke="#000" stroke-width="2"/>
    <text x="420" y="65" text-anchor="middle" font-size="16">x₂</text>
    <circle cx="420" cy="150" r="25" fill="#FFA500" stroke="#000" stroke-width="2"/>
    <text x="420" y="155" text-anchor="middle" font-size="16">C</text>
    
    <!-- Edges -->
    <line x1="105" y1="65" x2="395" y2="65" stroke="#000" stroke-width="2"/>
    <line x1="105" y1="65" x2="395" y2="155" stroke="#000" stroke-width="2"/>
    <line x1="105" y1="155" x2="395" y2="65" stroke="#000" stroke-width="2"/>
    <line x1="105" y1="155" x2="395" y2="155" stroke="#000" stroke-width="2"/>
    <line x1="105" y1="245" x2="395" y2="65" stroke="#000" stroke-width="2"/>
    <line x1="105" y1="245" x2="395" y2="155" stroke="#000" stroke-width="2"/>
    
    <!-- Partition boundary -->
    <line x1="250" y1="40" x2="250" y2="290" stroke="#888" stroke-width="3" stroke-dasharray="10,5"/>
  </svg>
</div>
```

The newly introduce node (in green) also introduce a new ```BlockVariable``` with a proximal-friendly function ```g```: the indicator function of the set of a free variable $z$, which has the same shape as $A_2x_2$ and $y_2$. The previous edge between $x_2$ and $C$, representing $A_2x_2 - y_2 = 0$, is replaced by two new edges: the one between $x_2$ and the new node represents $A_2x_2-z = 0$, and the other one between $C$ and the new node represents $y_2-z=0$. 

The edge subdivision operation essentially introduces auxiliary variables to break complicated couplings between variables and constraints. The goal is to construct an equivalent reformulation of the original problem, where all ```BlockVariable```s can be assigned either to the left or to the right, and each constraint couples exactly one ```BlockVariable``` on the left, and another on the right. Currently ```PDMO.jl``` supports the following bipartization algorithms based on edge splitting. 

| Algorithm | Procedure |
|-----------|-----------|
| ```MILP_BIPARTIZATION``` | (1) Formulate as MILP with binary variables (2) Add constraints for valid bipartition (3) Minimize operator norms and graph complexity |
| ```BFS_BIPARTIZATION``` | (1) Traverse graph level-by-level using BFS (2) Assign nodes to alternating partitions (3) Split edges when conflicts detected |
| ```DFS_BIPARTIZATION``` | (1) Traverse graph depth-first using DFS (2) Assign nodes to alternating partitions (3) Split edges when conflicts detected |
| ```SPANNING_TREE_BIPARTIZATION``` | (1) Construct spanning tree (2) 2-color the spanning tree (3) Split back edges only if endpoints have same partition |



The specific bipartization algorithm can be selected as a keyword argument to the function `runBipartiteADMM` via 
```julia 
# suppose the following arguments have been constructed: 
#  1. mbp: a MultiblockProblem of interest
#  2. param: user-specified ADMM parameters
result = runBipartiteADMM(mbp, param;
    bipartizationAlgorithm = BFS_BIPARTIZATION
    # or bipartizationAlgorithm = MILP_BIPARTIZATION
    # or bipartizationAlgorithm = DFS_BIPARTIZATION
    # or bipartizationAlgorithm = SPANNING_TREE_BIPARTIZATION
)
```


## ADMM Algorithmic Components

### Subproblem Solvers

PDMO.jl implements multiple subproblem solvers corresponding to different ADMM variants. 

#### Original ADMM Subproblem Solver (`OriginalADMMSubproblemSolver`)

The `OriginalADMMSubproblemSolver` solves the ADMM subproblems exactly, i.e., without any linearization techniques, 
```math 
  x^{k+1} = \argmin_{x} f(x) + g(x) + \langle u^k, Ax + By^k - c\rangle + \frac{\rho}{2}\|Ax+ By^k-c\|^2
```
using specialized methods based on automatic problem structure detection.
```julia
param = ADMMParam()
param.solver = OriginalADMMSubproblemSolver()
```
To handle diverse problem characteristics and computational requirements, ```OriginalADMMSubproblemSolver``` might further invoke different *specialized solvers* for different nodal subproblems: 
- **`LinearSolver`**: A linear solver based on Cholesky or LDL' decomposition for subproblems equivalent to linear systems
- **`ProximalMappingSolver`**: A solver for subproblems reducible to proximal mappings of corresponding ```g```
- **`JuMPSolver`**: A general-purpose solver using [JuMP](https://jump.dev) optimization modeling and [Ipopt](https://github.com/jump-dev/Ipopt.jl). Note: supports for function modeling in `JuMP` are limited at the moment and under active development. 

When a specialized solver cannot be determined for a nodal problem, ```PDMO.jl``` will switch to the doubly linearized solver. 

#### Doubly Linearized Solver (`DoublyLinearizedSolver`)
The `DoublyLinearizedSolver` implements the *Doubly Linearized ADMM* for updating primal variables: 
```math 
\begin{aligned}
x^{k+1} = & \mathrm{prox}_{\alpha g}(x^k - \alpha (\nabla f(x^k) + A^\top u^k + \rho A^\top(Ax^k+By^k-c))), \\
\end{aligned}
```
where $\alpha >0$ is a proximal coefficient. Since linearization is applied to the original smooth function $f$ as well as the augmented Lagrangian terms, this update only invokes gradient oracles of $f$ and proximal oracles of $g$.
See Chapter 8 of Ryu and Yin [[1]](#references) for details. The `DoublyLinearizedSolver` can be initialized as follows. 
```julia
param = ADMMParam()
param.solver = DoublyLinearizedSolver()
```


#### Adaptive Linearized Solver (`AdaptiveLinearizedSolver`)

The `AdaptiveLinearizedSolver` is a newly developed method that combines linearization with adaptive step size mechanisms for robust performance. Details of this method will be released soon. The `AdaptiveLinearizedSolver` can be initialized as follows.
```julia
param = ADMMParam()
param.solver = AdaptiveLinearizedSolver()
```

### Penalty Adapter

Penalty parameter adapters dynamically adjust the penalty parameter $\rho$ during ADMM iterations to improve convergence. `PDMO.jl` currently implements two penalty adapters: 
- Residual Balancing Adapter (`RBAdapter`): Balances primal and dual residuals by adjusting $\rho$ based on their ratio to improve convergence stability.
```julia 
param = ADMMParam() 
param.adapter = RBAdapter(
  testRatio = 10.0,   # Threshold ratio for primal and dual residuals
  adapterRatio = 2.0) # Factor by which to multiply/divide ρ 
```
- Spectral Radius Approximation Adapter (`SRAAdapter`): Uses spectral analysis of iteration history to adaptively update $\rho$ based on convergence rate estimation. See [[2]](#references) for more details. 
```julia 
param = ADMMParam() 
param.adapter = SRAAdapter(
  T=5,                   # History length
  increasingFactor=2.0,  # Factor by which to multiply ρ
  decreasingFactor=2.0)  # Factor by which to divide ρ
```

### Accelerator

Acceleration schemes enhance ADMM convergence by exploiting iteration history. `PDMO.jl` currently implements two acceleraton schemes: 

- Anderson Accelerator (`AndersonAccelerator`): See [[3]](#references) for more details. 
```julia 
param = ADMMParam()
param.accelerator = AndersonAccelerator() 
```
- Auto Halpern Accelerator (`AutoHalpernAccelerator`): A newly developed restarted Halpern acceleration scheme for ADMM. Details will be released soon. 
```julia 
param = ADMMParam()
param.accelerator = AutoHalpernAccelerator()
```

**Note**: Although the modular design enables flexible combination of algorithmic components, not all combinations of solver, adapter, and accelerator are equally effective. Some may result in slower convergence or numerical issues. Please experiment thoughtfully and validate performance for your specific use case.

### Termination Criteria

`PDMO.jl` implements comprehensive termination criteria with multiple levels:

**Level 1: Basic Termination**
- **Optimality**: Primal and dual residuals satisfy tolerances
- **Iteration limit**: Maximum iterations reached
- **Time limit**: Wall-clock time limit exceeded
- **Numerical errors**: NaN or Inf values detected

**Level 2: Advanced Problem Classification (Under active development)** 
- **Infeasibility detection**: Problem has no feasible solution
- **Unboundedness detection**: Objective function is unbounded below
- **Ill-posed problem detection**: Problem is weakly infeasible, or has non-zero duality gap

The system automatically classifies problems into categories (Case A-F) based on convergence behavior patterns, helping users understand why ADMM may not converge on certain problem instances.


```julia
param = ADMMParam()
param.enablePathologyCheck = true  # Enable advanced problem classification
result = runBipartiteADMM(mbp, param)
``` 

# References

1. Ryu, E. K., & Yin, W. (2022). *Large-scale convex optimization: algorithms & analyses via monotone operators*. Cambridge University Press.

2. McCann, M. T., & Wohlberg, B. (2024). Robust and Simple ADMM Penalty Parameter Selection. *IEEE Open Journal of Signal Processing*, 5, 402-420.

3. Pollock, S., & Rebholz, L. G. (2023). Filtering for Anderson acceleration. *SIAM Journal on Scientific Computing*, 45(4), A1571-A1590.

