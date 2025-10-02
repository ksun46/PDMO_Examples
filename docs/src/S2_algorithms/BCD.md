# Block Coordinate Descent (BCD)

## Introduction and Solution Process Overview

Block Coordinate Descent (BCD) is a classic algorithm for solving multiblock problems of the form 

```math
\min_{\mathbf{x}}~F(\mathbf{x}) + \sum_{j=1}^n \left( f_j(x_j) + g_j(x_j) \right),
```

where:
- The variable $\mathbf{x}$ is decomposed into $n$ blocks, i.e.,  $\mathbf{x} = [{x}^\top_1, \ldots, {x}_n^\top]^\top$.
- The function $F$ is smooth and couples all block variables. 
- For $j\in \{1,\cdots, n\}$, $f_j$ is a smooth function in variable $x_j$.
- For $j\in \{1,\cdots, n\}$, $g_j$ is a proximable function in variable $x_j$. including the domain of $x_j$ (i.e., when $g_j$ is an indicator function). 

At iteration $k$, BCD cyclically updates each block $j = 1, \ldots, n$ by solving a subproblem while fixing other blocks at their current values, until a certain termination criteria is met.

## Subproblem Solvers
In this section, we illustrate three different BCD subproblem solvers [[1]](#references), and how to specify them in `PDMO.jl`. Let
```math
F^{k+1}_j(x_j):= F(x_1^{k+1}, \cdots, x^{k+1}_{j-1}, x_j, x_{j+1}^{k}, \cdots, x_n^{k}). 
```
### Original BCD Subproblem Solver 

The original BCD solver directly minimizes the objective function with respect to block $j$ with all other blocks fixed at their latest value:

```math
x_j^{(k+1)} = \arg\min_{x_j} F^{k+1}_j(x_j) + f_j(x_j) + g_j(x_j). 
```

**Parameter Setup Example:**
```julia
using PDMO

# Define your multiblock problem (mbp)
# ...

# Set up BCD parameters

param = BCDParam(
    blockOrderRule = CyclicRule(),
    solver = BCDProximalSubproblemSolver(originalSubproblem = true),  # Original direct minimization
    dresTolL2 = 1e-6,
    dresTolLInf = 1e-6,
    maxIter = 1000,
    timeLimit = 3600.0,
    logLevel = 1
)

# Solve the problem
result = runBCD(mbp, param)
```

### Proximal BCD Subproblem Solver

The proximal BCD solver adds an additional quadratic proximal term to regularize or convexify the subproblem:

```math
x_j^{(k+1)} = \arg\min_{x_j} F^{k+1}_j(x_j) + f_j(x_j) + g_j(x_j) + \frac{L_j^{k}}{2}\|x_j - x_k^k\|^2,
```

where $L_j^{(k)} > 0$ is the proximal parameter (penalty coefficient) for block $j$ at iteration $k$. This regularization helps improve convergence properties and numerical stability.

**Parameter Setup Example:**
```julia
param = BCDParam(
    blockOrderRule = CyclicRule(),
    solver = BCDProximalSubproblemSolver(originalSubproblem = false),  # Proximal regularization
    dresTolL2 = 1e-6,
    dresTolLInf = 1e-6,
    maxIter = 1000,
    timeLimit = 3600.0,
    logLevel = 1
)
```

### Prox-linear BCD Subproblem Solver

The prox-linear solver further linearizes the coupling function $F$ and block function $f_j$ around the current point:

```math
x_j^{(k+1)} = \arg\min_{x_j} \langle \nabla F^{k+1}_j(x_j^k) + \nabla f_j(x_j^k) ,x_j - x_j^k\rangle + g_j(x_j) + \frac{L_j^{k}}{2}\|x_j - x_k^k\|^2. 
```

where $\nabla_j F^{k+1}_j(x_j^k)$ is the partial gradient of $F$ with respect to block $j$ evaluated at the current iterate. This approach is computationally efficient as it only requires gradient evaluations and proximal operators.

**Parameter Setup Example:**
```julia
param = BCDParam(
    blockOrderRule = CyclicRule(),
    solver = BCDProximalLinearSubproblemSolver(),  # Prox-linear solver
    dresTolL2 = 1e-6,
    dresTolLInf = 1e-6,
    maxIter = 1000,
    timeLimit = 3600.0,
    logLevel = 1
)
```


# References

1. Xu, Y., & Yin, W. (2012). A block coordinate descent method for regularized multi-convex optimization with applications to nonnegative tensor factorization and completion. *UCLA CAM Report 12-47*. Available at: https://ww3.math.ucla.edu/camreport/cam12-47.pdf