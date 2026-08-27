# PlasmaCore.jl

Core data structures and numerical infrastructure for plasma physics simulations. Provides backend-agnostic tensor fields, simulation grids, phase-space distribution data, and spectral differential operators. Physics-level logic (particle species, moments, advection) lives in higher-level packages such as [bslLD](https://github.com/mraeth/bslLD).

## Overview

| Module area | What it provides |
|---|---|
| **Fields** | `TensorField`, `ScalarField`, `VectorField`, `MatrixField` — N-dimensional field types with arithmetic |
| **Grids** | `CartGrid`, `PolarGrid` — phase-space grids with spatial and velocity axes |
| **Distribution** | `DistributionGrid` — phase-space array f(x,v) backed by a `ScalarField` |
| **Spectral operators** | `differentiate`, `grad`, `div`, `curl` — allocation-free spectral derivatives |
| **Time** | `SimulationTime` — iteration-based time stepper with progress display |
| **Execution** | CPU/CUDA/ROCm/Metal backend switching; `backend_copy`, `backend_synchronize!` |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mraeth/PlasmaCore.jl")
```

GPU backends are optional weak dependencies. Load them before calling the corresponding `use_*!()` function:

```julia
using Metal        # or CUDA, AMDGPU
using PlasmaCore
PlasmaCore.use_metal!()   # all subsequent field constructions land on the GPU
```

## Quick start

```julia
using PlasmaCore

# 1D-1V Cartesian grid: x ∈ [0, 10], v ∈ [-6, 6]
grid = Grid([0.0, -6.0], [10.0, 6.0], [64, 128], 1)

# Fields on the spatial grid
phi  = empty_scalarfield(grid)          # ScalarField over x
E    = empty_vectorfield(grid)          # VectorField, one component per velocity dim

# Spectral derivative
dphidx = differentiate(phi, grid, 1)   # allocation-free on repeated calls

# Phase-space distribution (data layer only — no mass/charge)
using PlasmaCore: DistributionGridImpl, Cart, ScalarField
raw = zeros(length(grid.xaxes[1]), length(grid.vaxes[1]))
sf  = ScalarField(raw)
f   = DistributionGridImpl{1, 1, Cart, typeof(sf)}(sf)
f.data   # raw 2D array
f.field  # the underlying ScalarField
```

## Types

### Fields

```julia
TensorField{DT, N, AT, Rank, NF}
```

Base field type. `DT` is the element type, `N` the array rank, `AT` the storage array type, `Rank` the tensor rank (0, 1, or 2), and `NF` the number of components.

| Alias | Rank | Components | Typical use |
|---|---|---|---|
| `ScalarField{DT,N,AT}` | 0 | 1 | potential, density |
| `VectorField{DT,N,AT,NF}` | 1 | NF | electric field, current |
| `MatrixField{DT,N,AT,NF}` | 2 | NF = NS² | pressure tensor |

Constructors accept raw arrays or vectors/matrices of arrays:

```julia
phi = ScalarField(zeros(64))
E   = VectorField([zeros(64), zeros(64)])
Pi  = MatrixField([zeros(64) for _ in 1:9])   # 3×3
```

Field arithmetic (`+`, `-`, `*`, scalar broadcast, matrix-vector) is supported directly:

```julia
E2 = 2.0 * E
J  = R * E        # R::AbstractMatrix, E::VectorField → VectorField
```

### Grid

```julia
Grid(etaMin, etaMax, N, nx; b0=1.0, Bdir=3, type=Cart)
```

`etaMin`, `etaMax`, `N` are vectors over all dimensions (spatial first, then velocity). `nx` is the number of spatial dimensions.

```julia
# 2D-2V Cartesian: x ∈ [0,2π]², v ∈ [-6,6]²
grid = Grid([0.0, 0.0, -6.0, -6.0], [2π, 2π, 6.0, 6.0], [64, 64, 64, 64], 2)

grid.xaxes  # Tuple of spatial StepRangeLen axes
grid.vaxes  # Tuple of velocity StepRangeLen axes
grid.b0     # background field magnitude
grid.Bdir   # background field direction (1, 2, or 3)
```

`CartGrid` = `Grid{...,Cart}`, `PolarGrid` = `Grid{...,Polar}` (polar velocity coordinates vp, φ).

### Distribution

`DistributionGrid` packs f(x,v) into a single `ScalarField` over the combined phase-space array. The first `NX` dimensions are spatial, the last `NV` are velocity:

```julia
DistributionGrid{DT, NX, NV, NXNV, ID, AT}
# aliased as:
DistributionGrid1d1v{T,ID,AT}   # 2D array
DistributionGrid1d2v{T,ID,AT}   # 3D array
DistributionGrid2d2v{T,ID,AT}   # 4D array
```

Access:
- `f.data`  — the raw `NXNV`-dimensional array
- `f.field` — the underlying `ScalarField`

Physics attributes (mass, charge, initialization, moments) are not part of `DistributionGrid`. Those belong in a higher-level `Species` type defined by the consuming package.

### SimulationTime

```julia
t = SimulationTime(dt, final_T)
```

Iterable and indexable. Use `advance!` inside a hand-rolled loop, or `for τ in t` for a simple time sweep:

```julia
t = SimulationTime(0.01, 10.0)
while continue_advection(t, show_progress=true)
    # ... solve one step ...
    advance!(t, gyro_frequency)
end

t.current_T  # elapsed simulation time
t.phase      # accumulated gyro-phase
t.step       # step counter
```

## Spectral operators

All operators are allocation-free on repeated calls for the same grid and field shape. `SpectralWorkspace` objects (FFT plans + wavenumber arrays) are created once and cached globally per thread.

```julia
dphidx = differentiate(phi, grid, 1)    # ∂φ/∂x₁
grad_phi = grad(phi, grid)              # ∇φ  → VectorField
div_E    = div(E, grid)                 # ∇·E  → ScalarField
curl_E   = curl(E, grid)               # ∇×E  → ScalarField or VectorField

# Lower-level: forward/inverse FFT over spatial dimensions only
phi_hat = fft_spatial(phi.data, grid)
```

Wavenumber access:
```julia
k  = spectral_wavenumbers(phi.data, grid, 1)   # 1D wavenumber vector
k2 = spectral_wavenumber_squared(phi, grid)     # |k|²  (for Poisson solvers)
```

## Backend management

```julia
PlasmaCore.use_cpu!()      # default
PlasmaCore.use_cuda!()     # requires `using CUDA` first
PlasmaCore.use_amdgpu!()   # requires `using AMDGPU` first
PlasmaCore.use_metal!()    # requires `using Metal` first; data is downcast to Float32

PlasmaCore.cuda_available()    # → Bool
PlasmaCore.metal_available()   # → Bool

# After switching backends, transfer existing data:
f_gpu = backend_copy(f)        # deep-copies any field or DistributionGrid to current backend
backend_synchronize!()         # flush GPU command queue (no-op on CPU)
```

`backend_array(x)` moves a raw array to the current backend. All field constructors call it internally.

## Field allocation helpers

```julia
empty_scalarfield(grid)              # zero ScalarField over spatial axes
empty_vectorfield(grid)              # zero VectorField, NV components
empty_vectorfield(grid, n)           # zero VectorField, n components
empty_matrixfield(grid, NS)          # zero NS×NS MatrixField
zero_vectorfield_like(vf)            # zero VectorField matching vf
zero_scalarfield_like(sf)            # zero ScalarField matching sf
vectorfield_from_spatial_components(cs)   # pad 1–3 ScalarFields to a 3-component VF
zero_vectorfield3(grid)              # 3-component zero VectorField
background_field(grid)               # uniform B-field at grid.b0 in direction grid.Bdir
```

## Field solver interface

Implement `AbstractFieldSolver` to plug in a Maxwell or Poisson solver:

```julia
struct MyFieldSolver <: AbstractFieldSolver end
# implement: solve!(::MyFieldSolver, moments::Moments, fields::FieldSolution, t) = ...
```

`Moments(rho, J, Pi_diff)` holds particle-moment inputs; `FieldSolution(E, B)` holds EM field outputs.

## Design notes

- **No exports**: all names are accessed as `PlasmaCore.X` or via explicit `using PlasmaCore: X`.
- **Float32 on Metal**: `use_metal!()` installs an allocator that downcasts `Float64` arrays to `Float32` on upload. Design simulation code with `DT` type parameters to run on any precision.
- **Cached spectral workspaces**: keyed by array size, element type, and grid delta. No FFT plan is rebuilt on repeated calls.
- **Adapt-compatible**: `Grid` and all `TensorField` subtypes implement `Adapt.adapt_structure`, so they can be passed directly into GPU kernels via KernelAbstractions.
