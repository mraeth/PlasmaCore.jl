# First-derivative operators along one spatial axis: central finite differences (`FD`)
# and the FFT derivative (`Spectral`), used by `differentiate!`/`differentiate` and `grad!`/`grad`.

abstract type DerivativeOperator end

"""
    Spectral()

FFT derivative along one axis (periodic). The default operator of `differentiate` and `grad`.
"""
struct Spectral <: DerivativeOperator end

"""
    FD(order, bc = Periodic())

Central finite-difference first derivative of even `order` (stencil of `order + 1` nodes),
`∂f/∂x_i ≈ Σ_k c_k (f_{i+k} - f_{i-k}) / Δx`. `FD(8)` is bsl6d's 9-point stencil
`(4/5, -1/5, 4/105, -1/280)`. Nodes beyond the axis ends follow `bc`, `Periodic()` or
`Mirror()`; the axis needs more than `order/2` nodes.
"""
struct FD{M,BC} <: DerivativeOperator
    coeffs::NTuple{M,Float64}   # c_k, k = 1:M
    bc::BC
end

# c_k = (-1)^(k+1) (M!)^2 / (k (M-k)! (M+k)!), exact in rationals and rounded once.
function FD(order::Integer, bc = Periodic())
    (order >= 2 && iseven(order)) ||
        throw(ArgumentError("FD order must be even and >= 2, got $order"))
    bc isa Union{Periodic,Mirror} ||
        throw(ArgumentError("FD boundary must be Periodic() or Mirror(), got $bc"))
    M = order ÷ 2
    f = factorial ∘ big
    coeffs = ntuple(k -> Float64((-1)^(k + 1) * f(M)^2 // (k * f(M - k) * f(M + k))), M)
    return FD{M,typeof(bc)}(coeffs, bc)
end

Base.show(io::IO, op::FD{M}) where {M} = print(io, "FD(", 2M, ", ", op.bc, ")")

# Index along the axis of node `j` (possibly outside 1:n) under the boundary condition.
@inline _fold(::Periodic, j, n) = mod1(j, n)
@inline _fold(::Mirror, j, n) = j < 1 ? 2 - j : (j > n ? 2n - j : j)
@inline _fd_at(f, I, bc, dir, j, n) = @inbounds f[Base.setindex(I, _fold(bc, j, n), dir)]

# Sum ordered as Σ_{k=1:M} c_k (f_{i+k} - f_{i-k}), then / dx: bit-identical to bsl6d's
# 9-point stencil written out by hand.
@kernel function fd_derivative_kernel!(out, @Const(f), coeffs, bc, dir, dx, negate)
    I = @index(Global, Cartesian)
    n = size(f, dir)
    i = I[dir]
    acc = coeffs[1] * (_fd_at(f, I, bc, dir, i + 1, n) - _fd_at(f, I, bc, dir, i - 1, n))
    for k = 2:length(coeffs)
        acc += coeffs[k] * (_fd_at(f, I, bc, dir, i + k, n) - _fd_at(f, I, bc, dir, i - k, n))
    end
    d = acc / dx
    @inbounds out[I] = negate ? -d : d
end

_field_data(x::ScalarField) = x.data
_field_data(x::AbstractArray) = x

function _check_axis(data::AbstractArray, grid::Grid, dir::Int)
    1 <= ndims(data) <= 3 ||
        throw(ArgumentError("differentiate supports 1D, 2D, and 3D fields"))
    1 <= dir <= ndims(data) ||
        throw(ArgumentError("direction $dir is outside the field dimensions"))
    dir <= spatial_ndims(grid) ||
        throw(ArgumentError("direction $dir is outside the spatial grid dimensions"))
    return nothing
end

"""
    differentiate!(out, field, grid, dir, op = Spectral(); negate = false)

Writes `∂field/∂x_dir` (its negative with `negate = true`, e.g. for `E = -∇φ`) into `out`,
with the derivative operator `op` (`Spectral()` or `FD(order, bc)`). `field` and `out` are
`ScalarField`s or arrays of the same size; for `FD` they must not alias.
"""
function differentiate!(out, field, grid::Grid, dir::Int, op::DerivativeOperator = Spectral(); negate::Bool = false)
    o, f = _field_data(out), _field_data(field)
    size(o) == size(f) || throw(DimensionMismatch("out has size $(size(o)), field has size $(size(f))"))
    _check_axis(f, grid, dir)
    _differentiate_op!(o, f, grid, dir, op, negate)
    return out
end

_differentiate_op!(out, f, grid, dir, ::Spectral, negate) =
    _differentiate_impl!(out, f, _get_spectral_workspace(f, grid), dir, negate)

function _differentiate_op!(out, f, grid, dir, op::FD{M}, negate) where {M}
    n = size(f, dir)
    n > M || throw(ArgumentError("$(op) needs more than $M nodes along axis $dir, got $n"))
    out === f && throw(ArgumentError("differentiate! with FD cannot work in place"))
    T = eltype(out)
    be = KernelAbstractions.get_backend(out)
    fd_derivative_kernel!(be)(
        out, f, map(T, op.coeffs), op.bc, dir, T(grid.delta[dir]), negate; ndrange = size(out))
    KernelAbstractions.synchronize(be)
    return out
end

"""
    differentiate(field, grid, dir, op = Spectral())

Allocating form of [`differentiate!`](@ref); returns a `ScalarField`.
"""
differentiate(field::ScalarField, grid::Grid, dir::Int, op::DerivativeOperator = Spectral()) =
    differentiate!(ScalarField(similar(field.data)), field, grid, dir, op)

_spectral_ops(grid::Grid) = ntuple(_ -> Spectral(), spatial_ndims(grid))

"""
    grad!(out::VectorField, field, grid, ops = (Spectral(), …); negate = false)

Gradient with one derivative operator per spatial axis, e.g. `(FD(8, Mirror()), Spectral())`.
`negate = true` gives `-∇field`. `out` may have more components than `ops` (e.g. 3 for a 2D grid);
the extra ones are left untouched.
"""
function grad!(out::VectorField, field, grid::Grid, ops::Tuple = _spectral_ops(grid); negate::Bool = false)
    length(ops) == spatial_ndims(grid) || throw(ArgumentError(
        "need one derivative operator per spatial axis ($(spatial_ndims(grid))), got $(length(ops))"))
    length(out) >= length(ops) ||
        throw(ArgumentError("out has $(length(out)) components, need $(length(ops))"))
    for d in eachindex(ops)
        differentiate!(out[d], field, grid, d, ops[d]; negate)
    end
    return out
end

"""
    grad(field::ScalarField, grid, ops = (Spectral(), …))

Allocating form of [`grad!`](@ref).
"""
function grad(field::ScalarField, grid::Grid, ops::Tuple = _spectral_ops(grid))
    spatial_ndims(grid) >= 1 || throw(ArgumentError("grad requires at least one spatial dimension"))
    out = VectorField([ScalarField(similar(field.data)) for _ in ops])
    return grad!(out, field, grid, ops)
end
