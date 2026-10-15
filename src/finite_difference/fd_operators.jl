# First-derivative operators along one spatial axis: central finite differences (`FD`)
# and the FFT derivative (`Spectral`). `ddx!` applies one of them along one axis,
# `grad!`/`grad(field, grid, ops)` take one operator per axis.

abstract type DerivativeOperator end

"""
    Spectral()

FFT derivative along one axis (periodic). Same as [`differentiate`](@ref).
"""
struct Spectral <: DerivativeOperator end

"""
    FD(width, bc = :periodic)

Central finite-difference first derivative with a stencil of odd `width` (3, 5, 7, 9, ...).
The weights are Fornberg weights, computed in exact rational arithmetic and rounded once
to `Float64`; for `width = 9` they are bsl6d's `(4/5, -1/5, 4/105, -1/280)`.

Boundary conditions (values used for the points outside `1:n` along the axis):
- `:periodic`    — `f[n + k] = f[k]`, `f[1 - k] = f[n + 1 - k]`.
- `:mirror_even` — even reflection about the end nodes: `f[1 - k] = f[1 + k]`,
  `f[n + k] = f[n - k]` (bsl6d's mirror halo).
- `:mirror_odd`  — odd reflection about the end nodes: `f[1 - k] = -f[1 + k]`,
  `f[n + k] = -f[n - k]`.
- `:dirichlet`   — homogeneous: the field is zero outside `1:n`.

The axis needs at least `width ÷ 2 + 1` points.
"""
struct FD{M,BC} <: DerivativeOperator
    coeffs::NTuple{M,Float64}   # c_k, k = 1:M; d/dx f_i = sum_k c_k (f_{i+k} - f_{i-k}) / dx
end

const FD_BOUNDARY_CONDITIONS = (:periodic, :mirror_even, :mirror_odd, :dirichlet)

function FD(width::Integer, bc::Symbol = :periodic)
    (width >= 3 && isodd(width)) ||
        throw(ArgumentError("FD stencil width must be odd and >= 3, got $width"))
    bc in FD_BOUNDARY_CONDITIONS ||
        throw(ArgumentError("unknown FD boundary condition :$bc; use one of $FD_BOUNDARY_CONDITIONS"))
    M = width ÷ 2
    w = fornberg_weights(-M:M, 1)
    return FD{M,bc}(ntuple(k -> Float64(w[M+1+k]), M))
end

stencil_width(::FD{M}) where {M} = 2M + 1
boundary_condition(::FD{M,BC}) where {M,BC} = BC

Base.show(io::IO, op::FD{M,BC}) where {M,BC} = print(io, "FD(", 2M + 1, ", :", BC, ")")

"""
    fornberg_weights(nodes, m)

Weights `w` with `sum(w .* f.(nodes)) ≈ f^(m)(0)` (Fornberg 1988), exact in `Rational{BigInt}`.
`nodes` are in units of the grid spacing.
"""
function fornberg_weights(nodes, m::Integer)
    x = Rational{BigInt}.(collect(nodes))
    n = length(x) - 1
    m <= n || throw(ArgumentError("derivative order $m needs at least $(m + 1) nodes"))
    C = zeros(Rational{BigInt}, n + 1, m + 1)
    C[1, 1] = 1
    c1 = big(1) // 1
    c4 = x[1]
    for i = 1:n
        mn = min(i, m)
        c2 = big(1) // 1
        c5 = c4
        c4 = x[i+1]
        for j = 0:i-1
            c3 = x[i+1] - x[j+1]
            c2 *= c3
            if j == i - 1
                for k = mn:-1:1
                    C[i+1, k+1] = c1 * (k * C[i, k] - c5 * C[i, k+1]) / c2
                end
                C[i+1, 1] = -c1 * c5 * C[i, 1] / c2
            end
            for k = mn:-1:1
                C[j+1, k+1] = (c4 * C[j+1, k+1] - k * C[j+1, k]) / c3
            end
            C[j+1, 1] = c4 * C[j+1, 1] / c3
        end
        c1 = c2
    end
    return C[:, m+1]
end

# Value of f at index j along `dir` (j may lie outside 1:n), with the halo of the boundary condition.
@inline function _fd_value(f, I::CartesianIndex, dir, j, n, ::Val{:periodic})
    return @inbounds f[Base.setindex(I, mod1(j, n), dir)]
end

@inline function _fd_value(f, I::CartesianIndex, dir, j, n, ::Val{:mirror_even})
    jj = j < 1 ? 2 - j : (j > n ? 2n - j : j)
    return @inbounds f[Base.setindex(I, jj, dir)]
end

@inline function _fd_value(f, I::CartesianIndex, dir, j, n, ::Val{:mirror_odd})
    if j < 1
        return @inbounds -f[Base.setindex(I, 2 - j, dir)]
    elseif j > n
        return @inbounds -f[Base.setindex(I, 2n - j, dir)]
    end
    return @inbounds f[Base.setindex(I, j, dir)]
end

@inline function _fd_value(f, I::CartesianIndex, dir, j, n, ::Val{:dirichlet})
    (1 <= j <= n) || return zero(eltype(f))
    return @inbounds f[Base.setindex(I, j, dir)]
end

# Sum ordered as sum_{k=1:M} c_k (f_{i+k} - f_{i-k}), then / dx: matches bsl6d's stencil
# written out by hand (bit-identical for width 9).
@kernel function fd_derivative_kernel!(out, @Const(f), coeffs, dir, n, dx, bcval, negate)
    I = @index(Global, Cartesian)
    i = I[dir]
    acc = coeffs[1] * (_fd_value(f, I, dir, i + 1, n, bcval) - _fd_value(f, I, dir, i - 1, n, bcval))
    for k = 2:length(coeffs)
        acc += coeffs[k] * (_fd_value(f, I, dir, i + k, n, bcval) - _fd_value(f, I, dir, i - k, n, bcval))
    end
    d = acc / dx
    @inbounds out[I] = negate ? -d : d
end

_field_data(x::ScalarField) = x.data
_field_data(x::AbstractArray) = x

function _check_axis(data::AbstractArray, grid::Grid, dir::Int)
    1 <= dir <= ndims(data) ||
        throw(ArgumentError("direction $dir is outside the field dimensions"))
    dir <= spatial_ndims(grid) ||
        throw(ArgumentError("direction $dir is outside the spatial grid dimensions"))
    return nothing
end

"""
    ddx!(out, field, grid, dir; op = Spectral(), negate = false)

Writes `∂field/∂x_dir` (or its negative with `negate = true`, e.g. for `E = -∇φ`) into `out`,
using the derivative operator `op` (`FD(...)` or `Spectral()`). `field` and `out` are
`ScalarField`s or plain arrays of the same size; they must not alias for `FD`.
"""
function ddx!(out, field, grid::Grid, dir::Int; op::DerivativeOperator = Spectral(), negate::Bool = false)
    o, f = _field_data(out), _field_data(field)
    size(o) == size(f) || throw(DimensionMismatch("out has size $(size(o)), field has size $(size(f))"))
    _check_axis(f, grid, dir)
    _ddx_impl!(o, f, grid, dir, op, negate)
    return out
end

function _ddx_impl!(out::AbstractArray, f::AbstractArray, grid::Grid, dir::Int, ::Spectral, negate::Bool)
    ws = _get_spectral_workspace(f, grid)
    _differentiate_impl!(out, f, ws, dir, negate)
    return out
end

function _ddx_impl!(out::AbstractArray, f::AbstractArray, grid::Grid, dir::Int, op::FD{M,BC}, negate::Bool) where {M,BC}
    n = size(f, dir)
    n >= M + 1 || throw(ArgumentError("$(op) needs at least $(M + 1) points along axis $dir, got $n"))
    out === f && throw(ArgumentError("ddx! with an FD operator cannot work in place (out === field)"))
    T = eltype(out)
    coeffs = map(T, op.coeffs)
    dx = T(grid.delta[dir])
    be = KernelAbstractions.get_backend(out)
    fd_derivative_kernel!(be)(out, f, coeffs, dir, n, dx, Val(BC), negate; ndrange = size(out))
    KernelAbstractions.synchronize(be)
    return out
end

function _check_ops(ops::Tuple, grid::Grid)
    length(ops) == spatial_ndims(grid) || throw(
        ArgumentError("need one derivative operator per spatial axis ($(spatial_ndims(grid))), got $(length(ops))"),
    )
    all(op -> op isa DerivativeOperator, ops) ||
        throw(ArgumentError("ops must be derivative operators (FD(...) or Spectral())"))
    return nothing
end

"""
    grad!(out::VectorField, field, grid, ops; negate = false)

Gradient with one derivative operator per spatial axis, e.g.
`ops = (FD(9, :mirror_even), Spectral(), Spectral())`. `negate = true` gives `-∇field`.
"""
function grad!(out::VectorField, field, grid::Grid, ops::Tuple; negate::Bool = false)
    _check_ops(ops, grid)
    length(out) >= length(ops) ||
        throw(ArgumentError("out has $(length(out)) components, need $(length(ops))"))
    for d in eachindex(ops)
        ddx!(out[d], field, grid, d; op = ops[d], negate)
    end
    return out
end

"""
    grad(field::ScalarField, grid, ops)

Allocating form of [`grad!`](@ref) with one derivative operator per spatial axis.
"""
function grad(field::ScalarField, grid::Grid, ops::Tuple)
    _check_ops(ops, grid)
    out = VectorField([ScalarField(similar(field.data)) for _ in ops])
    return grad!(out, field, grid, ops)
end
