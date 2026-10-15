# Vector calculus with one derivative operator per spatial axis -- `Spectral()` (default) or
# `FD(order, bc)`: differentiate, grad, div, curl. Every term is a single-axis derivative.

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

"""
    differentiate(field, grid, dir, op = Spectral())

Allocating form of [`differentiate!`](@ref); returns a `ScalarField`.
"""
differentiate(field::ScalarField, grid::Grid, dir::Int, op::DerivativeOperator = Spectral()) =
    differentiate!(ScalarField(similar(field.data)), field, grid, dir, op)

_spectral_ops(grid::Grid) = ntuple(_ -> Spectral(), spatial_ndims(grid))

function _check_ops(ops::Tuple, grid::Grid)
    length(ops) == spatial_ndims(grid) || throw(ArgumentError(
        "need one derivative operator per spatial axis ($(spatial_ndims(grid))), got $(length(ops))"))
    return nothing
end

# Single-axis derivative `D!(out, f, dir, negate)` with the operator of each axis.
_operator_derivative(grid::Grid, ops::Tuple) =
    (o, f, d, neg) -> differentiate!(o, f, grid, d, ops[d]; negate = neg)

"""
    grad!(out::VectorField, field, grid, ops = (Spectral(), …); negate = false)

Gradient with one derivative operator per spatial axis, e.g. `(FD(8, Mirror()), Spectral())`.
`negate = true` gives `-∇field`. `out` may have more components than `ops` (e.g. 3 for a 2D grid);
the extra ones are left untouched.
"""
function grad!(out::VectorField, field, grid::Grid, ops::Tuple = _spectral_ops(grid); negate::Bool = false)
    _check_ops(ops, grid)
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

# div and curl differentiate vector components, whose parity under a mirror depends on the
# component (the normal one is odd, the tangential ones even). Until that is implemented they
# accept periodic operators only.
_is_periodic(::Spectral) = true
_is_periodic(op::FD) = op.bc isa Periodic

function _check_periodic(ops::Tuple, name)
    all(_is_periodic, ops) || throw(ArgumentError(
        "$name supports periodic operators only (Spectral() or FD(order, Periodic())); " *
        "Mirror() needs the per-component parity, not implemented yet"))
    return nothing
end

# Σ_d ∂_d field[d] into `out`; `temp` is a scratch array.
function _div!(D!, out, field::VectorField, temp, ndims_x)
    D!(out, field[1], 1, false)
    for d = 2:ndims_x
        D!(temp, field[d], d, false)
        out .+= temp
    end
    return out
end

# Curl into the three arrays `out`; `temp` is a scratch array. For 2 components on a 2D grid the
# scalar curl goes to out[1] and out[2], out[3] are zeroed.
function _curl!(D!, out::NTuple{3}, field::VectorField, temp, ndims_x)
    ncomp = ncomponents(field)

    if ndims_x == 1
        if ncomp == 2
            D!(out[1], field[2], 1, true)    # -d/dx f[2]
            D!(out[2], field[1], 1, false)   # +d/dx f[1]
            fill!(out[3], 0)
        elseif ncomp == 3
            fill!(out[1], 0)
            D!(out[2], field[3], 1, true)    # -d/dx f[3]
            D!(out[3], field[2], 1, false)   # +d/dx f[2]
        else
            throw(ArgumentError("1D curl requires 2 or 3 field components"))
        end
    elseif ndims_x == 2 && ncomp == 2
        # 2-component 2D curl: scalar result stored in out[1]; out[2], out[3] zeroed
        D!(out[1], field[2], 1, false)   # d/dx f[2]
        D!(temp, field[1], 2, false)     # d/dy f[1]
        out[1] .-= temp
        fill!(out[2], 0)
        fill!(out[3], 0)
    elseif ndims_x == 2
        ncomp == 3 || throw(ArgumentError("2D curl requires 3 field components"))
        D!(out[1], field[3], 2, false)   # d/dy f[3]
        D!(out[2], field[3], 1, true)    # -d/dx f[3]
        D!(out[3], field[2], 1, false)   # d/dx f[2]
        D!(temp, field[1], 2, false)     # d/dy f[1]
        out[3] .-= temp
    elseif ndims_x == 3
        ncomp == 3 || throw(ArgumentError("3D curl requires 3 field components"))
        D!(out[1], field[3], 2, false)   # d/dy f[3]
        D!(temp, field[2], 3, false)     # d/dz f[2]
        out[1] .-= temp
        D!(out[2], field[1], 3, false)   # d/dz f[1]
        D!(temp, field[3], 1, false)     # d/dx f[3]
        out[2] .-= temp
        D!(out[3], field[2], 1, false)   # d/dx f[2]
        D!(temp, field[1], 2, false)     # d/dy f[1]
        out[3] .-= temp
    else
        throw(ArgumentError("curl supports 1D, 2D, and 3D grids"))
    end
    return out
end

"""
    div(field, grid, ops = (Spectral(), …))

Divergence of a `VectorField` (a `ScalarField`) or, row by row, of a `MatrixField` (a
`VectorField`), with one periodic derivative operator per spatial axis.
"""
function div(field::VectorField, grid::Grid, ops::Tuple = _spectral_ops(grid))
    ndirs = spatial_ndims(grid)
    ndirs >= 1 || throw(ArgumentError("div requires at least one spatial dimension"))
    ncomponents(field) >= ndirs || throw(
        ArgumentError("div on a $ndirs-D grid requires at least $ndirs vector components"),
    )
    _check_ops(ops, grid)
    _check_periodic(ops, "div")
    out = similar(field[1].data, eltype(field[1].data))
    _div!(_operator_derivative(grid, ops), out, field, similar(out), ndirs)
    return ScalarField(out)
end

function div(field::MatrixField{SF,NF}, grid::Grid, ops::Tuple = _spectral_ops(grid)) where {SF,NF}
    ndirs = spatial_ndims(grid)
    NS = isqrt(NF)
    ndirs >= 1 || throw(ArgumentError("div requires at least one spatial dimension"))
    NS >= ndirs || throw(
        ArgumentError("div on a $ndirs-D grid requires at least $ndirs matrix columns"),
    )
    _check_ops(ops, grid)
    _check_periodic(ops, "div")
    D! = _operator_derivative(grid, ops)
    out = similar(field[1, 1].data, eltype(field[1, 1].data))
    temp = similar(out)
    rows = Vector{ScalarField}(undef, NS)
    for i = 1:NS
        _div!(D!, out, VectorField([field[i, d] for d = 1:ndirs]), temp, ndirs)
        rows[i] = ScalarField(copy(out))
    end
    return VectorField(rows)
end

"""
    curl(field::VectorField, grid, ops = (Spectral(), …))

Curl with one periodic derivative operator per spatial axis: a `ScalarField` for 2 components
on a 2D grid, otherwise a `VectorField`.
"""
function curl(field::VectorField, grid::Grid, ops::Tuple = _spectral_ops(grid))
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)
    _check_ops(ops, grid)
    _check_periodic(ops, "curl")
    temp = similar(field[1].data, eltype(field[1].data))
    out = ntuple(_ -> similar(field[1].data, eltype(field[1].data)), 3)
    _curl!(_operator_derivative(grid, ops), out, field, temp, ndims_x)

    if ndims_x == 2 && ncomp == 2
        return ScalarField(out[1])
    elseif ndims_x == 1 && ncomp == 2
        return VectorField([ScalarField(out[1]), ScalarField(out[2])])
    else
        return VectorField([ScalarField(out[1]), ScalarField(out[2]), ScalarField(out[3])])
    end
end
