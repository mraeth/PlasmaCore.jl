struct DifferentiateContext{KT,ST}
    k::KT
    sizes::ST
    dir::Int
end

Adapt.@adapt_structure DifferentiateContext

@inline function (ctx::DifferentiateContext)(index::Int)
    return compute_derivative_multiplier(ctx, index)
end

@inline function compute_derivative_multiplier(ctx::DifferentiateContext, index::Int)
    idxs = index_1d_to_nd(index, ctx.sizes)
    return im * ctx.k[idxs[ctx.dir]]
end

function spectral_wavenumbers(field_data::AbstractArray, grid::Grid, dir::Int)
    T = real(eltype(field_data))
    1 <= dir <= length(grid.xaxes) ||
        throw(ArgumentError("direction $dir is outside the spatial grid dimensions"))
    n = size(field_data, dir)
    axis_length = n * grid.delta[dir]
    kvec = T.((2pi / axis_length) .* fftfreq(n, n))
    if iseven(n)
        kvec[n÷2+1] = zero(T)
    end
    k = similar(field_data, T, n)
    copyto!(k, kvec)
    return k
end

spatial_ndims(grid::Grid) = length(grid.xaxes)
ncomponents(field::VectorField) = length(field)

struct SpectralWorkspace{FB,FP,IP,KT,BK}
    fft_buf::FB
    fwd_plans::FP
    inv_plans::IP
    k::KT
    backend::BK
end

function SpectralWorkspace(arr::AbstractArray, grid::Grid)
    ndims_x = spatial_ndims(grid)
    fft_buf = similar(arr, Complex{eltype(arr)})
    fwd_plans = ntuple(d -> plan_fft!(fft_buf, d), ndims_x)
    inv_plans = ntuple(d -> plan_ifft!(fft_buf, d), ndims_x)
    k = ntuple(d -> spectral_wavenumbers(arr, grid, d), ndims_x)
    return SpectralWorkspace(fft_buf, fwd_plans, inv_plans, k, backend())
end

# Unified cache for all solver workspaces.
# Key: (solver_tag::Symbol, size, eltype, ndims_x, ncomp, bdir, delta_tuple)
const _solver_workspace_cache = Dict{Any,Any}()
const _solver_workspace_cache_lock = ReentrantLock()

# Shared cache for SpectralWorkspace objects keyed by grid shape.
# Key: (size, eltype, ndims_x, delta_tuple)
const _spectral_ws_cache = Dict{Any,Any}()
const _spectral_ws_cache_lock = ReentrantLock()

function _spectral_ws_key(arr::AbstractArray, grid::Grid)
    ndims_x = spatial_ndims(grid)
    return (size(arr), eltype(arr), ndims_x, Tuple(grid.delta[1:ndims_x]))
end

function _get_spectral_workspace(arr::AbstractArray, grid::Grid)
    key = _spectral_ws_key(arr, grid)
    lock(_spectral_ws_cache_lock) do
        get!(() -> SpectralWorkspace(arr, grid), _spectral_ws_cache, key)
    end
end

# Allocation-free variant: writes into pre-allocated `out` using cached buffers and plans.
# Set negate=true to compute -d(field)/dx_dir (for E = -∇φ).
function _differentiate_impl!(
    out::AbstractArray,
    field_data::AbstractArray,
    fhat_buf::AbstractArray,
    fwd_plan,
    inv_plan,
    k::AbstractArray,
    dir::Int,
    exec,
    negate::Bool = false,
)
    kernel! = spectral_multiply_kernel!(exec)
    @. fhat_buf = field_data
    fwd_plan * fhat_buf
    ctx = DifferentiateContext(k, size(fhat_buf), dir)
    kernel!(fhat_buf, ctx; ndrange = length(fhat_buf))
    KernelAbstractions.synchronize(exec)
    inv_plan * fhat_buf
    if negate
        @. out = -real(fhat_buf)
    else
        @. out = real(fhat_buf)
    end
    return out
end

function _differentiate_impl!(
    out::AbstractArray,
    field_data::AbstractArray,
    ws::SpectralWorkspace,
    dir::Int,
    negate::Bool = false,
)
    _differentiate_impl!(
        out,
        field_data,
        ws.fft_buf,
        ws.fwd_plans[dir],
        ws.inv_plans[dir],
        ws.k[dir],
        dir,
        ws.backend,
        negate,
    )
end

# Copy src → fft_buf, apply all fwd_plans in-place, copy result → dst.
function _fwd_fft_to!(
    sw::SpectralWorkspace,
    src::AbstractArray,
    dst::AbstractArray,
    ndims_x::Int,
)
    @. sw.fft_buf = src
    for d = 1:ndims_x
        sw.fwd_plans[d] * sw.fft_buf
    end
    dst .= sw.fft_buf
    return dst
end

# Copy src → fft_buf, apply all inv_plans in-place, write real part → dst.
function _inv_fft_from!(
    sw::SpectralWorkspace,
    src::AbstractArray,
    dst::AbstractArray,
    ndims_x::Int,
)
    @. sw.fft_buf = src
    for d = 1:ndims_x
        sw.inv_plans[d] * sw.fft_buf
    end
    @. dst = real(sw.fft_buf)
    return dst
end

# In-place real-space divergence using SpectralWorkspace.
# Writes sum_d d/dx_d field[d] into `out`. Uses `temp` as a scratch Float buffer.
function _apply_div!(
    out::AbstractArray,
    field::VectorField,
    temp::AbstractArray,
    ws::SpectralWorkspace,
    grid::Grid,
)
    ndims_x = spatial_ndims(grid)
    _differentiate_impl!(out, field[1], ws, 1, false)
    for d = 2:ndims_x
        _differentiate_impl!(temp, field[d], ws, d, false)
        out .+= temp
    end
    return out
end

# In-place real-space curl using SpectralWorkspace.
# `out` is a NTuple{3} of Float64 field-sized arrays; `temp` is a single scratch Float array.
function _apply_curl!(
    out::NTuple{3},
    field::VectorField,
    temp::AbstractArray,
    ws::SpectralWorkspace,
    grid::Grid,
)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)

    if ndims_x == 1
        if ncomp == 2
            _differentiate_impl!(out[1], field[2], ws, 1, true)    # -d/dx f[2]
            _differentiate_impl!(out[2], field[1], ws, 1, false)   # +d/dx f[1]
            fill!(out[3], 0)
        elseif ncomp == 3
            fill!(out[1], 0)
            _differentiate_impl!(out[2], field[3], ws, 1, true)    # -d/dx f[3]
            _differentiate_impl!(out[3], field[2], ws, 1, false)   # +d/dx f[2]
        else
            throw(ArgumentError("1D curl requires 2 or 3 field components"))
        end
    elseif ndims_x == 2 && ncomp == 2
        # 2-component 2D curl: scalar result stored in out[1]; out[2], out[3] zeroed
        _differentiate_impl!(out[1], field[2], ws, 1, false)   # d/dx f[2]
        _differentiate_impl!(temp, field[1], ws, 2, false)     # d/dy f[1]
        out[1] .-= temp
        fill!(out[2], 0)
        fill!(out[3], 0)
    elseif ndims_x == 2
        ncomp == 3 || throw(ArgumentError("2D curl requires 3 field components"))
        _differentiate_impl!(out[1], field[3], ws, 2, false)   # d/dy f[3]
        _differentiate_impl!(out[2], field[3], ws, 1, true)    # -d/dx f[3]
        _differentiate_impl!(out[3], field[2], ws, 1, false)   # d/dx f[2]
        _differentiate_impl!(temp, field[1], ws, 2, false)     # d/dy f[1]
        out[3] .-= temp
    elseif ndims_x == 3
        ncomp == 3 || throw(ArgumentError("3D curl requires 3 field components"))
        _differentiate_impl!(out[1], field[3], ws, 2, false)   # d/dy f[3]
        _differentiate_impl!(temp, field[2], ws, 3, false)     # d/dz f[2]
        out[1] .-= temp
        _differentiate_impl!(out[2], field[1], ws, 3, false)   # d/dz f[1]
        _differentiate_impl!(temp, field[3], ws, 1, false)     # d/dx f[3]
        out[2] .-= temp
        _differentiate_impl!(out[3], field[2], ws, 1, false)   # d/dx f[2]
        _differentiate_impl!(temp, field[1], ws, 2, false)     # d/dy f[1]
        out[3] .-= temp
    else
        throw(ArgumentError("_apply_curl! supports 1D, 2D, and 3D grids"))
    end
    return out
end

function differentiate(field::ScalarField{T,N}, grid::Grid, dir::Int) where {T,N}
    1 <= N <= 3 ||
        throw(ArgumentError("differentiate currently supports 1D, 2D, and 3D ScalarFields"))
    1 <= dir <= N || throw(ArgumentError("direction $dir is outside the field dimensions"))
    dir <= length(grid.xaxes) ||
        throw(ArgumentError("direction $dir is outside the spatial grid dimensions"))
    ws = _get_spectral_workspace(field.data, grid)
    out = similar(field.data, eltype(field.data))
    _differentiate_impl!(out, field, ws, dir)
    return ScalarField(out)
end

function grad(field::ScalarField{T,N}, grid::Grid) where {T,N}
    ndirs = spatial_ndims(grid)
    ndirs >= 1 || throw(ArgumentError("grad requires at least one spatial dimension"))
    ws = _get_spectral_workspace(field.data, grid)
    return VectorField([
        begin
            out = similar(field.data, eltype(field.data))
            _differentiate_impl!(out, field, ws, d)
            ScalarField(out)
        end for d = 1:ndirs
    ])
end

function div(field::VectorField, grid::Grid)
    ndirs = spatial_ndims(grid)
    ncomp = ncomponents(field)
    ndirs >= 1 || throw(ArgumentError("div requires at least one spatial dimension"))
    ncomp >= ndirs || throw(
        ArgumentError("div on a $ndirs-D grid requires at least $ndirs vector components"),
    )
    ws = _get_spectral_workspace(field[1].data, grid)
    out = similar(field[1].data, eltype(field[1].data))
    temp = similar(out)
    _apply_div!(out, field, temp, ws, grid)
    return ScalarField(out)
end

function div(field::MatrixField{SF,NF}, grid::Grid) where {SF,NF}
    ndirs = spatial_ndims(grid)
    NS = isqrt(NF)
    ndirs >= 1 || throw(ArgumentError("div requires at least one spatial dimension"))
    NS >= ndirs || throw(
        ArgumentError("div on a $ndirs-D grid requires at least $ndirs matrix columns"),
    )
    ws = _get_spectral_workspace(field[1, 1].data, grid)
    out = similar(field[1, 1].data, eltype(field[1, 1].data))
    temp = similar(out)
    rows = Vector{ScalarField}(undef, NS)
    for i = 1:NS
        row = VectorField([field[i, d] for d = 1:ndirs])
        _apply_div!(out, row, temp, ws, grid)
        rows[i] = ScalarField(copy(out))
    end
    return VectorField(rows)
end

function curl(field::VectorField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)
    ws = _get_spectral_workspace(field[1].data, grid)
    temp = similar(field[1].data, eltype(field[1].data))
    out = ntuple(_ -> similar(field[1].data, eltype(field[1].data)), 3)
    _apply_curl!(out, field, temp, ws, grid)

    if ndims_x == 2 && ncomp == 2
        return ScalarField(out[1])
    elseif ndims_x == 1 && ncomp == 2
        return VectorField([ScalarField(out[1]), ScalarField(out[2])])
    else
        return VectorField([ScalarField(out[1]), ScalarField(out[2]), ScalarField(out[3])])
    end
end

spatial_fft_dims(grid::Grid) = Tuple(1:spatial_ndims(grid))

function fft_spatial(data::AbstractArray, grid::Grid)
    return fft(data, spatial_fft_dims(grid))
end

function ifft_spatial(data::AbstractArray, grid::Grid)
    return ifft(data, spatial_fft_dims(grid))
end

function spectral_wavenumber_views(field::ScalarField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    ndims_data = ndims(field.data)
    return ntuple(ndims_x) do dir
        reshape(
            spectral_wavenumbers(field.data, grid, dir),
            ntuple(d -> d == dir ? size(field.data, dir) : 1, ndims_data),
        )
    end
end

function spectral_wavenumber_squared(field::ScalarField, grid::Grid)
    kviews = spectral_wavenumber_views(field, grid)
    k2 = fill!(similar(field.data, eltype(field.data)), zero(eltype(field.data)))
    for kview in kviews
        k2 .+= kview .^ 2
    end
    return k2
end

function _spectral_curl_hat!(out, fieldhat, kviews, ndims_x::Int)
    ncomp = length(fieldhat)
    if ndims_x == 1
        kx = kviews[1]
        if ncomp == 2
            @. out[1] = -(im * kx * fieldhat[2])
            @. out[2] = im * kx * fieldhat[1]
            return out
        elseif ncomp == 3
            fill!(out[1], 0)
            @. out[2] = -(im * kx * fieldhat[3])
            @. out[3] = im * kx * fieldhat[2]
            return out
        end
    elseif ndims_x == 2
        kx, ky = kviews[1], kviews[2]
        if ncomp == 3
            @. out[1] = im * ky * fieldhat[3]
            @. out[2] = -(im * kx * fieldhat[3])
            @. out[3] = im * kx * fieldhat[2] - im * ky * fieldhat[1]
            return out
        end
    elseif ndims_x == 3
        kx, ky, kz = kviews[1], kviews[2], kviews[3]
        if ncomp == 3
            @. out[1] = im * ky * fieldhat[3] - im * kz * fieldhat[2]
            @. out[2] = im * kz * fieldhat[1] - im * kx * fieldhat[3]
            @. out[3] = im * kx * fieldhat[2] - im * ky * fieldhat[1]
            return out
        end
    end
    throw(
        ArgumentError(
            "unsupported spectral curl layout for $ndims_x spatial dimensions and $ncomp components",
        ),
    )
end
