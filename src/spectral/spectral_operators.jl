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

# Spectral divergence and curl on a given workspace (used by solver workspaces); the generic,
# per-axis-operator forms `div` and `curl` are in vector_calculus.jl.
_spectral_derivative(ws::SpectralWorkspace) = (o, f, d, neg) -> _differentiate_impl!(o, f, ws, d, neg)

_apply_div!(out::AbstractArray, field::VectorField, temp::AbstractArray, ws::SpectralWorkspace, grid::Grid) =
    _div!(_spectral_derivative(ws), out, field, temp, spatial_ndims(grid))

_apply_curl!(out::NTuple{3}, field::VectorField, temp::AbstractArray, ws::SpectralWorkspace, grid::Grid) =
    _curl!(_spectral_derivative(ws), out, field, temp, spatial_ndims(grid))

# --- Derivative operators ------------------------------------------------------------

abstract type DerivativeOperator end

"""
    Spectral()

FFT derivative along one axis (periodic). The default operator of `differentiate`, `grad`,
`div` and `curl`.
"""
struct Spectral <: DerivativeOperator end

_differentiate_op!(out, f, grid, dir, ::Spectral, negate) =
    _differentiate_impl!(out, f, _get_spectral_workspace(f, grid), dir, negate)

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
