"""
    fourier_filter!(field, grid; remove)

Zero Fourier modes of `field` (a `ScalarField` or an array) along periodic axes.
`remove` is a tuple of `axis => bins` pairs; `bins` are 0-based DFT bin indices
(`0` is the mean, `N/2` the Nyquist bin), e.g. `remove = (1 => 31:33, 2 => 31:33)` on
`N = 64` removes the three bins around the Nyquist bin of axes 1 and 2, as bsl6d does.

The transform is a periodic DFT of the node values: apply it only to periodic axes
(a mirror axis would be filtered as if it were periodic). Bins should be symmetric
about `N/2` (bin `m` pairs with `N - m`), otherwise the imaginary part is discarded.
"""
function fourier_filter!(data::AbstractArray, grid::Grid, remove::Tuple)
    isempty(remove) && return data
    ndims_x = spatial_ndims(grid)
    for (axis, bins) in remove
        1 <= axis <= ndims_x ||
            throw(ArgumentError("axis $axis is outside the spatial grid dimensions"))
        n = size(data, axis)
        all(b -> 0 <= b < n, bins) ||
            throw(ArgumentError("bins $bins outside 0:$(n - 1) for axis $axis"))
    end
    ws = spectral_workspace(data, grid)
    T = real(eltype(data))
    @. ws.fft_buf = data
    for (axis, _) in remove
        ws.fwd_plans[axis] * ws.fft_buf
    end
    for (axis, bins) in remove
        keep = fill!(similar(data, T, size(data, axis)), one(T))
        mask = ones(T, size(data, axis))
        mask[bins .+ 1] .= zero(T)
        copyto!(keep, mask)
        ws.fft_buf .*= reshape(keep, ntuple(d -> d == axis ? size(data, axis) : 1, ndims(data)))
    end
    for (axis, _) in remove
        ws.inv_plans[axis] * ws.fft_buf
    end
    @. data = real(ws.fft_buf)
    return data
end

function fourier_filter!(field::ScalarField, grid::Grid; remove)
    fourier_filter!(field.data, grid, Tuple(remove))
    return field
end
