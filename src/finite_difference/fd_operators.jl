# Central finite-difference first derivative along one axis (`FD`), a `DerivativeOperator`
# like `Spectral()`; applied through `differentiate!` and the operators in vector_calculus.jl.

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
