using PlasmaCore: FD, Spectral, Periodic, Mirror, differentiate, differentiate!, grad, grad!

# Straightforward reference: fold the index by the boundary condition, then apply the stencil.
function fd_reference(f::AbstractArray, dir::Int, op::FD{M}, dx) where {M}
    n = size(f, dir)
    fold(j) = op.bc isa Periodic ? mod1(j, n) : (j < 1 ? 2 - j : (j > n ? 2n - j : j))
    out = similar(f)
    for I in CartesianIndices(f)
        val(j) = f[Base.setindex(I, fold(j), dir)]
        out[I] = sum(op.coeffs[k] * (val(I[dir] + k) - val(I[dir] - k)) for k = 1:M) / dx
    end
    return out
end

# bsl6d's 9-point derivative with its even mirror halo in x1, written out by hand.
function ddx1_9!(out, phi, dx)
    P = cat(phi[5:-1:2, :, :], phi, phi[end-1:-1:end-4, :, :]; dims = 1)
    n = size(phi, 1)
    p(k) = view(P, 5+k:4+n+k, :, :)
    p1, m1, p2, m2, p3, m3, p4, m4 = p(1), p(-1), p(2), p(-2), p(3), p(-3), p(4), p(-4)
    @. out = (4/5 * (p1 - m1) - 1/5 * (p2 - m2) + 4/105 * (p3 - m3) - 1/280 * (p4 - m4)) / dx
end

@testset "FD operators: coefficients" begin
    @test FD(8, Mirror()).coeffs === (4/5, -1/5, 4/105, -1/280)
    @test FD(2).coeffs === (1/2,)
    @test FD(4).coeffs === (2/3, -1/12)
    @test FD(2).bc === Periodic()
    @test sprint(show, FD(8, Mirror())) == "FD(8, Mirror())"
    @test_throws ArgumentError FD(3)
    @test_throws ArgumentError FD(0)
    @test_throws ArgumentError FD(4, :periodic)
end

@testset "FD operators: polynomial exactness (interior)" begin
    n = 24
    grid = Grid([0.0], [1.0], [n], 1)
    x = collect(grid.xaxes[1])
    for order in (2, 4, 6, 8, 10)
        M = order ÷ 2
        op = FD(order)
        interior = (M+1):(n-M)                  # untouched by the boundary condition
        out = zeros(n)
        for p = 0:order                         # exact up to degree `order`
            differentiate!(out, x .^ p, grid, 1, op)
            exact = p == 0 ? zeros(n) : p .* x .^ (p - 1)
            @test isapprox(out[interior], exact[interior]; atol = 1e-9, rtol = 1e-9)
        end
        order <= 6 || continue                  # and not beyond (for wider stencils the error is below tolerance)
        differentiate!(out, x .^ (order + 1), grid, 1, op)
        @test !isapprox(out[interior], (order + 1) .* x[interior] .^ order; atol = 1e-9, rtol = 1e-9)
    end
end

@testset "FD operators: boundary conditions against a reference" begin
    sz = (11, 7, 6)
    grid = Grid([0.0, 0.0, 0.0], [1.1, 2.0, 3.0], collect(sz), 3)
    f = randn(sz...)
    out = similar(f)
    for bc in (Periodic(), Mirror()), order in (2, 4, 8), dir = 1:3
        op = FD(order, bc)
        ref = fd_reference(f, dir, op, grid.delta[dir])
        differentiate!(out, f, grid, dir, op)
        @test out ≈ ref rtol = 1e-14 atol = 1e-12
        differentiate!(out, ScalarField(f), grid, dir, op; negate = true)
        @test out ≈ -ref rtol = 1e-14 atol = 1e-12
    end
    @test_throws ArgumentError differentiate!(out, f, grid, 4, FD(2))
    @test_throws ArgumentError differentiate!(f, f, grid, 1, FD(2))
    small = Grid([0.0], [1.0], [4], 1)
    @test_throws ArgumentError differentiate!(zeros(4), zeros(4), small, 1, FD(8))
end

@testset "FD operators: convergence" begin
    # Functions whose extension by the boundary condition is smooth converge at the order.
    function maxerr(n, op, f, df)
        grid = Grid([0.0], [1.0], [n], 1)
        x = collect(grid.xaxes[1])
        L = x[end] - x[1]                       # mirror walls on the end nodes
        out = zeros(n)
        differentiate!(out, f.(x, L), grid, 1, op)
        return maximum(abs.(out .- df.(x, L)))
    end
    cases = (
        (Periodic(), (x, L) -> sin(2pi * x), (x, L) -> 2pi * cos(2pi * x)),
        (Mirror(), (x, L) -> cos(pi * x / L), (x, L) -> -pi / L * sin(pi * x / L)),
    )
    for (bc, f, df) in cases, order in (2, 4, 8)
        @test log2(maxerr(32, FD(order, bc), f, df) / maxerr(64, FD(order, bc), f, df)) > order - 0.3
    end
end

@testset "FD operators: bit-identical to bsl6d's 9-point stencil" begin
    sz = (16, 6, 5)
    grid = Grid([0.0, 0.0, 0.0], [3.7, 2pi, 2pi], collect(sz), 3)
    phi = randn(sz...)
    ref = similar(phi)
    ddx1_9!(ref, phi, grid.delta[1])
    out = similar(phi)
    differentiate!(out, phi, grid, 1, FD(8, Mirror()))
    @test out == ref
    differentiate!(out, phi, grid, 1, FD(8, Mirror()); negate = true)
    @test out == -ref
end

@testset "FD operators: mixed gradient" begin
    N = (32, 16, 8)
    grid = Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], collect(N), 3)
    xs, ys, zs = collect.(grid.xaxes)
    phi = ScalarField([sin(x) * cos(2y) * sin(z) for x in xs, y in ys, z in zs])
    ops = (FD(8), Spectral(), Spectral())
    g = grad(phi, grid, ops)
    gs = grad(phi, grid)
    @test g[1].data ≈ gs[1].data atol = 1e-5           # FD error on the x1 axis
    @test g[2].data == gs[2].data                      # spectral axes identical to grad(field, grid)
    @test g[3].data == gs[3].data
    @test differentiate(phi, grid, 1, FD(8)).data == g[1].data
    E = PlasmaCore.zero_vectorfield3(grid)
    grad!(E, phi, grid, ops; negate = true)
    @test E[1].data == -g[1].data
    @test E[2].data == -gs[2].data
    @test_throws ArgumentError grad(phi, grid, (FD(2), Spectral()))
end

@testset "FD operators: Float32" begin
    grid = Grid([0.0], [1.0], [20], 1)
    f = Float32.(sin.(2pi .* collect(grid.xaxes[1])))
    out = similar(f)
    differentiate!(out, f, grid, 1, FD(8))
    @test eltype(out) == Float32
    @test out ≈ Float32.(2pi .* cos.(2pi .* collect(grid.xaxes[1]))) rtol = 1e-3
end

@testset "Vector calculus: div and curl with per-axis operators" begin
    N = (32, 16, 8)
    grid = Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], collect(N), 3)
    xs, ys, zs = collect.(grid.xaxes)
    F = VectorField([
        ScalarField([sin(x) * cos(y) for x in xs, y in ys, z in zs]),
        ScalarField([cos(2x) * sin(z) for x in xs, y in ys, z in zs]),
        ScalarField([sin(y + z) for x in xs, y in ys, z in zs]),
    ])
    spectral = (Spectral(), Spectral(), Spectral())
    mixed = (FD(8), FD(8), Spectral())               # FD on the finer axes (kh <= 0.4)

    @test PlasmaCore.div(F, grid, spectral).data == PlasmaCore.div(F, grid).data
    @test PlasmaCore.div(F, grid, mixed).data ≈ PlasmaCore.div(F, grid).data atol = 1e-4
    c, cm = PlasmaCore.curl(F, grid), PlasmaCore.curl(F, grid, mixed)
    @test all(d -> isapprox(cm[d].data, c[d].data; atol = 1e-4), 1:3)

    # the workspace forms used by solvers give the same result as the default operators
    ws = PlasmaCore._get_spectral_workspace(F[1].data, grid)
    out, temp = similar(F[1].data), similar(F[1].data)
    PlasmaCore._apply_div!(out, F, temp, ws, grid)
    @test out == PlasmaCore.div(F, grid).data
    outs = ntuple(_ -> similar(F[1].data), 3)
    PlasmaCore._apply_curl!(outs, F, temp, ws, grid)
    @test all(d -> outs[d] == c[d].data, 1:3)

    # vector components have a parity under a mirror; not implemented, so rejected
    mirror = (FD(8, Mirror()), Spectral(), Spectral())
    @test_throws ArgumentError PlasmaCore.div(F, grid, mirror)
    @test_throws ArgumentError PlasmaCore.curl(F, grid, mirror)
    @test_throws ArgumentError PlasmaCore.div(F, grid, (Spectral(), Spectral()))
end
