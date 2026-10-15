using PlasmaCore: FD, Spectral, ddx!, grad!, fornberg_weights, stencil_width, boundary_condition

# Straightforward reference: pad along `dir` with the halo of `bc`, then apply the stencil.
function fd_reference(f::AbstractArray, dir::Int, op::FD{M,BC}, dx) where {M,BC}
    n = size(f, dir)
    out = similar(f)
    for I in CartesianIndices(f)
        acc = 0.0
        for k = 1:M
            val(j) = if 1 <= j <= n
                f[Base.setindex(I, j, dir)]
            elseif BC === :periodic
                f[Base.setindex(I, mod1(j, n), dir)]
            elseif BC === :mirror_even
                f[Base.setindex(I, j < 1 ? 2 - j : 2n - j, dir)]
            elseif BC === :mirror_odd
                -f[Base.setindex(I, j < 1 ? 2 - j : 2n - j, dir)]
            else
                0.0
            end
            acc += op.coeffs[k] * (val(I[dir] + k) - val(I[dir] - k))
        end
        out[I] = acc / dx
    end
    return out
end

# The 9-point derivative of the benchmark script run_f0.jl (bsl6d stencil, even mirror halo in x1).
function ddx1_9!(out, phi, dx)
    P = cat(phi[5:-1:2, :, :], phi, phi[end-1:-1:end-4, :, :]; dims = 1)
    n = size(phi, 1)
    p(k) = view(P, 5+k:4+n+k, :, :)
    p1, m1, p2, m2, p3, m3, p4, m4 = p(1), p(-1), p(2), p(-2), p(3), p(-3), p(4), p(-4)
    @. out = (4/5 * (p1 - m1) - 1/5 * (p2 - m2) + 4/105 * (p3 - m3) - 1/280 * (p4 - m4)) / dx
end

@testset "FD operators: weights" begin
    @test FD(9, :mirror_even).coeffs === (4/5, -1/5, 4/105, -1/280)
    @test FD(3).coeffs === (1/2,)
    @test FD(5).coeffs === (2/3, -1/12)
    @test stencil_width(FD(7, :dirichlet)) == 7
    @test boundary_condition(FD(7, :dirichlet)) === :dirichlet
    # Fornberg weights reproduce the exact rational values, incl. a one-sided stencil
    @test fornberg_weights(-1:1, 1) == [-1//2, 0, 1//2]
    @test fornberg_weights(0:2, 1) == [-3//2, 2, -1//2]
    @test fornberg_weights(-1:1, 2) == [1, -2, 1]
    @test_throws ArgumentError FD(4)
    @test_throws ArgumentError FD(1)
    @test_throws ArgumentError FD(5, :neumann)
end

@testset "FD operators: polynomial exactness (interior)" begin
    n = 24
    grid = Grid([0.0], [1.0], [n], 1)
    x = collect(grid.xaxes[1])
    for width in (3, 5, 7, 9, 11)
        M = width ÷ 2
        op = FD(width, :dirichlet)
        interior = (M+1):(n-M)
        out = zeros(n)
        for p = 0:(2M)                       # exact up to degree 2M
            ddx!(out, x .^ p, grid, 1; op)
            exact = p == 0 ? zeros(n) : p .* x .^ (p - 1)
            @test isapprox(out[interior], exact[interior]; atol = 1e-9, rtol = 1e-9)
        end
        width <= 7 || continue                  # and not beyond (the error (M!)^2 h^2M is below tolerance for wider stencils)
        ddx!(out, x .^ (2M + 1), grid, 1; op)
        @test !isapprox(out[interior], (2M + 1) .* x[interior] .^ (2M); atol = 1e-9, rtol = 1e-9)
    end
end

@testset "FD operators: all boundary conditions against a reference" begin
    sz = (11, 7, 6)
    grid = Grid([0.0, 0.0, 0.0], [1.1, 2.0, 3.0], collect(sz), 3)
    f = randn(sz...)
    out = similar(f)
    for bc in (:periodic, :mirror_even, :mirror_odd, :dirichlet), width in (3, 5, 9), dir = 1:3
        op = FD(width, bc)
        ref = fd_reference(f, dir, op, grid.delta[dir])
        ddx!(out, f, grid, dir; op)
        @test out ≈ ref rtol = 1e-14 atol = 1e-12
        ddx!(out, ScalarField(f), grid, dir; op, negate = true)
        @test out ≈ -ref rtol = 1e-14 atol = 1e-12
    end
    @test_throws ArgumentError ddx!(out, f, grid, 4; op = FD(3))
    @test_throws ArgumentError ddx!(f, f, grid, 1; op = FD(3))
    small = Grid([0.0], [1.0], [4], 1)
    @test_throws ArgumentError ddx!(zeros(4), zeros(4), small, 1; op = FD(9))
end

@testset "FD operators: convergence with smooth halos" begin
    # Functions whose extension by the boundary condition is smooth converge at order width - 1.
    function maxerr(n, op, f, df)
        grid = Grid([0.0], [1.0], [n], 1)
        x = collect(grid.xaxes[1])
        L = x[end] - x[1]
        out = zeros(n)
        ddx!(out, f.(x, L), grid, 1; op)
        return maximum(abs.(out .- df.(x, L)))
    end
    cases = (
        (:periodic, (x, L) -> sin(2pi * x), (x, L) -> 2pi * cos(2pi * x)),
        (:mirror_even, (x, L) -> cos(pi * x / L), (x, L) -> -pi / L * sin(pi * x / L)),
        (:mirror_odd, (x, L) -> sin(pi * x / L), (x, L) -> pi / L * cos(pi * x / L)),
    )
    for (bc, f, df) in cases, width in (3, 5, 9)
        e1 = maxerr(32, FD(width, bc), f, df)
        e2 = maxerr(64, FD(width, bc), f, df)
        order = log2(e1 / e2)
        @test order > width - 1.3
    end
end

@testset "FD operators: bsl6d stencil, bit-identical to run_f0.jl ddx1_9!" begin
    sz = (16, 6, 5)
    grid = Grid([0.0, 0.0, 0.0], [3.7, 2pi, 2pi], collect(sz), 3)
    phi = randn(sz...)
    ref = similar(phi)
    ddx1_9!(ref, phi, grid.delta[1])
    out = similar(phi)
    ddx!(out, phi, grid, 1; op = FD(9, :mirror_even))
    @test out == ref
    ddx!(out, phi, grid, 1; op = FD(9, :mirror_even), negate = true)
    @test out == -ref                     # E = -dphi/dx1 as in the script (ddx1_9! then .*= -1)
end

@testset "FD operators: mixed gradient" begin
    N = (32, 16, 8)
    grid = Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], collect(N), 3)
    xs, ys, zs = collect.(grid.xaxes)
    phi = ScalarField([sin(x) * cos(2y) * sin(z) for x in xs, y in ys, z in zs])
    ops = (FD(9, :periodic), Spectral(), Spectral())
    g = PlasmaCore.grad(phi, grid, ops)
    gs = grad(phi, grid)
    @test g[1].data ≈ gs[1].data atol = 1e-5           # FD error on the x1 axis
    @test g[2].data == gs[2].data                      # spectral axes identical to grad(field, grid)
    @test g[3].data == gs[3].data
    E = PlasmaCore.zero_vectorfield3(grid)
    grad!(E, phi, grid, ops; negate = true)
    @test E[1].data == -g[1].data
    @test E[2].data == -gs[2].data
    @test_throws ArgumentError PlasmaCore.grad(phi, grid, (FD(3), Spectral()))
end

@testset "FD operators: Float32" begin
    grid = Grid([0.0], [1.0], [20], 1)
    f = Float32.(sin.(2pi .* collect(grid.xaxes[1])))
    out = similar(f)
    ddx!(out, f, grid, 1; op = FD(9))
    @test eltype(out) == Float32
    @test out ≈ Float32.(2pi .* cos.(2pi .* collect(grid.xaxes[1]))) rtol = 1e-3
end
