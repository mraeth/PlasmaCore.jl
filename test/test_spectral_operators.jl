@testset "Spectral operators (grad/div/curl) — analytic check" begin
    for N in (16, 32)
        grid = Grid([0.0, 0.0], [2pi, 2pi], [N, N], 2)
        xs = collect(grid.xaxes[1])
        ys = collect(grid.xaxes[2])
        Xg = [x for x in xs, y in ys]
        Yg = [y for x in xs, y in ys]

        phi = ScalarField(sin.(Xg) .* cos.(Yg))

        gradphi = grad(phi, grid)
        @test gradphi isa VectorField
        @test length(gradphi) == 2
        @test isapprox(gradphi[1].data, cos.(Xg) .* cos.(Yg); atol = 1e-9)
        @test isapprox(gradphi[2].data, -sin.(Xg) .* sin.(Yg); atol = 1e-9)
        @test ncomponents(gradphi) == 2

        lap = div(gradphi, grid)
        @test lap isa ScalarField
        @test isapprox(lap.data, -2 .* sin.(Xg) .* cos.(Yg); atol = 1e-9)

        c = curl(gradphi, grid)
        @test c isa ScalarField
        @test isapprox(c.data, zeros(N, N); atol = 1e-9)
    end
end

using FFTW: fft, ifft

@testset "fourier_filter!" begin
    N = (16, 12, 8)
    grid = Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], collect(N), 3)
    data = randn(N...)

    # explicit definition: 3D FFT, zero the bins, inverse
    function explicit(a, remove)
        ahat = fft(a)
        for (axis, bins) in remove, b in bins
            idx = ntuple(d -> d == axis ? b + 1 : Colon(), 3)
            ahat[idx...] .= 0
        end
        return real(ifft(ahat))
    end

    remove = (1 => 7:9, 2 => 5:7)
    f = ScalarField(copy(data))
    @test PlasmaCore.fourier_filter!(f, grid; remove) === f
    @test isapprox(f.data, explicit(data, remove); atol = 1e-13)

    # idempotent, and the removed bins are empty afterwards
    g = copy(f.data)
    PlasmaCore.fourier_filter!(f, grid; remove)
    @test isapprox(f.data, g; atol = 1e-13)
    fh = fft(f.data)
    @test maximum(abs, fh[8:10, :, :]) < 1e-12
    @test maximum(abs, fh[:, 6:8, :]) < 1e-12

    # untouched when the removed bins carry nothing: a pure low mode
    x = collect(grid.xaxes[1])
    low = [sin(x[i]) for i in 1:N[1], j in 1:N[2], k in 1:N[3]]
    h = copy(low)
    PlasmaCore.fourier_filter!(h, grid, (1 => 7:9,))
    @test isapprox(h, low; atol = 1e-13)

    # removing the mean bin on one axis
    m = PlasmaCore.fourier_filter!(copy(data), grid, (3 => 0:0,))
    @test isapprox(m, explicit(data, (3 => 0:0,)); atol = 1e-13)

    @test_throws ArgumentError PlasmaCore.fourier_filter!(copy(data), grid, (4 => 0:0,))
    @test_throws ArgumentError PlasmaCore.fourier_filter!(copy(data), grid, (3 => 7:9,))
end
