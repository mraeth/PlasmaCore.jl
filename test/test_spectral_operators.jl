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
